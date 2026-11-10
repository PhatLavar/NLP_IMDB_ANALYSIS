"""DistilBERT + LoRA for IMDB sentiment classification (Person 3: DL model).

Two stages, so that the test set is touched exactly once:

    # 1) Hyperparameter search. Trains on `train`, scores on `val` only.
    python dl/train_lora.py --stage tune

    # 2) Final run. Re-trains the best configuration, picks the best epoch on
    #    `val`, then evaluates on `test` once and writes the result files.
    python dl/train_lora.py --stage final

Input : imdb_outputs/imdb_{train,val,test}_prepared.jsonl  (columns used:
        review_id, label, text_clean)
Output: results/dl_tuning.csv          one row per tuning run (val metrics only)
        results/dl_training_log.csv    per-epoch log of every run
        results/dl_results.csv         same columns as ml_results.csv (final run)
        results/dl_config.csv          one-row configuration of the final run
        results/predictions/distilbert_lora.csv
                                       review_id, split, label, pred, score
        <ckpt_dir>/distilbert_lora/    LoRA adapter of the final run

Works locally and on Kaggle (paths are auto-detected, see `--data_dir`).
Tuning is resumable: runs already listed in dl_tuning.csv are skipped, so a
Kaggle session that dies mid-way can simply be started again.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Keep timing comparable and avoid implicit DataParallel on Kaggle's 2xT4.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
IS_KAGGLE = Path("/kaggle/working").exists()
try:
    ROOT = Path(__file__).resolve().parents[1]  # repo root (dl/ is one level down)
except NameError:  # pasted into a notebook cell
    ROOT = Path.cwd()

MODEL_ID = "distilbert_lora"
REPRESENTATION = "DistilBERT (contextual)"
CLASSIFIER = "LoRA fine-tuned"

# Same column order as results/ml_results.csv so that the files can be concatenated.
RESULT_COLUMNS = [
    "representation",
    "classifier",
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "training_time_seconds",
]

# DistilBERT linear layer names.
TARGETS = {
    "qv": ["q_lin", "v_lin"],
    "attn": ["q_lin", "k_lin", "v_lin", "out_lin"],
    "all": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
}


def find_data_dir(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    candidate = ROOT / "imdb_outputs"
    if (candidate / "imdb_train_prepared.jsonl").exists():
        return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = sorted(kaggle_input.rglob("imdb_train_prepared.jsonl"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        "Cannot find imdb_train_prepared.jsonl. Unzip imdb_outputs.zip into the "
        "repository root, attach it as a Kaggle dataset, or pass --data_dir."
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Cfg:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    truncation: str = "head"  # "head": keep the first tokens; "head_tail": first + last
    head_frac: float = 0.25   # share of the token budget kept from the start (head_tail)
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target: str = "qv"        # key of TARGETS
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 4
    warmup: float = 0.06
    weight_decay: float = 0.01
    seed: int = 42
    run_id: str = field(default="")

    def finalize(self) -> "Cfg":
        if not self.run_id:
            self.run_id = (
                f"r{self.r}_{self.target}_lr{self.lr:g}_L{self.max_length}_{self.truncation}"
            )
        return self


def default_runs(base: Cfg) -> list[Cfg]:
    """One-factor-at-a-time search around a centre point (r=8, qv, lr=3e-4, L=256).

    Edit this list to change the search; every entry becomes one tuning run.
    """

    def make(**kw) -> Cfg:
        d = asdict(base)
        d.update(kw)
        d["run_id"] = ""
        return Cfg(**d).finalize()

    return [
        make(),                                               # centre
        make(lr=1e-4),
        make(lr=5e-4),
        make(r=4, alpha=8),
        make(r=16, alpha=32),
        make(target="all"),
        make(max_length=512, truncation="head"),
        make(max_length=512, truncation="head_tail"),
    ]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def log(*a):
    print(*a, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_split(data_dir: Path, name: str, limit: int | None) -> pd.DataFrame:
    path = data_dir / f"imdb_{name}_prepared.jsonl"
    df = pd.read_json(path, lines=True)[["review_id", "label", "text_clean"]]
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def tokenize_full(tokenizer, texts: list[str], chunk: int = 1000) -> list[list[int]]:
    """Token ids without special tokens and without truncation (truncation is
    applied later, so one tokenisation serves every max_length setting)."""
    ids: list[list[int]] = []
    for i in range(0, len(texts), chunk):
        enc = tokenizer(texts[i : i + chunk], add_special_tokens=False, truncation=False)
        ids.extend(enc["input_ids"])
    return ids


def build_inputs(tokenizer, ids: list[int], max_length: int, mode: str, head_frac: float):
    budget = max_length - 2  # [CLS] and [SEP]
    if len(ids) <= budget:
        kept = ids
    elif mode == "head":
        kept = ids[:budget]
    elif mode == "head_tail":
        head = int(round(budget * head_frac))
        tail = budget - head
        kept = ids[:head] + ids[len(ids) - tail :]
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")
    return [tokenizer.cls_token_id] + kept + [tokenizer.sep_token_id]


def encode_split(tokenizer, full_ids, cfg: Cfg):
    return [
        build_inputs(tokenizer, ids, cfg.max_length, cfg.truncation, cfg.head_frac)
        for ids in full_ids
    ]


def collate(encoded, idx, pad_id: int, device):
    longest = max(len(encoded[i]) for i in idx)
    ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), longest), dtype=torch.long)
    for row, i in enumerate(idx):
        seq = encoded[i]
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    return ids.to(device), mask.to(device)


def metrics_from_logits(logits: np.ndarray, y: np.ndarray):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    pred = logits.argmax(axis=1)
    prob_pos = np.exp(logp[:, 1])
    m = {
        "loss": float(-logp[np.arange(len(y)), y].mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    return m, pred, prob_pos


@torch.no_grad()
def predict_logits(model, encoded, pad_id, batch_size, device, use_amp) -> np.ndarray:
    model.eval()
    order = np.argsort([len(e) for e in encoded], kind="stable")  # fast: similar lengths together
    out = np.zeros((len(encoded), 2), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids, mask = collate(encoded, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def build_model(cfg: Cfg, device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    # Loaded in fp32; fp16 is applied through autocast (loading in fp16 breaks
    # the GradScaler on trainable LoRA weights).
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1}
    )
    base_total = sum(p.numel() for p in base.parameters())
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=TARGETS[cfg.target],
        # DistilBERT has TWO randomly-initialised head layers. PEFT only adds
        # `classifier` automatically, so `pre_classifier` must be listed or it
        # would stay frozen at its random initial weights.
        modules_to_save=["pre_classifier", "classifier"],
        bias="none",
    )
    model = get_peft_model(base, lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable, base_total


# --------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------
def train_run(cfg: Cfg, tokenizer, full, labels, device, stage: str):
    """Train on `train`, select the best epoch on `val`.

    Returns (summary, epoch_log, model, val_logits). The returned model has the
    best-epoch weights restored."""
    use_amp = device.type == "cuda"
    set_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    enc_tr = encode_split(tokenizer, full["train"], cfg)
    enc_va = encode_split(tokenizer, full["val"], cfg)
    y_tr, y_va = labels["train"], labels["val"]
    budget = cfg.max_length - 2
    truncated_frac = float(np.mean([len(x) > budget for x in full["train"]]))

    model, trainable, total = build_model(cfg, device)
    log(
        f"[{cfg.run_id}] trainable params {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%), train reviews truncated: {100 * truncated_frac:.1f}%"
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(enc_tr) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warm = max(1, int(total_steps * cfg.warmup))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return step / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.seed)
    pad_id = tokenizer.pad_token_id

    epoch_log, best = [], None
    train_seconds = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sync()
        t0 = time.time()
        order = rng.permutation(len(enc_tr))
        running, seen = 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s : s + cfg.batch_size]
            ids, mask = collate(enc_tr, idx, pad_id, device)
            y = torch.as_tensor(y_tr[idx], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        sync()
        epoch_seconds = time.time() - t0
        train_seconds += epoch_seconds

        val_logits = predict_logits(model, enc_va, pad_id, 128, device, use_amp)
        vm, _, _ = metrics_from_logits(val_logits, y_va)
        row = {
            "run_id": cfg.run_id,
            "stage": stage,
            "epoch": epoch,
            "train_loss": running / seen,
            "val_loss": vm["loss"],
            "val_accuracy": vm["accuracy"],
            "val_f1": vm["f1"],
            "epoch_time_seconds": epoch_seconds,
        }
        epoch_log.append(row)
        log(
            f"[{cfg.run_id}] epoch {epoch}/{cfg.epochs} train_loss {row['train_loss']:.4f} "
            f"val_loss {vm['loss']:.4f} val_acc {vm['accuracy']:.4f} val_f1 {vm['f1']:.4f} "
            f"({epoch_seconds:.0f}s)"
        )
        if best is None or vm["f1"] > best["metrics"]["f1"]:
            best = {
                "epoch": epoch,
                "metrics": vm,
                "logits": val_logits,
                "state": {
                    n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
                },
            }

    with torch.no_grad():  # restore best-epoch weights
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best["state"][n].to(p.device))

    summary = {
        "best_epoch": best["epoch"],
        "val": best["metrics"],
        "train_seconds": train_seconds,
        "peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        ),
        "trainable_params": trainable,
        "total_params": total,
        "truncated_frac_train": truncated_frac,
    }
    return summary, epoch_log, model, best["logits"]


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------
def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def env_info() -> dict:
    import peft
    import transformers

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def prepare(args, need_test: bool):
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    data_dir = find_data_dir(args.data_dir)
    log(f"data dir: {data_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splits = {"train": args.limit_train, "val": args.limit_val}
    if need_test:
        splits["test"] = args.limit_test
    frames = {name: load_split(data_dir, name, lim) for name, lim in splits.items()}
    labels = {k: v["label"].to_numpy() for k, v in frames.items()}
    t0 = time.time()
    full = {k: tokenize_full(tokenizer, v["text_clean"].tolist()) for k, v in frames.items()}
    log(f"tokenised {', '.join(f'{k}={len(v)}' for k, v in frames.items())} in {time.time() - t0:.0f}s")
    return tokenizer, frames, full, labels


def stage_tune(args, device):
    results_dir = Path(args.results_dir)
    tuning_csv = results_dir / "dl_tuning.csv"
    base = Cfg(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    runs = default_runs(base)
    if args.runs:
        wanted = set(args.runs.split(","))
        runs = [c for c in runs if c.run_id in wanted]
    done = set(pd.read_csv(tuning_csv)["run_id"]) if tuning_csv.exists() else set()
    todo = [c for c in runs if c.run_id not in done]
    log(f"{len(runs)} planned runs, {len(runs) - len(todo)} already done, {len(todo)} to run")
    for c in todo:
        log(f"  - {c.run_id}")
    if args.dry_run or not todo:
        return

    tokenizer, _, full, labels = prepare(args, need_test=False)
    env = env_info()
    for cfg in todo:
        summary, epoch_log, model, _ = train_run(cfg, tokenizer, full, labels, device, "tune")
        append_csv(results_dir / "dl_training_log.csv", epoch_log)
        row = {
            "run_id": cfg.run_id,
            "best_epoch": summary["best_epoch"],
            "val_accuracy": summary["val"]["accuracy"],
            "val_precision": summary["val"]["precision"],
            "val_recall": summary["val"]["recall"],
            "val_f1": summary["val"]["f1"],
            "val_loss": summary["val"]["loss"],
            "train_seconds": summary["train_seconds"],
            "peak_gpu_mem_mb": summary["peak_gpu_mem_mb"],
            "trainable_params": summary["trainable_params"],
            "truncated_frac_train": summary["truncated_frac_train"],
            "gpu": env["gpu"],
            "config_json": json.dumps(asdict(cfg)),
        }
        append_csv(tuning_csv, [row])  # written after every run -> resumable
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log(f"\ntuning table: {tuning_csv}")


def choose_final_cfg(args) -> Cfg:
    tuning_csv = Path(args.results_dir) / "dl_tuning.csv"
    if not tuning_csv.exists():
        raise SystemExit(f"{tuning_csv} not found. Run `--stage tune` first.")
    table = pd.read_csv(tuning_csv)
    if args.run_id:
        match = table[table["run_id"] == args.run_id]
        if match.empty:
            raise SystemExit(f"run_id {args.run_id!r} not in {tuning_csv}")
        row = match.iloc[0]
    else:  # best validation F1; ties go to the cheaper run
        row = table.sort_values(["val_f1", "train_seconds"], ascending=[False, True]).iloc[0]
    log(f"final configuration: {row['run_id']} (val_f1 {row['val_f1']:.4f}, best epoch {int(row['best_epoch'])})")
    return Cfg(**json.loads(row["config_json"]))


def stage_final(args, device):
    results_dir = Path(args.results_dir)
    out_results = results_dir / "dl_results.csv"
    if out_results.exists() and not args.force:
        raise SystemExit(
            f"{out_results} already exists. The test set is meant to be evaluated once; "
            "pass --force to overwrite deliberately."
        )
    cfg = choose_final_cfg(args)
    tokenizer, frames, full, labels = prepare(args, need_test=True)
    env = env_info()

    summary, epoch_log, model, val_logits = train_run(cfg, tokenizer, full, labels, device, "final")
    use_amp = device.type == "cuda"
    enc_te = encode_split(tokenizer, full["test"], cfg)
    test_logits = predict_logits(model, enc_te, tokenizer.pad_token_id, 128, device, use_amp)
    tm, test_pred, test_prob = metrics_from_logits(test_logits, labels["test"])
    vm, val_pred, val_prob = metrics_from_logits(val_logits, labels["val"])
    log(f"\nFINAL  val_f1 {vm['f1']:.4f} | test_acc {tm['accuracy']:.4f}  test_f1 {tm['f1']:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "representation": REPRESENTATION,
                "classifier": CLASSIFIER,
                "validation_accuracy": vm["accuracy"],
                "validation_precision": vm["precision"],
                "validation_recall": vm["recall"],
                "validation_f1": vm["f1"],
                "test_accuracy": tm["accuracy"],
                "test_precision": tm["precision"],
                "test_recall": tm["recall"],
                "test_f1": tm["f1"],
                "training_time_seconds": summary["train_seconds"],
            }
        ],
        columns=RESULT_COLUMNS,
    ).to_csv(out_results, index=False)

    config_row = asdict(cfg)
    config_row.pop("run_id")
    config_row.update(
        {
            "tuning_run_id": cfg.run_id,
            "target_modules": ",".join(TARGETS[cfg.target]),
            "best_epoch": summary["best_epoch"],
            "trainable_params": summary["trainable_params"],
            "total_params": summary["total_params"],
            "trainable_pct": 100 * summary["trainable_params"] / summary["total_params"],
            "truncated_frac_train": summary["truncated_frac_train"],
            "peak_gpu_mem_mb": summary["peak_gpu_mem_mb"],
            "fp16": use_amp,
            **env,
        }
    )
    pd.DataFrame([config_row]).to_csv(results_dir / "dl_config.csv", index=False)

    # The final run's per-epoch log replaces any previous "final" rows.
    log_path = results_dir / "dl_training_log.csv"
    if log_path.exists():
        old = pd.read_csv(log_path)
        old[old["stage"] != "final"].to_csv(log_path, index=False)
    append_csv(log_path, epoch_log)

    pred_rows = []
    for split, df, pred, prob in (
        ("val", frames["val"], val_pred, val_prob),
        ("test", frames["test"], test_pred, test_prob),
    ):
        pred_rows.append(
            pd.DataFrame(
                {
                    "review_id": df["review_id"],
                    "split": split,
                    "label": df["label"],
                    "pred": pred.astype(int),
                    "score": prob,  # P(positive) from softmax
                }
            )
        )
    pred_path = results_dir / "predictions" / f"{MODEL_ID}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows, ignore_index=True).to_csv(pred_path, index=False)

    ckpt = Path(args.ckpt_dir) / MODEL_ID
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    log(f"wrote: {out_results}, dl_config.csv, dl_training_log.csv, {pred_path}, adapter -> {ckpt}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["tune", "final"], required=True)
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32, help="lower to 16 if a 512-token run runs out of memory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", default="", help="tune: comma-separated run_ids to run (default: all)")
    p.add_argument("--run_id", default="", help="final: run_id from dl_tuning.csv (default: best val F1)")
    p.add_argument("--dry_run", action="store_true", help="tune: only list the planned runs")
    p.add_argument("--force", action="store_true", help="final: overwrite an existing dl_results.csv")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--limit_train", type=int, default=None, help="debug: subsample train")
    p.add_argument("--limit_val", type=int, default=None, help="debug: subsample val")
    p.add_argument("--limit_test", type=int, default=None, help="debug: subsample test")
    args = p.parse_args(argv)
    if not args.results_dir:
        args.results_dir = "/kaggle/working/results" if IS_KAGGLE else str(ROOT / "results")
    if not args.ckpt_dir:
        args.ckpt_dir = "/kaggle/working/checkpoints" if IS_KAGGLE else str(ROOT / "dl" / "checkpoints")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if args.stage == "tune":
        stage_tune(args, device)
    else:
        stage_final(args, device)


if __name__ == "__main__":
    main()
