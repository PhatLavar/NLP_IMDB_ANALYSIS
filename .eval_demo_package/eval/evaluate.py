"""Evaluate prediction files in the team format and compare the models.

Input:  results/predictions/preds_<model>.csv  with columns id, text, label, pred, prob
        (prob = probability of the POSITIVE class, labels: 0 = negative, 1 = positive)

Output (results/eval/):
    metrics_comparison.csv / .md   one row per model (accuracy, precision, recall, F1, errors)
    confusion_<model>.png          confusion matrix with counts and row percentages
    misclassified_<model>.csv      all wrong predictions, most confident mistakes first
    cross_model_summary.md         reviews wrong for ALL models / for only one model
    hard_cases.csv                 the reviews wrong for ALL models

Usage (from the repository root):
    python eval/evaluate.py                                              # all preds_*.csv
    python eval/evaluate.py results/predictions/preds_distilbert_lora.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["id", "text", "label", "pred", "prob"]
METRICS = ["accuracy", "precision", "recall", "f1"]
TOLERANCE = 0.001  # max allowed difference from the teammates' own numbers

# Words in a model name -> start of the "representation" / "classifier" cell in
# results/ml_results.csv and results/dl_results.csv (only used for the cross-check).
REPRESENTATION_WORDS = {"tfidf": "TF-IDF", "word2vec": "Word2Vec", "w2v": "Word2Vec", "distilbert": "DistilBERT"}
CLASSIFIER_WORDS = {"logreg": "Logistic Regression", "lr": "Logistic Regression",
                    "logistic": "Logistic Regression", "svm": "SVM", "lora": "LoRA"}


def model_name(path):
    """preds_distilbert_lora.csv -> distilbert_lora"""
    return Path(path).stem.removeprefix("preds_")


def load_preds(path):
    """Read one prediction file and check it follows the team format.

    Stops with a clear error message if anything is wrong.
    """
    path = Path(path)
    if not path.is_file():
        sys.exit(f"ERROR: file not found: {path}")
    # utf-8-sig: preds files are written with a BOM (for Excel); this keeps it out of the first column name
    df = pd.read_csv(path, dtype={"id": str}, encoding="utf-8-sig")

    if sorted(df.columns) != sorted(COLUMNS):
        sys.exit(f"ERROR in {path.name}: columns must be exactly {COLUMNS}, found {list(df.columns)}")

    problems = []
    n_empty = df[["id", "label", "pred", "prob"]].isna().any(axis=1).sum()
    if n_empty:
        problems.append(f"{n_empty} rows have an empty id/label/pred/prob")
    dupes = df.loc[df["id"].duplicated(), "id"]
    if len(dupes):
        problems.append(f"{dupes.nunique()} ids appear more than once, e.g. {dupes.iloc[0]}")
    for col in ["label", "pred"]:
        bad = df.loc[~df[col].isin([0, 1]), col]
        if len(bad):
            problems.append(f"'{col}' must be 0 or 1, found {bad.iloc[0]!r} ({len(bad)} rows)")
    prob = pd.to_numeric(df["prob"], errors="coerce")
    bad = df[~prob.between(0, 1)]
    if len(bad):
        problems.append(f"'prob' must be a number in [0, 1], e.g. id {bad['id'].iloc[0]} "
                        f"has {bad['prob'].iloc[0]!r} ({len(bad)} rows)")
    if problems:
        sys.exit(f"ERROR in {path.name}:\n  - " + "\n  - ".join(problems))

    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["prob"] = prob

    mismatch = df[df["pred"] != (df["prob"] > 0.5).astype(int)]
    if len(mismatch):
        examples = "; ".join(f"{r.id} (pred={r.pred}, prob={r.prob:.4f})" for r in mismatch.head(3).itertuples())
        sys.exit(f"ERROR in {path.name}: {len(mismatch)} rows where pred != (prob > 0.5), e.g. {examples}.\n"
                 "  pred must be 1 when prob > 0.5 and 0 otherwise (prob = P(positive)).")
    return df


def check_same_test_set(dfs, test):
    """Every file must have exactly the test-set ids, with the same true label for each id.

    Checking each file against the test jsonl also means all files share one id set.
    """
    ref = test.set_index("id")["label"]
    for name, df in dfs.items():
        labels = df.set_index("id")["label"]
        missing = ref.index.difference(labels.index)
        extra = labels.index.difference(ref.index)
        if len(missing) or len(extra):
            sys.exit(f"ERROR in preds_{name}.csv: ids differ from the test set "
                     f"({len(missing)} test ids missing, e.g. {list(missing[:3])}; "
                     f"{len(extra)} ids not in the test set, e.g. {list(extra[:3])}).\n"
                     f"  Every model must predict all {len(ref):,} test reviews, using review_id as id.")
        diff = labels[labels.values != ref.loc[labels.index].values]
        if len(diff):
            sys.exit(f"ERROR in preds_{name}.csv: true label differs from the test set "
                     f"for {len(diff)} ids, e.g. {diff.index[0]}.")


def compute_metrics(df):
    y, p = df["label"], df["pred"]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, p),
        # precision / recall / F1 are for the positive class (label 1)
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "n_errors": int(fp + fn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def save_confusion(df, name, path):
    """Confusion matrix: counts, and in brackets the % of each true class (row-normalized)."""
    cm = confusion_matrix(df["label"], df["pred"], labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified(df, test, path):
    """All wrong predictions, most confident mistakes first. Text comes from the test jsonl."""
    wrong = df.loc[df["label"] != df["pred"], ["id", "label", "pred", "prob"]]
    wrong = wrong.merge(test[["id", "text", "word_count", "token_count_original"]], on="id")
    # confidence = probability the model gave to the class it predicted
    wrong["confidence"] = wrong["prob"].where(wrong["pred"] == 1, 1 - wrong["prob"])
    cols = ["id", "text", "label", "pred", "prob", "confidence", "word_count", "token_count_original"]
    wrong.sort_values("confidence", ascending=False)[cols].to_csv(path, index=False, encoding="utf-8-sig")


def cross_model_summary(dfs, test, out_dir):
    """Count reviews wrong for all models / only one model and save the hard cases."""
    wrong = {}
    for name, df in dfs.items():
        d = df.set_index("id")
        wrong[name] = d["label"] != d["pred"]
    n_wrong = pd.DataFrame(wrong).sum(axis=1)  # per review id: how many models got it wrong

    hard = test.loc[test["id"].isin(n_wrong.index[n_wrong == len(dfs)]),
                    ["id", "text", "label", "word_count", "token_count_original"]]
    for name, df in dfs.items():
        hard = hard.merge(df[["id", "prob"]].rename(columns={"prob": f"prob_{name}"}), on="id")
    hard.to_csv(out_dir / "hard_cases.csv", index=False, encoding="utf-8-sig")  # BOM for Excel

    lines = [
        f"- Models compared: {len(dfs)} ({', '.join(dfs)})",
        f"- Reviews wrong for ALL models: {len(hard):,} (saved in hard_cases.csv)",
        f"- Reviews wrong for exactly one model: {(n_wrong == 1).sum():,}",
    ]
    if len(dfs) == 1:
        lines.append("- Note: with only one model, both counts are just that model's errors.")
    text = "# Cross-model summary\n\n" + "\n".join(lines) + "\n"
    (out_dir / "cross_model_summary.md").write_text(text, encoding="utf-8")
    return text


def find_teammate_row(name, results):
    """Return (row, None) if exactly one row of the teammates' results matches the model
    name, else (None, reason to skip)."""
    words = name.lower().split("_")
    reps = {REPRESENTATION_WORDS[w] for w in words if w in REPRESENTATION_WORDS}
    clfs = {CLASSIFIER_WORDS[w] for w in words if w in CLASSIFIER_WORDS}
    if len(reps) != 1 or len(clfs) != 1:
        return None, "cannot tell representation + classifier from the model name"
    rep, clf = reps.pop(), clfs.pop()
    rows = results[results["representation"].str.startswith(rep) & results["classifier"].str.startswith(clf)]
    if len(rows) != 1:
        return None, f"{len(rows)} rows match '{rep}' + '{clf}'"
    return rows.iloc[0], None


def check_against_teammates(metrics):
    """Compare my test accuracy / F1 with the numbers in results/dl_results.csv and ml_results.csv."""
    files = [f for f in [ROOT / "results/dl_results.csv", ROOT / "results/ml_results.csv"] if f.is_file()]
    if not files:
        print("\nNo results/dl_results.csv or results/ml_results.csv found, skipping the check.")
        return
    results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    print(f"\nCheck against teammates' own test numbers (flag if difference > {TOLERANCE}):")
    print(f"  {'model':<22}{'metric':<15}{'mine':>9}{'theirs':>9}{'diff':>10}")
    for name, m in metrics.items():
        row, reason = find_teammate_row(name, results)
        if row is None:
            print(f"  {name:<22}skipped: {reason}")
            continue
        print(f"  {name:<22}matched row: {row['representation']} + {row['classifier']}")
        for metric in ["accuracy", "f1"]:
            mine, theirs = m[metric], row[f"test_{metric}"]
            flag = "DIFFERENT" if abs(mine - theirs) > TOLERANCE else "ok"
            print(f"  {'':<22}{'test_' + metric:<15}{mine:>9.5f}{theirs:>9.5f}{mine - theirs:>+10.6f}  {flag}")


def to_markdown(table):
    """Markdown table: metrics with 4 decimals, counts as integers."""
    cols = list(table.columns)
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for model, row in table.iterrows():
        cells = [f"{row[c]:.4f}" if c in METRICS else f"{int(row[c]):,}" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate team prediction files (preds_<model>.csv).")
    parser.add_argument("files", nargs="*", type=Path,
                        help="prediction files (default: all results/predictions/preds_*.csv)")
    parser.add_argument("--test", type=Path, default=ROOT / "imdb_outputs/imdb_test_prepared.jsonl",
                        help="test jsonl (default: imdb_outputs/imdb_test_prepared.jsonl)")
    parser.add_argument("--out", type=Path, default=ROOT / "results/eval", help="output folder (default: results/eval)")
    args = parser.parse_args()

    files = args.files or sorted((ROOT / "results/predictions").glob("preds_*.csv"))
    if not files:
        sys.exit("ERROR: no results/predictions/preds_*.csv files found. Run eval/convert_dl_preds.py first.")
    if not args.test.is_file():
        sys.exit(f"ERROR: test file not found: {args.test}\n  Unzip imdb_outputs.zip into imdb_outputs/ first.")

    dfs = {}
    for path in files:
        name = model_name(path)
        if name in dfs:
            sys.exit(f"ERROR: two files give the same model name '{name}'")
        dfs[name] = load_preds(path)

    test = pd.read_json(args.test, lines=True)[["review_id", "text", "label", "token_count_original"]]
    test = test.rename(columns={"review_id": "id"})
    test["word_count"] = test["text"].str.split().str.len()
    check_same_test_set(dfs, test)
    print(f"Loaded {len(dfs)} prediction file(s); all match the {len(test):,}-review test set.")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, df in dfs.items():
        metrics[name] = compute_metrics(df)
        save_confusion(df, name, args.out / f"confusion_{name}.png")
        save_misclassified(df, test, args.out / f"misclassified_{name}.csv")

    table = pd.DataFrame.from_dict(metrics, orient="index")
    table.index.name = "model"
    table.to_csv(args.out / "metrics_comparison.csv")
    (args.out / "metrics_comparison.md").write_text(to_markdown(table), encoding="utf-8")

    print("\nTest-set metrics (positive class = 1):")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n" + cross_model_summary(dfs, test, args.out))
    check_against_teammates(metrics)
    print(f"\nSaved all outputs to {args.out}")


if __name__ == "__main__":
    main()
