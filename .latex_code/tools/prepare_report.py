"""Extract report figures and validate saved evidence; never modify contents/."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
ASSETS = BASE / "assets" / "report"
TABLES = BASE / "tables"
EVIDENCE = BASE / "evidence"


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def read_csv(path):
    return rows(path.read_bytes())


def write_table(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-10, (a, b)


def main():
    for p in (ASSETS, TABLES, EVIDENCE):
        p.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(BASE)).replace("\\", "/"):
               hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((BASE / "contents").rglob("*")) if p.is_file()}
    baseline = EVIDENCE / "contents_sha256.json"
    if baseline.exists():
        assert json.loads(baseline.read_text()) == sources, "contents/ changed"
    else:
        baseline.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    assets = []
    for doc in sorted((BASE / "contents").rglob("*.docx")):
        with ZipFile(doc) as z:
            for name in z.namelist():
                if not name.startswith("word/media/"):
                    continue
                target = ASSETS / (doc.parent.name + "_" + Path(name).name)
                target.write_bytes(z.read(name))
                assets.append({"document": str(doc.relative_to(BASE)).replace("\\", "/"),
                               "member": name, "asset": target.name,
                               "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    assert len(assets) == 12
    (EVIDENCE / "figure_sources.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")

    evaluation = ROOT / ".eval_demo_package/results/eval"
    metrics = read_csv(evaluation / "metrics_table.csv")
    annotations = read_csv(evaluation / "annotation/annotations_final.csv")
    design = read_csv(evaluation / "error_analysis/sample_design.csv")
    weighted = read_csv(evaluation / "error_analysis/category_weighted.csv")
    for name in ("metrics_table.csv", "error_analysis/sample_design.csv",
                 "error_analysis/category_counts.csv", "error_analysis/category_weighted.csv",
                 "error_analysis/ai_agreement.csv", "error_analysis/ai_decisions.csv"):
        (EVIDENCE / Path(name).name).write_bytes((evaluation / name).read_bytes())

    saved = read_csv(ROOT / "results/ml_results.csv") + read_csv(ROOT / "results/dl_results.csv")
    for m, original in zip(metrics, saved, strict=True):
        tn, fp, fn, tp = (int(m[k]) for k in ("tn", "fp", "fn", "tp"))
        assert tn + fp == fn + tp == 12500
        computed = {"accuracy": (tn+tp)/25000, "precision": tp/(tp+fp),
                    "recall": tp/(tp+fn), "f1": 2*tp/(2*tp+fp+fn)}
        for k, value in computed.items():
            close(value, m[k]); close(value, original["test_" + k])

    pred = [r for r in read_csv(ROOT / "results/predictions/distilbert_lora.csv") if r["split"] == "test"]
    assert len(pred) == len({r["review_id"] for r in pred}) == 25000
    cm = Counter((int(r["label"]), int(r["pred"])) for r in pred)
    assert [cm[0,0], cm[0,1], cm[1,0], cm[1,1]] == [11382,1118,741,11759]
    assert all(int(r["pred"]) == int(float(r["score"]) > .5) for r in pred)
    # Ranking AUC with average ranks for ties, independent of third-party packages.
    ranked = sorted((float(r["score"]), int(r["label"])) for r in pred)
    rank_sum = 0.; i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += ((i+1+j)/2) * sum(label for _, label in ranked[i:j])
        i = j
    auc = (rank_sum - 12500*12501/2)/(12500*12500)
    close(auc, metrics[-1]["roc_auc"])

    assert len(annotations) == len({r["id"] for r in annotations}) == 100
    main_counts = Counter(r["main_category"] for r in annotations)
    any_counts = Counter(c for r in annotations for c in r["final_category"].split(";") if c)
    assert main_counts == Counter(mixed=47, other=21, label_noise=17, sarcasm=9, negation=5, long_truncated=1)
    assert any_counts == Counter(mixed=72, other=28, label_noise=20, sarcasm=18, negation=24, long_truncated=2)
    assert sum(len(r["final_category"].split(";")) > 1 for r in annotations) == 62
    assert sum(str(r["unsure"]).lower() in ("true", "1", "yes") for r in annotations) == 9
    assert sorted(int(r["sampled_n"]) for r in design) == [40,60]
    assert sum(int(r["population_N"]) for r in design) == 1840
    assert Counter(r["group"] for r in annotations) == Counter(all5=40, other=60)
    for row in weighted:
        estimate = variance = 0.
        for group, population in (("all5",702),("other",1138)):
            sample = [r for r in annotations if r["group"] == group]
            p = sum(row["category"] == r["main_category"] if row["label_kind"] == "main"
                    else row["category"] in r["final_category"].split(";") for r in sample) / len(sample)
            weight = population / 1840
            estimate += weight * p
            variance += weight**2 * (1-len(sample)/population)*p*(1-p)/(len(sample)-1)
        half = 1.96 * variance**.5
        close(estimate,row["weighted_share"])
        close(max(0.,estimate-half),row["ci95_low"])
        close(min(1.,estimate+half),row["ci95_high"])

    lines=[]
    for m in metrics:
        vals = " & ".join(f"{float(m[k]):.4f}" for k in ("accuracy","precision","recall","f1","roc_auc"))
        lines.append(m["model"].replace("LogReg","LR") + " & " + vals + r" \\")
    write_table("test_metrics.tex", lines)
    write_table("confusion_counts.tex", [m["model"].replace("LogReg","LR") + " & " +
                " & ".join(f"{int(m[k]):,}" for k in ("tn","fp","fn","tp","n_errors")) + r" \\" for m in metrics])
    write_table("validation_metrics.tex", [m["model"].replace("LogReg","LR") + " & " +
                " & ".join(f"{float(r['validation_'+k]):.4f}" for k in ("accuracy","precision","recall","f1")) + r" \\"
                for m,r in zip(metrics,saved,strict=True)])
    tuning = read_csv(ROOT / "results/dl_tuning.csv")
    lines=[]
    for r in sorted(tuning, key=lambda r:float(r["val_f1"]), reverse=True):
        c=json.loads(r["config_json"])
        strategy="head + tail" if c["truncation"] == "head_tail" else "head"
        lines.append(f"{c['max_length']}, {strategy} & {c['r']} & {c['target']} & " +
                     f"{c['lr']:.0e} & {r['best_epoch']} & {float(r['val_accuracy']):.4f} & {float(r['val_f1']):.4f}" + r" \\")
    write_table("dl_search.tex",lines)
    write_table("error_categories.tex", [c.replace("_",r"\_") + f" & {main_counts[c]} & {any_counts[c]}" + r" \\"
                for c in ("mixed","other","label_noise","sarcasm","negation","long_truncated")])

    with ZipFile(ROOT / "imdb_outputs.zip") as z:
        meta=json.loads(z.read("metadata.json"))
        for split, spec in meta["exports"].items():
            assert hashlib.sha256(z.read(spec["file"])).hexdigest() == spec["sha256"]
        (EVIDENCE / "dataset_metadata.json").write_bytes(z.read("metadata.json"))
        gold = {r["review_id"]:r["label"] for r in
                (json.loads(line) for line in z.read(meta["exports"]["test"]["file"]).splitlines())}
        assert set(gold) == {r["review_id"] for r in pred}
        assert all(gold[r["review_id"]] == int(r["label"]) for r in pred)
        assert all(gold[r["id"]] == int(r["label"]) for r in annotations)
    (EVIDENCE / "validation_summary.txt").write_text(
        "PASS: contents/ hashes unchanged; 12 original embedded figures extracted.\n"
        "PASS: all five confusion-count tables reproduce Accuracy/Precision/Recall/F1 in root results.\n"
        "PASS: 25,000 distinct DL test predictions reproduce confusion counts and ROC-AUC.\n"
        "PASS: all 100 annotations reproduce counts, uncertainty, strata and weighted estimates/intervals.\n"
        "PASS: all three prepared JSONL files match the SHA-256 values in metadata.json.\n"
        "PASS: DL prediction IDs and labels match the prepared test records; annotated labels also match.\n"
        "LIMIT: four classical per-review prediction files absent; their AUC and shared-error counts\n"
        "are recorded evaluation outputs, not independently regenerated in this check.\n"
        "LIMIT: demo's recorded 12/12 app checks were not rerun by this validation.\n",
        encoding="utf-8")
    print("Report assets prepared; evidence checks passed.")


if __name__ == "__main__":
    main()
