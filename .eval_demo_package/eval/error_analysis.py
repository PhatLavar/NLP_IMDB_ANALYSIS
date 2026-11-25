"""Shared metrics table and error-analysis tables for the Results and Error Analysis sections.

Inputs (all read only):
    results/predictions/preds_*.csv                  team prediction files (id, text, label, pred, prob)
    imdb_outputs/imdb_test_prepared.jsonl            text_clean = what the DL model reads
    results/ml_results.csv, results/dl_results.csv   teammates' own test numbers (cross-check only)
    results/eval/annotation/annotator_*.csv          hand labels (completeness check only)
    results/eval/annotation/annotations_final.csv    merged hand labels (eval/merge_annotations.py)
    results/eval/manual_annotation_sample.csv        sample design columns and the stored heuristic tags
    results/eval/suggestions/llm_suggestions.csv     AI-suggested categories
    results/eval/suggestions/review_log.csv          decision log of the labelling tool
    reports/label_backups/annotator_A_kiet_20261112_051914.csv   labels saved before the AI suggestions were used
                                                                 (one that differs now = revised after the AI)

Outputs:
    results/eval/metrics_table.csv / .md             step 3
    results/eval/error_analysis/*.csv, tables.md     steps 1, 4, 5, 6

Usage (from the repository root):
    python eval/error_analysis.py
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the tokenizer is already in models/hf_cache
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from transformers import AutoTokenizer

from evaluate import check_same_test_set, find_teammate_row, load_preds
from merge_annotations import CATEGORIES, category_problem, read_csv
from tag_errors import tag_reviews

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/eval/error_analysis"
ANN = ROOT / "results/eval/annotation"
SUG = ROOT / "results/eval/suggestions"
PRE_AI_BACKUP = ROOT / "reports/label_backups/annotator_A_kiet_20261112_051914.csv"
MODELS = {  # file stem -> display name, in report order
    "tfidf_logreg": "TF-IDF + LogReg",
    "tfidf_svm": "TF-IDF + LinearSVC",
    "w2v_logreg": "Word2Vec + LogReg",
    "w2v_svm": "Word2Vec + LinearSVC",
    "distilbert_lora": "DistilBERT + LoRA",
}
PROB_KIND = {
    "tfidf_logreg": "probability (LogisticRegression.predict_proba)",
    "tfidf_svm": "uncalibrated score: expit(LinearSVC.decision_function)",
    "w2v_logreg": "probability (LogisticRegression.predict_proba)",
    "w2v_svm": "uncalibrated score: expit(LinearSVC.decision_function)",
    "distilbert_lora": "softmax probability of the fine-tuned model",
}
TOL = 0.0005
N_TEST = 25_000
# DL truncation constants are derived from results/dl_config.csv, not typed in
DL_CFG = pd.read_csv(ROOT / "results/dl_config.csv").iloc[0]
BUDGET = int(DL_CFG["max_length"]) - 2
HEAD = int(round(BUDGET * float(DL_CFG["head_frac"])))
TAIL = BUDGET - HEAD
# a score the author wrote in the text, e.g. "3/10", "7 out of 10", "4 stars" (not the IMDb rating itself)
SCORE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:/|out of)\s*(?:10|5|4)\b|\b\d+(?:\.\d+)?\s*stars?\b", re.I)
HEURISTICS = [("truncated", "long_truncated"), ("neg_cue_strict", "negation"), ("neg_of_negative", "negation"),
              ("neg_of_positive", "negation"), ("negated_sentiment_any", "negation"),
              ("has_contrast_cue", "mixed"), ("mixed_candidate", "mixed")]


def to_md(df, floats=4):
    """Minimal markdown table (tabulate is not installed)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else f"{v:.{floats}f}"
        return str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(df, name, md_parts, title, note=""):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    md_parts.append(f"## {title}\n\nFile: `results/eval/error_analysis/{name}.csv`. {note}\n\n{to_md(df)}\n")


def kappa_band(k):
    """Landis and Koch (1977) wording for a kappa value."""
    if pd.isna(k):
        return "undefined"
    return next(b for lim, b in [(0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                                 (0.8, "substantial"), (1.01, "almost perfect")] if k < lim)


# ---------------------------------------------------------------- step 1 + 3: predictions
def load_all_preds(test):
    dfs = {m: load_preds(ROOT / f"results/predictions/preds_{m}.csv") for m in MODELS}  # schema + pred==prob>0.5
    check_same_test_set(dfs, test.rename(columns={"review_id": "id"}))  # same ids, same labels as the test set
    ref = dfs["distilbert_lora"].set_index("id")
    for m, df in dfs.items():
        assert len(df) == N_TEST, (m, len(df))
        d = df.set_index("id").loc[ref.index]
        assert (d["label"] == ref["label"]).all() and (d["text"] == ref["text"]).all(), m
    assert (test.set_index("review_id").loc[ref.index, "text"] == ref["text"]).all(), "preds text != jsonl text"
    return dfs


def metrics_table(dfs):
    rows = []
    for m, df in dfs.items():
        y, p, s = df["label"], df["pred"], df["prob"]
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append({"model": MODELS[m], "file": f"preds_{m}.csv", "accuracy": accuracy_score(y, p),
                     "precision": precision_score(y, p), "recall": recall_score(y, p), "f1": f1_score(y, p),
                     "roc_auc": roc_auc_score(y, s), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                     "n_errors": int(fp + fn), "prob_column": PROB_KIND[m]})
    return pd.DataFrame(rows)


def crosscheck(table):
    results = pd.concat([pd.read_csv(ROOT / "results/ml_results.csv"), pd.read_csv(ROOT / "results/dl_results.csv")])
    rows = []
    for m, r in zip(MODELS, table.itertuples()):
        row, reason = find_teammate_row(m, results)
        assert row is not None, (m, reason)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mine, theirs = getattr(r, metric), float(row[f"test_{metric}"])
            rows.append({"model": MODELS[m], "metric": f"test_{metric}", "from_preds": mine, "in_results_file": theirs,
                         "diff": mine - theirs, "flag": "DIFF>0.0005" if abs(mine - theirs) > TOL else "ok"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 1c + 4: hand labels
def completeness():
    rows = []
    for path in sorted(ANN.glob("annotator_*.csv")):
        df = read_csv(path)
        problems = [f"{r.id}: {category_problem(r.category)}" for r in df.itertuples() if category_problem(r.category)]
        spaced = int(df["category"].str.contains(r"\s").sum())  # ' ; ' would pass merge but is not the agreed form
        rows.append({"file": path.name, "rows": len(df), "empty_category": int((df["category"].str.strip() == "").sum()),
                     "invalid_category": len(problems), "category_with_whitespace": spaced,
                     "unsure_yes": int((df["unsure"] == "yes").sum()), "problems": "; ".join(problems) or "none"})
    return pd.DataFrame(rows)


def load_labels(dfs):
    final = read_csv(ANN / "annotations_final.csv")
    sample = read_csv(ROOT / "results/eval/manual_annotation_sample.csv")
    assert len(final) == 100 and set(final["id"]) == set(sample["id"])
    final = final.merge(sample.drop(columns=["text", "label", "pred_dl", "category", "annotator", "note"]), on="id")
    err = pd.DataFrame({m: (df.set_index("id")["label"] != df.set_index("id")["pred"]).astype(int)
                        for m, df in dfs.items()})
    final = final.join(err.add_prefix("err_"), on="id")
    final["n_wrong"] = final[[f"err_{m}" for m in MODELS]].sum(axis=1)
    # the stored design columns must agree with the predictions
    assert (final["n_wrong"].astype(str) == final["n_models_wrong"]).all()
    assert (final["err_distilbert_lora"] == 1).all()
    assert ((final["group"] == "all5") == (final["n_wrong"] == 5)).all()
    dl = dfs["distilbert_lora"].set_index("id")
    assert (final["label"].astype(int).values == dl.loc[final["id"], "label"].values).all()
    assert (final["pred_dl"].astype(int).values == dl.loc[final["id"], "pred"].values).all()
    final["cats"] = final["final_category"].str.split(";")
    return final, err


def population(dfs, err):
    """Stratum sizes of the sampling frame, rebuilt as in tag_errors.make_annotation_sample."""
    text = dfs["distilbert_lora"].set_index("id")["text"]
    n_wrong = err.sum(axis=1)
    keep = ~text.duplicated(keep="first")
    return {"all5": int(((n_wrong == 5) & keep).sum()),
            "other": int(((err["distilbert_lora"] == 1) & (n_wrong < 5) & keep).sum()),
            "dl_errors_raw": int(err["distilbert_lora"].sum()),
            "dl_errors_dedup": int(((err["distilbert_lora"] == 1) & keep).sum())}


def has(df, cat, kind):
    return (df["main_category"] == cat) if kind == "main" else df["cats"].map(lambda c: cat in c)


def category_counts(lab):
    slices = {"all": lab, "group=all5": lab[lab["group"] == "all5"], "group=other": lab[lab["group"] == "other"],
              "file=A": lab[lab["annotator"] == "A"], "file=B": lab[lab["annotator"] == "B"]}
    rows = []
    for kind in ["main", "any"]:
        for name, part in slices.items():
            for c in CATEGORIES:
                n = int(has(part, c, kind).sum())
                rows.append({"label_kind": kind, "slice": name, "n_reviews": len(part), "category": c, "n": n,
                             "share": n / len(part)})
    return pd.DataFrame(rows)


def weighted(lab, pop):
    """Stratified estimate of each category's share among DL errors (deduplicated texts), with a 95% normal CI."""
    N = {h: pop[h] for h in ["all5", "other"]}
    W = {h: N[h] / sum(N.values()) for h in N}
    rows = []
    for kind in ["main", "any"]:
        for c in CATEGORIES:
            est, var = 0.0, 0.0
            for h in N:
                x = has(lab[lab["group"] == h], c, kind)
                n, p = len(x), x.mean()
                est += W[h] * p
                var += W[h] ** 2 * (1 - n / N[h]) * p * (1 - p) / (n - 1)
            half = 1.96 * var ** 0.5
            rows.append({"label_kind": kind, "category": c, "unweighted_share": has(lab, c, kind).mean(),
                         "weighted_share": est, "ci95_low": max(0.0, est - half), "ci95_high": min(1.0, est + half)})
    return pd.DataFrame(rows)


def per_model(lab, err):
    dl = err["distilbert_lora"] == 1
    rows = []
    for m in MODELS:
        part = lab[lab[f"err_{m}"] == 1]
        row = {"model": MODELS[m], "errors_on_test": int(err[m].sum()),
               "errors_shared_with_dl": int((err[m].astype(bool) & dl).sum()),
               "share_of_errors_in_frame": float((err[m].astype(bool) & dl).sum() / err[m].sum()),
               "sampled_reviews_wrong": len(part)}
        row.update({f"main_{c}": int((part["main_category"] == c).sum()) for c in CATEGORIES})
        rows.append(row)
    return pd.DataFrame(rows)


def dl_length(lab, test):
    """Wordpieces under the DL rule (budget BUDGET = head HEAD + tail TAIL) for each labelled review."""
    sys.path.insert(0, str(ROOT / "demo"))
    from label_app import dl_view  # same head/tail code the labelling tool used to grey out the unseen part
    tok = AutoTokenizer.from_pretrained(DL_CFG["model_name"], cache_dir=ROOT / "models/hf_cache", local_files_only=True)
    clean = test.set_index("review_id")["text_clean"]
    rows = []
    for r in lab.itertuples():
        ids, kept, dropped = dl_view(clean[r.id], tok, int(DL_CFG["max_length"]), float(DL_CFG["head_frac"]))
        rows.append({"id": r.id, "group": r.group, "main_category": r.main_category, "final_category": r.final_category,
                     "n_wordpieces": len(ids), "truncated": int(len(ids) > BUDGET), "n_unseen": len(ids) - len(kept),
                     "unseen_chars": 0 if dropped is None else dropped[1] - dropped[0],
                     "long_truncated_label": int("long_truncated" in r.cats), "note": r.note,
                     "unseen_middle": "" if dropped is None else clean[r.id][dropped[0]:dropped[1]]})
    return pd.DataFrame(rows)


def label_noise(lab):
    part = lab[lab["cats"].map(lambda c: "label_noise" in c)]
    return pd.DataFrame({
        "id": part["id"], "group": part["group"], "label": part["label"], "pred_dl": part["pred_dl"],
        "prob_dl": part["prob_dl"].astype(float), "main_category": part["main_category"],
        "final_category": part["final_category"],
        "score_written_in_text": part["text"].map(lambda t: "; ".join(SCORE_RE.findall(t)) or "none found"),
        "note": part["note"]})


# ---------------------------------------------------------------- step 5: heuristics
def heuristics(lab, test):
    sub = test[test["review_id"].isin(lab["id"])].rename(columns={"review_id": "id"})
    tags = tag_reviews(sub).set_index("id")
    for col, _ in HEURISTICS:  # recomputed tags must equal the stored ones
        assert (tags.loc[lab["id"], col].astype(str).values == lab[col].values).all(), col
    rows = []
    for col, cat in HEURISTICS:
        for kind in ["any", "main"]:
            y = has(lab, cat, kind).astype(int).values
            h = tags.loc[lab["id"], col].astype(int).values
            tn, fp, fn, tp = confusion_matrix(y, h, labels=[0, 1]).ravel()
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            k = cohen_kappa_score(y, h) if len(set(y)) > 1 or len(set(h)) > 1 else float("nan")
            rows.append({"heuristic": col, "hand_category": cat, "hand_label_kind": kind, "tp": int(tp), "fp": int(fp),
                         "fn": int(fn), "tn": int(tn), "heuristic_fires": int(tp + fp), "hand_prevalence": y.mean(),
                         "precision": prec, "recall": rec,
                         "f1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
                         "lift": prec / y.mean() if y.mean() else float("nan"), "kappa": k, "kappa_band": kappa_band(k)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 6: AI assistance
def ai_assist(lab):
    sug = read_csv(SUG / "llm_suggestions.csv").set_index("id")
    log = read_csv(SUG / "review_log.csv").drop_duplicates("id", keep="last").set_index("id")
    pre_ai = read_csv(PRE_AI_BACKUP)
    pre_ai = pre_ai[pre_ai["category"].str.strip() != ""].set_index("id")["category"]
    cats_now = lab.set_index("id")["cats"]
    # a pre-AI label that differs now was revised later, with the suggestion seen (saved outside the tool UI: no log row)
    revised = {i for i, c in pre_ai.items() if [p.strip() for p in c.split(";")] != list(cats_now[i])}
    pre_ai = set(pre_ai.index)
    assert len(sug) == 100 and set(sug.index) == set(lab["id"])
    rows = []
    for r in lab.itertuples():
        s = sug.loc[r.id]
        ai_also = [x for x in s["also_applies"].split(";") if x]
        fin_also = r.cats[1:]
        main_same = s["main_category"] == r.main_category
        also_same = set(ai_also) == set(fin_also)
        unsure_same = (s["unsure"] == "yes") == (r.unsure == "yes")
        logged = r.id in log.index
        if logged:  # the log's final values must match the file
            lg = log.loc[r.id]
            assert lg["final_main"] == r.main_category and set(filter(None, lg["final_also"].split(";"))) == set(fin_also), r.id
        rows.append({"id": r.id, "file": r.annotator, "ai_main": s["main_category"], "ai_also": s["also_applies"],
                     "ai_unsure": s["unsure"], "final_main": r.main_category, "final_also": ";".join(fin_also),
                     "final_unsure": r.unsure, "main_changed": int(not main_same), "also_changed": int(not also_same),
                     "unsure_changed": int(not unsure_same),
                     "log_decision": log.loc[r.id, "decision"] if logged else "no_log_row",
                     "labelled_before_ai": int(r.id in pre_ai),
                     "revised_after_ai": int(r.id in revised),
                     "derived_decision": "accepted" if main_same and also_same and unsure_same else "edited"})
    df = pd.DataFrame(rows)
    logged = df[df["log_decision"] != "no_log_row"]
    assert (logged["log_decision"] == logged["derived_decision"]).all(), "log decision differs from the file"
    assert set(df.loc[df["log_decision"] == "no_log_row", "id"]) == pre_ai
    return df


def ai_summaries(ai):
    dec = ai["log_decision"].value_counts().rename_axis("decision").reset_index(name="n")
    agree = []
    seen = ai[(ai["labelled_before_ai"] == 0) | (ai["revised_after_ai"] == 1)]
    for name, part in {"all 100 reviews": ai,
                       f"{len(seen)} reviews whose final label was set with the AI suggestion seen": seen}.items():
        agree.append({"comparison": "AI suggestion main vs final main (NOT inter-annotator agreement)", "subset": name,
                      "n": len(part), "n_same": int((part["main_changed"] == 0).sum()),
                      "raw_agreement": (part["main_changed"] == 0).mean(),
                      "cohen_kappa": cohen_kappa_score(part["ai_main"], part["final_main"])})
    per_cat = []
    for c in CATEGORIES:
        part = ai[ai["ai_main"] == c]
        per_cat.append({"ai_main": c, "n_suggested": len(part), "main_kept": int((part["main_changed"] == 0).sum()),
                        "main_changed": int(part["main_changed"].sum()),
                        "main_edit_rate": part["main_changed"].mean() if len(part) else float("nan"),
                        "any_edit": int((part["derived_decision"] == "edited").sum()),
                        "final_still_has_it": int(sum(c in ([f] + a.split(";")) for f, a in
                                                      zip(part["final_main"], part["final_also"])))})
    confusion = pd.crosstab(ai["ai_main"], ai["final_main"]).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    confusion = confusion.rename_axis(None, axis=1).rename_axis("ai_main \\ final_main").reset_index()
    return dec, pd.DataFrame(agree), pd.DataFrame(per_cat), confusion


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Error-analysis tables\n\nWritten by `eval/error_analysis.py`. Every number is recomputed from "
          "`results/predictions/preds_*.csv` and the annotation files.\n"]
    test = pd.read_json(ROOT / "imdb_outputs/imdb_test_prepared.jsonl", lines=True, encoding="utf-8")
    test = test[["review_id", "text", "text_clean", "label"]]

    dfs = load_all_preds(test)
    print(f"Step 1a: {len(dfs)} preds files, schema ok, {N_TEST:,} rows each, same ids/labels/text in every file.")
    table = metrics_table(dfs)
    table.to_csv(ROOT / "results/eval/metrics_table.csv", index=False, encoding="utf-8-sig")
    note = ("ROC-AUC uses only the ranking of `prob`, so it is valid for the SVM scores too. Log loss, Brier score "
            "and calibration are NOT reported: for the two LinearSVC models `prob` = expit(decision_function) is an "
            "uncalibrated score, not a probability.")
    (ROOT / "results/eval/metrics_table.md").write_text(
        "# Test-set metrics (25,000 reviews, positive class = 1)\n\nWritten by `eval/error_analysis.py` from "
        f"`results/predictions/preds_*.csv`. {note}\n\n{to_md(table)}\n", encoding="utf-8")
    print(table.drop(columns=["file", "prob_column"]).to_string(index=False))
    cc = crosscheck(table)
    save(cc, "preflight_metric_crosscheck", md, "Step 1b: metrics from preds vs teammates' results files",
         f"Flag if |diff| > {TOL}.")
    print(cc.to_string(index=False))
    comp = completeness()
    save(comp, "preflight_annotation_completeness", md, "Step 1c: annotation files")
    print(comp.to_string(index=False))
    if (comp[["empty_category", "invalid_category"]].values.sum()):
        sys.exit("STOP: annotation files are not complete.")

    lab, err = load_labels(dfs)
    pop = population(dfs, err)
    design = pd.DataFrame([{"stratum": "all5 (wrong for all 5 models)", "population_N": pop["all5"],
                            "sampled_n": int((lab["group"] == "all5").sum())},
                           {"stratum": "other (DL wrong, not all 5 wrong)", "population_N": pop["other"],
                            "sampled_n": int((lab["group"] == "other").sum())}])
    design["sampling_fraction"] = design["sampled_n"] / design["population_N"]
    design["weight_N_over_n"] = design["population_N"] / design["sampled_n"]
    save(design, "sample_design", md, "Sampling design (recomputed from preds)",
         f"Frame = DL errors with duplicated texts removed (keep first): {pop['dl_errors_dedup']:,} of "
         f"{pop['dl_errors_raw']:,} DL errors. All 100 sampled reviews are DL errors.")
    save(category_counts(lab), "category_counts", md, "Step 4a: category counts in the sample (SAMPLE COUNTS)",
         "label_kind main = first value; any = every value (multi-label, shares can add up to more than 1).")
    save(weighted(lab, pop), "category_weighted", md, "Step 4a: estimated share among all DL errors (WEIGHTED)",
         "Stratified estimate with weights N/n; 95% normal CI with finite-population correction; rough at n = 100.")
    multi = lab["cats"].map(len)
    save(pd.DataFrame([{"reviews": len(lab), "with_2plus_categories": int((multi > 1).sum()),
                        "mean_categories_per_review": multi.mean(), "unsure_yes": int((lab["unsure"] == "yes").sum())}]),
         "multilabel_summary", md, "Step 4a: multi-label summary")
    save(per_model(lab, err), "per_model_categories", md, "Step 4b: main category among each model's sampled errors",
         "SAMPLE COUNTS, not rates: the sample only contains DL errors, so for an ML model it covers only the errors "
         "it shares with DL.")
    lengths = dl_length(lab, test)
    save(lengths.drop(columns=["unseen_middle", "note"]), "dl_wordpieces", md, "Step 4c: DL wordpieces per labelled review",
         f"Rule from results/dl_config.csv: budget {BUDGET} = head {HEAD} + tail {TAIL} (plus [CLS]/[SEP]).")
    lengths[lengths["long_truncated_label"] == 1].to_csv(OUT / "long_truncated_cases.csv", index=False,
                                                         encoding="utf-8-sig")
    save(label_noise(lab), "label_noise_cases", md, "Step 4d: label_noise cases",
         "The IMDb rating is not in the dataset (stanfordnlp/imdb has text and label only); "
         "score_written_in_text is a score the author typed in the review, found by regex.")

    h = heuristics(lab, test)
    save(h, "heuristic_vs_hand", md, "Step 5: heuristic tags vs hand labels (100 DL errors)",
         "Gold = hand label. lift = precision / hand_prevalence. `truncated` was NOT blind: the labelling tool greyed "
         "out the unseen middle.")

    ai = ai_assist(lab)
    ai.to_csv(OUT / "ai_vs_final.csv", index=False, encoding="utf-8-sig")
    dec, agree, per_cat, conf = ai_summaries(ai)
    save(dec, "ai_decisions", md, "Step 6: decisions (last log row per id)",
         "no_log_row = labelled before the AI suggestions were used (present in the 2026-11-12 05:19 backup). "
         "revised_after_ai = 1 in ai_vs_final.csv marks such a label changed later with the suggestion seen "
         "(saved outside the tool UI, so it has no log row).")
    save(agree, "ai_agreement", md, "Step 6: AI suggestion vs final label", "This is NOT inter-annotator agreement.")
    save(per_cat, "ai_edit_rate_by_category", md, "Step 6: edit rate per AI-suggested main category")
    save(conf, "ai_confusion", md, "Step 6: AI main (rows) vs final main (columns)")
    edited = ai[ai["derived_decision"] == "edited"].drop(columns=["derived_decision"])
    save(edited, "ai_edited_cases", md, "Step 6: edited cases")

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nPopulation strata: {pop}")
    print(f"Wrote results/eval/metrics_table.csv/.md and {len(list(OUT.glob('*.csv')))} CSV files + tables.md to {OUT}")


if __name__ == "__main__":
    main()
