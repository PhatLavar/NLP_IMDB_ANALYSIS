"""Re-derive every number in reports/results_error_analysis_draft.md from the source files and assert equality.

Independent of eval/error_analysis.py: it reads the prediction files, the two annotator files (not the merged
file), the AI suggestions, the decision log, the configs and the test jsonl, and recomputes everything itself.

Checks:
    1. every claim below, rendered from recomputed values, appears verbatim in the draft;
    2. after removing those claims, no other number is left in the draft (only a short allowlist of literals);
    3. key tables written by eval/error_analysis.py agree with the recomputed values;
    4. the two hand-label files still have the recorded SHA-256 (see LABEL_SHA256).

Usage (from the repository root):
    python eval/verify_draft.py
"""
import hashlib
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer
from transformers import logging as hf_logging

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "reports/results_error_analysis_draft.md"
EA = ROOT / "results/eval/error_analysis"
LABEL_SHA256 = {  # recorded 2026-11-16; A updated 2026-11-17 after test_04366 was revised (mixed -> label_noise;mixed)
    # A before that revision: 995a55fd6b271d2329bbae0903ceb5b6a1fb57374d0aeede9994fbef471e082b
    # (= reports/label_backups/annotator_A_kiet_20261117_213837.csv; only test_04366's category differs)
    "annotator_A_kiet.csv": "5e936e7e2464c45867c54e9556d0f7516725a105db681b6e2636d4b6ea009d6a",
    "annotator_B_Phat.csv": "a6e54a1eb7194d382c267a67dc27c660143fc7b28e4f9142c8e63debc0de86c4",
}
MODELS = [("tfidf_logreg", "TF-IDF + LogReg"), ("tfidf_svm", "TF-IDF + LinearSVC"),
          ("w2v_logreg", "Word2Vec + LogReg"), ("w2v_svm", "Word2Vec + LinearSVC"),
          ("distilbert_lora", "DistilBERT + LoRA")]
CATS = ["sarcasm", "negation", "mixed", "long_truncated", "label_noise", "other"]
# literals that are not data from this project's files (checked by eye, listed in the final report)
ALLOWED_LITERALS = ["Kiet, 17/11", "positive class = 1", "Table 1",
                    "1-10 ratings (4 or lower = negative, 7 or higher = positive, 5-6 excluded; Maas et al., 2011)"]

claims, results = [], []


def claim(text):
    claims.append(text)


def check(name, ok):
    results.append((name, bool(ok)))


def csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def labels_from_files():
    parts = []
    for name in LABEL_SHA256:
        df = csv(ROOT / "results/eval/annotation" / name, dtype=str, keep_default_na=False)
        parts.append(df.assign(file=name[10]))
    lab = pd.concat(parts, ignore_index=True)
    lab["cats"] = lab["category"].map(lambda s: [p.strip() for p in s.split(";")])
    lab["main"] = lab["cats"].str[0]
    key = csv(ROOT / "results/eval/annotation/_split_key.csv", dtype=str)
    return lab.merge(key[["id", "group"]], on="id", validate="one_to_one")


def main():
    draft = DRAFT.read_text(encoding="utf-8")
    test = pd.read_json(ROOT / "imdb_outputs/imdb_test_prepared.jsonl", lines=True).set_index("review_id")

    # ---------------------------------------------------------------- Results
    m = {}
    for stem, name in MODELS:
        df = csv(ROOT / f"results/predictions/preds_{stem}.csv", dtype={"id": str}).set_index("id")
        y, p = df["label"], df["pred"]
        tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        m[stem] = dict(name=name, n=len(df), acc=(tp + tn) / len(df), prec=tp / (tp + fp), rec=tp / (tp + fn),
                       auc=roc_auc_score(y, df["prob"]), fp=fp, fn=fn, err=fp + fn, wrong=(y != p), y=y)
        m[stem]["f1"] = 2 * m[stem]["prec"] * m[stem]["rec"] / (m[stem]["prec"] + m[stem]["rec"])
        r = m[stem]
        claim(f"| {name} | {r['acc']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | {r['f1']:.4f} | {r['auc']:.4f} "
              f"| {r['fp']:,} | {r['fn']:,} |")
    dl, svm = m["distilbert_lora"], m["tfidf_svm"]
    ref_ids = dl["y"].index
    check("all preds files: 25,000 rows, same ids and labels", all(
        r["n"] == 25_000 and r["y"].index.equals(ref_ids) and r["y"].equals(dl["y"]) for r in m.values()))
    claim(f"same {dl['n']:,} test reviews")
    classical = [s for s, _ in MODELS[:4]]
    check("DL best on every metric", all(dl[k] > m[s][k] for s in classical for k in ["acc", "prec", "rec", "f1", "auc"]))
    check("TF-IDF + LinearSVC is the best classical model by accuracy", max(classical, key=lambda s: m[s]["acc"]) == "tfidf_svm")
    claim(f"Its accuracy is {dl['acc'] - svm['acc']:.4f} higher than the best classical model, TF-IDF + LinearSVC")
    claim(f"makes {svm['err'] - dl['err']:,} fewer errors ({dl['err']:,} vs {svm['err']:,})")
    check("TF-IDF beats Word2Vec with both classifiers",
          m["tfidf_logreg"]["acc"] > m["w2v_logreg"]["acc"] and m["tfidf_svm"]["acc"] > m["w2v_svm"]["acc"])
    claim(f"changes accuracy by only {abs(m['tfidf_svm']['acc'] - m['tfidf_logreg']['acc']):.4f} (TF-IDF) and "
          f"{abs(m['w2v_svm']['acc'] - m['w2v_logreg']['acc']):.4f} (Word2Vec)")
    claim(f"more false positives than false negatives ({dl['fp']:,} vs {dl['fn']:,}), so its recall ({dl['rec']:.4f}) "
          f"is higher than its precision ({dl['prec']:.4f})")
    imbalance = {s: abs(r["fp"] - r["fn"]) / r["err"] for s, r in m.items()}
    check("classical models closer to FP/FN balance than DL", all(imbalance[s] < imbalance["distilbert_lora"] for s in classical))
    teams = pd.concat([pd.read_csv(ROOT / "results/ml_results.csv"), pd.read_csv(ROOT / "results/dl_results.csv")])
    rows = {("TF-IDF", "Logistic"): "tfidf_logreg", ("TF-IDF", "SVM"): "tfidf_svm", ("Word2Vec", "Logistic"): "w2v_logreg",
            ("Word2Vec", "SVM"): "w2v_svm", ("DistilBERT", "LoRA"): "distilbert_lora"}
    diffs = []
    for (rep, clf), stem in rows.items():
        row = teams[teams["representation"].str.startswith(rep) & teams["classifier"].str.startswith(clf)]
        assert len(row) == 1, (rep, clf)
        for k, col in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "f1")]:
            diffs.append(abs(m[stem][k] - float(row[f"test_{col}"].iloc[0])))
    claim(f"the largest absolute difference is {max(diffs):.6f}")
    wrong = pd.DataFrame({s: r["wrong"] for s, r in m.items()})
    n_wrong = wrong.sum(axis=1)
    claim(f"{int((n_wrong == 5).sum()):,} test reviews are misclassified by all five models")
    check("cross_model_summary.md agrees on all-five-wrong count",
          f"Reviews wrong for ALL models: {int((n_wrong == 5).sum()):,}" in
          (ROOT / "results/eval/cross_model_summary.md").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- sample design
    lab = labels_from_files()
    claim(f"We labelled {len(lab)} reviews")
    text = csv(ROOT / "results/predictions/preds_distilbert_lora.csv", dtype={"id": str}).set_index("id")["text"]
    keep = ~text.duplicated(keep="first")
    dl_wrong = wrong["distilbert_lora"]
    N = {"all5": int(((n_wrong == 5) & keep).sum()), "other": int((dl_wrong & (n_wrong < 5) & keep).sum())}
    n = {g: int((lab["group"] == g).sum()) for g in N}
    check("every sampled review is a DL error", dl_wrong.loc[lab["id"]].all())
    check("group all5 == wrong for all five", ((lab["group"] == "all5") == (n_wrong.loc[lab["id"]].values == 5)).all())
    check("tag_errors draws the sample with seed 42", "seed=42" in (ROOT / "eval/tag_errors.py").read_text(encoding="utf-8"))
    claim("(seed 42)")
    claim(f"({int((dl_wrong & keep).sum()):,} of {int(dl_wrong.sum()):,} errors)")
    claim(f"{n['all5']} of the {N['all5']:,} reviews misclassified by all five models")
    claim(f"{n['other']} of the {N['other']:,} reviews misclassified by DistilBERT but not by all five models")
    claim(f"({n['all5'] / N['all5']:.4f} and {n['other'] / N['other']:.4f})")
    claim(f"({N['all5']:,} after removing duplicated texts, the population used for sampling)")
    W = {g: N[g] / sum(N.values()) for g in N}
    est = var = 0.0
    for g in N:
        x = (lab.loc[lab["group"] == g, "main"] == "mixed").astype(float)
        est += W[g] * x.mean()
        var += W[g] ** 2 * (1 - n[g] / N[g]) * x.var(ddof=1) / n[g]
    k_mixed = int((lab["main"] == "mixed").sum())
    claim(f"`mixed` as main category is {k_mixed} of {len(lab)} in the sample, and the weighted estimate is {est:.4f} "
          f"with a 95% interval of {est - 1.96 * var ** 0.5:.4f} to {est + 1.96 * var ** 0.5:.4f}")
    claim(f"With n = {len(lab)} those estimates")
    claim(f"wide intervals at n = {len(lab)}")

    # ---------------------------------------------------------------- categories
    main_n = {c: int((lab["main"] == c).sum()) for c in CATS}
    any_n = {c: int(lab["cats"].map(lambda v: c in v).sum()) for c in CATS}
    for c in CATS:
        claim(f"| {c} | {main_n[c]} | {any_n[c]} |")
    claim(f"`mixed` is the main reason for {main_n['mixed']} of {len(lab)} errors and appears in {any_n['mixed']}")
    claim(f"{int((lab['cats'].map(len) > 1).sum())} of {len(lab)} reviews have two or more categories")
    claim(f"common as a secondary reason ({any_n['negation']}) but rarely the main one ({main_n['negation']})")

    def by_group(c):
        a = int(((lab["group"] == "all5") & (lab["main"] == c)).sum())
        o = int(((lab["group"] == "other") & (lab["main"] == c)).sum())
        return f"{a} of {n['all5']} `all5` reviews but {o} of {n['other']} `other` reviews"
    claim(f"`label_noise` is the main category for {by_group('label_noise')}")
    claim(f"Sarcasm goes the other way: main category for {by_group('sarcasm')}")
    halves = {f: lab[lab["file"] == f] for f in ("A", "B")}
    check("halves A and B have the same size", len(halves["A"]) == len(halves["B"]))
    claim(f"`mixed` is the main category ({int((halves['A']['main'] == 'mixed').sum())} vs "
          f"{int((halves['B']['main'] == 'mixed').sum())} of {len(halves['A'])})")
    pct = [f"{float((wrong[s] & dl_wrong).sum() / wrong[s].sum()):.0%}" for s in classical]
    claim(f"Only {pct[0]} of TF-IDF + LogReg's test errors ({pct[1]} for TF-IDF + LinearSVC, {pct[2]} for "
          f"Word2Vec + LogReg and {pct[3]} for Word2Vec + LinearSVC) are also DistilBERT errors")

    # ---------------------------------------------------------------- truncation (independent head/tail split)
    cfg = pd.read_csv(ROOT / "results/dl_config.csv").iloc[0]
    budget = int(cfg["max_length"]) - 2
    head = int(round(budget * float(cfg["head_frac"])))
    tail = budget - head
    check("dl_config truncation is head_tail", cfg["truncation"] == "head_tail")
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], cache_dir=ROOT / "models/hf_cache", local_files_only=True)
    parts = {}
    n_trunc = 0
    for rid in lab["id"]:
        t = test.loc[rid, "text_clean"]
        enc = tok(t, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids, off = enc["input_ids"], enc["offset_mapping"]
        if len(ids) > budget:
            n_trunc += 1
            s, e = off[head][0], off[len(ids) - tail - 1][1]
            parts[rid] = (t[:s], t[s:e], t[e:])
    lt = lab[lab["cats"].map(lambda v: "long_truncated" in v)]
    check("both long_truncated reviews are truncated", set(lt["id"]) <= set(parts))
    claim(f"DistilBERT reads at most {budget} wordpieces: the first {head} and the last {tail}")
    claim(f"{n_trunc} of the {len(lab)} sampled reviews are longer than that")
    claim(f"only {len(lt)} were labelled `long_truncated`")
    claim(f"truncation explains at most {len(lt)} of {len(lab)} sampled errors")
    claim(f"reads at most {budget} wordpieces (head {head} + tail {tail}); {n_trunc} of {len(lab)} sampled errors are longer")
    h14, mid14, t14 = parts["test_14218"]
    h22, mid22, t22 = parts["test_22750"]
    check("test_14218 quote only in unseen middle", "Undoubtedly it has been a major success" in mid14
          and "Undoubtedly" not in h14 + t14)
    check("test_22750 quote only in unseen middle", "I liked Brain of Blood a lot" in mid22
          and "I liked Brain of Blood a lot" not in h22 + t22)
    check("test_22750 visible tail says 'I liked it'", "I liked it" in t22)
    check("both long_truncated examples are labelled positive", test.loc[["test_14218", "test_22750"], "label"].eq(1).all())

    # ---------------------------------------------------------------- label noise
    ln = lab[lab["cats"].map(lambda v: "label_noise" in v)]
    claim(f"{len(ln)} of {len(lab)} reviews have `label_noise` ({main_n['label_noise']} as main category")
    claim(f"{len(ln)} of {len(lab)} sampled errors were labelled `label_noise`")
    check("test_06593 is label_noise, labelled negative, ends with 8/10",
          "test_06593" in set(ln["id"]) and test.loc["test_06593", "label"] == 0
          and test.loc["test_06593", "text"].rstrip().endswith("8/10"))
    check("dataset has no rating column", not any("rating" in c.lower() for c in test.columns))

    # ---------------------------------------------------------------- heuristics (stored tags in the sample file)
    tags = csv(ROOT / "results/eval/manual_annotation_sample.csv", dtype={"id": str}).set_index("id").loc[lab["id"]]

    def score(tag, cat):
        y = lab["cats"].map(lambda v: cat in v).astype(int).values
        h = tags[tag].astype(int).values
        tp = int(((y == 1) & (h == 1)).sum())
        return dict(prec=tp / h.sum(), rec=tp / y.sum(), prev=y.mean(), fires=int(h.sum()), kappa=cohen_kappa_score(y, h))
    pairs = {("truncated", "long_truncated"), ("neg_cue_strict", "negation"), ("neg_of_negative", "negation"),
             ("neg_of_positive", "negation"), ("negated_sentiment_any", "negation"), ("has_contrast_cue", "mixed"),
             ("mixed_candidate", "mixed")}
    sc = {p: score(*p) for p in pairs}
    best = max(sc, key=lambda p: sc[p]["kappa"])
    check("has_contrast_cue vs mixed has the highest kappa", best == ("has_contrast_cue", "mixed"))
    check("best kappa is in the 'fair' band (0.2 to 0.4)", 0.2 <= sc[best]["kappa"] < 0.4)
    claim(f"`has_contrast_cue` vs `mixed` with Cohen's kappa {sc[best]['kappa']:.4f}")
    s = sc[("neg_cue_strict", "negation")]
    claim(f"`neg_cue_strict` fires on {s['fires']} of {len(lab)} reviews; its precision for `negation` is "
          f"{s['prec']:.4f}, barely above the {s['prev']:.4f} base rate")
    s = sc[("negated_sentiment_any", "negation")]
    claim(f"`negated_sentiment_any` finds {s['rec']:.4f} of the `negation` reviews, with precision {s['prec']:.4f}")
    s = sc[("mixed_candidate", "mixed")]
    claim(f"`mixed_candidate` has precision {s['prec']:.4f} for `mixed`, but {s['prev']:.4f} of all labelled errors")
    s = sc[("truncated", "long_truncated")]
    check("truncated finds both long_truncated reviews", s["rec"] == 1 and len(lt) == 2)
    claim(f"(recall {s['rec']:.4f}) with precision {s['prec']:.4f}")
    claim(f"On the {len(lab)} labelled errors")

    # ---------------------------------------------------------------- AI assistance
    sug = csv(ROOT / "results/eval/suggestions/llm_suggestions.csv", dtype=str, keep_default_na=False).set_index("id")
    log = csv(ROOT / "results/eval/suggestions/review_log.csv", dtype=str, keep_default_na=False)
    last = log.drop_duplicates("id", keep="last").set_index("id")
    backup = csv(ROOT / "reports/label_backups/annotator_A_kiet_20261112_051914.csv", dtype=str, keep_default_na=False)
    backup = backup[backup["category"].str.strip() != ""].set_index("id")["category"]
    pre_ai = set(backup.index)
    on_screen = lab[~lab["id"].isin(pre_ai)]
    check("ids without a log row are exactly the pre-AI labels", set(lab["id"]) - set(last.index) == pre_ai)
    a = lab.set_index("id")
    # a pre-AI label that differs now was revised later, with the suggestion seen
    revised = sorted(i for i in pre_ai if [p.strip() for p in backup[i].split(";")] != a.at[i, "cats"])
    check("exactly one pre-AI label was revised", len(revised) == 1)
    rv = revised[0]
    claim(f"{len(revised)} of those {len(pre_ai)} ({rv}) was revised after its suggestion was seen: its category "
          f"changed from `{backup[rv]}` to `{a.at[rv, 'category']}`")
    check("the revised label equals the suggested main + also categories", a.at[rv, "cats"] ==
          [sug.at[rv, "main_category"], *filter(None, sug.at[rv, "also_applies"].split(";"))])
    ai_main = sug.loc[a.index, "main_category"]
    edited = []
    for rid, r in a.iterrows():
        sg = sug.loc[rid]
        same = (sg["main_category"] == r["main"] and set(filter(None, sg["also_applies"].split(";"))) == set(r["cats"][1:])
                and (sg["unsure"] == "yes") == (r["unsure"].strip() == "yes"))
        if not same:
            unsure_only = (sg["main_category"] == r["main"]
                           and set(filter(None, sg["also_applies"].split(";"))) == set(r["cats"][1:]))
            edited.append((rid, unsure_only))
    dec = last["decision"].value_counts()
    claim(f"were shown for {len(on_screen)} reviews; {len(pre_ai)} reviews were labelled before the suggestions existed")
    edited_on = [u for rid, u in edited if rid not in pre_ai]
    claim(f"Of the {len(on_screen)}, {dec['accepted']} suggestions were accepted unchanged and {dec['edited']} were edited; "
          f"{sum(edited_on)} of the {len(edited_on)} edits only removed the \"unsure\" flag")
    claim(f"For {len(edited) - len(edited_on)} of the {len(pre_ai)} reviews labelled before the suggestions existed, "
          f"the final label differs from the suggestion")
    check("the revised label differs from its suggestion only in the unsure flag", (rv, True) in edited)
    claim(f"so `error_analysis/ai_edited_cases.csv` lists {len(edited)} reviews")
    check("ai_edited_cases.csv has one row per differing review", len(csv(EA / "ai_edited_cases.csv")) == len(edited))
    check("log decisions agree with the files", set(last.index[last["decision"] == "edited"]) | pre_ai >=
          {rid for rid, _ in edited} and dec.get("edited", 0) == len([e for e in edited if e[0] not in pre_ai]))
    same_main = (ai_main.values == a["main"].values)
    seen = ~a.index.isin(pre_ai) | a.index.isin(revised)
    claim(f"the main category is the same for {int(same_main.sum())} of {len(a)} reviews (raw agreement "
          f"{same_main.mean():.4f}, Cohen's kappa {cohen_kappa_score(ai_main, a['main']):.4f}), and for "
          f"{int(same_main[seen].sum())} of the {int(seen.sum())} reviews whose final label was set with the suggestion "
          f"seen (kappa {cohen_kappa_score(ai_main[seen], a.loc[seen, 'main']):.4f})")
    claim(f"All {len(lab)} reviews were labelled by one person")
    claim(f"equals the AI suggestion for {int(same_main.sum())} of {len(a)} reviews")
    mixed_ids = ai_main.index[ai_main == "mixed"]
    noise_ids = ai_main.index[ai_main == "label_noise"]
    noise_changed = [i for i in noise_ids if a.loc[i, "main"] != "label_noise"]
    claim(f"The AI suggested `mixed` as main category {len(mixed_ids)} times; the final main category differs in "
          f"{int((a.loc[mixed_ids, 'main'] != 'mixed').sum())} of them. It suggested `label_noise` {len(noise_ids)} "
          f"times, and the final main category is `label_noise` in {len(noise_ids) - len(noise_changed)} of them")
    claim(f"{int((a['unsure'] == 'yes').sum())} reviews were marked unsure and were not discussed")
    check("no discussion.csv exists", not (ROOT / "results/eval/annotation/discussion.csv").exists())

    # ---------------------------------------------------------------- examples
    examples = [("sarcasm", "test_01183", ["That ballot box was stuffed better than a Christmas turkey!"]),
                ("sarcasm", "test_02554", ["hilarious antics ensue (I wish)"]),
                ("negation", "test_14247", ["it's not too bad"]),
                ("negation", "test_00383", ["It's not half as good any of those films."]),
                ("mixed", "test_13076", ["It is really stupid and lame for the first half, then it actually starts to "
                                         "get good for the last half."]),
                ("mixed", "test_02576", ["Most of this film was okay", "you should go watch something else."]),
                ("long_truncated", "test_14218", ["Undoubtedly it has been a major success"]),
                ("long_truncated", "test_22750", ["I liked Brain of Blood a lot"]),
                ("label_noise", "test_04811", ["an innovative comedy that is genuinely funny and really touching at many times"]),
                ("label_noise", "test_12989", ["one-liners so flat and unfunny"]),
                ("label_noise", "test_06593", ["one of the strongest Russian movie for the past few years", "8/10"]),
                ("other", "test_16793", ["an insufferable jerk who I was praying would meet an extremely brutal and "
                                         "prolonged demise"]),
                ("other", "test_24629", ["PLEASE PICK UP THE SHOW!!!!"])]
    for cat, rid, quotes in examples:
        label = "positive" if test.loc[rid, "label"] == 1 else "negative"
        claim(f"| {cat} | {rid} | {label} |")
        check(f"example {rid}: category {cat} in hand labels", cat in a.loc[rid, "cats"])
        for q in quotes:
            claim(f"\"{q}\"")
            check(f"example {rid}: quote in review text and under 30 words", q in test.loc[rid, "text"] and len(q.split()) < 30)
    check("2-3 examples per category", all(2 <= sum(c == cat for c, _, _ in examples) <= 3 for cat in CATS))

    # ---------------------------------------------------------------- limitations: configs
    mlc = pd.read_csv(ROOT / "results/ml_config.csv").iloc[0]
    model_py = (ROOT / "ml/model.py").read_text(encoding="utf-8")
    check("TF-IDF sublinear_tf is True", str(mlc["tfidf_sublinear_tf"]) == "True")
    claim(f"TF-IDF uses ngram_range {mlc['tfidf_ngram_range']}, min_df {mlc['tfidf_min_df']}, max_df {mlc['tfidf_max_df']} "
          "and sublinear_tf")
    check("LogReg/LinearSVC built without C, and the sklearn default C is 1.0",
          "C=" not in model_py and LogisticRegression().C == 1.0 and LinearSVC().C == 1.0)
    claim(f"keep the default C = {LogisticRegression().C}")
    check("Word2Vec document vectors are averaged", "np.mean(vectors, axis=0)" in model_py)
    check("Word2Vec is skip-gram", int(mlc["word2vec_sg"]) == 1)
    claim(f"Word2Vec uses {mlc['word2vec_vector_size']} dimensions, window {mlc['word2vec_window']}, min_count "
          f"{mlc['word2vec_min_count']}, skip-gram and {mlc['word2vec_epochs']} epochs")
    tuning = pd.read_csv(ROOT / "results/dl_tuning.csv")
    check("DL tuning table has validation metrics only", any(c.startswith("val_") for c in tuning.columns)
          and not any(c.startswith("test") for c in tuning.columns))
    claim(f"tuned over {len(tuning)} configurations on the validation set")
    claim(f"best epoch ({int(cfg['best_epoch'])})")
    check("train_lora.py touches the test set once", "test set is touched exactly once" in
          (ROOT / "dl/train_lora.py").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 1: claims present
    for c in claims:
        check(f"claim in draft: {c[:90]}", c in draft)
        if c not in draft:
            print(f"MISSING CLAIM (full text): {c}")
    # ---------------------------------------------------------------- 2: no unverified number left
    rest = draft
    for c in sorted(claims + ALLOWED_LITERALS, key=len, reverse=True):
        rest = rest.replace(c, " ")
    rest = re.sub(r"test_\d{5}|Word2Vec|F1|all5|preds_\*|dl_config|ml_config|dl_tuning|ml_results|dl_results", " ", rest)
    rest = re.sub(r"(?m)^\d+\. ", " ", rest)  # numbered-list markers
    orphans = [rest[max(0, x.start() - 40):x.end() + 20].replace("\n", " ") for x in re.finditer(r"\d[\d,./]*", rest)]
    check(f"no unverified numbers left in the draft (found: {orphans})", not orphans)

    # ---------------------------------------------------------------- 3: tables agree
    mt = csv(ROOT / "results/eval/metrics_table.csv")
    check("metrics_table.csv matches recomputed metrics", all(
        np.isclose(mt.loc[mt["model"] == m[s]["name"], col].iloc[0], m[s][k], atol=1e-12)
        for s in m for k, col in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "f1"),
                                  ("auc", "roc_auc")]))
    cc = csv(EA / "category_counts.csv")
    allc = cc[cc["slice"] == "all"]
    check("category_counts.csv matches the annotator files", all(
        allc.loc[(allc["label_kind"] == "main") & (allc["category"] == c), "n"].iloc[0] == main_n[c] and
        allc.loc[(allc["label_kind"] == "any") & (allc["category"] == c), "n"].iloc[0] == any_n[c] for c in CATS))
    fin = csv(ROOT / "results/eval/annotation/annotations_final.csv", dtype=str, keep_default_na=False).set_index("id")
    check("annotations_final.csv equals the annotator files", (fin.loc[a.index, "final_category"] == a["category"]).all())

    # ---------------------------------------------------------------- 4: hand-label files unchanged
    for name, sha in LABEL_SHA256.items():
        got = hashlib.sha256((ROOT / "results/eval/annotation" / name).read_bytes()).hexdigest()
        check(f"SHA-256 unchanged: {name} = {got[:12]}...", got == sha)

    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed; {len(claims)} claims rendered.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
