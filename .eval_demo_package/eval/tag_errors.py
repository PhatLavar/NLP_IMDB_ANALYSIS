"""Tag every test review with error-analysis categories and compute error rates per model.

Inputs:
    results/predictions/preds_*.csv        all models (team format, read with utf-8-sig)
    imdb_outputs/imdb_test_prepared.jsonl  review_id, text, text_clean, label
    results/eval/metrics_comparison.csv    written by eval/evaluate.py (used for a sanity check)

Outputs (results/eval/):
    tagged_test.csv                  one row per test review (all 25,000): tags + err_<model> (1 = wrong).
                                     No review text, to keep it small.
    error_rate_by_category.csv/.md   error rate per model and category value, with a 95% Wilson CI
    manual_annotation_sample.csv     100 wrong reviews to label by hand (see annotation_guide.md).
                                     Never overwritten once it exists, because it holds hand labels.

Tags:
    n_wordpieces, truncated   DistilBERT wordpieces of text_clean, [CLS]/[SEP] included, no truncation.
                              truncated = more than 512 wordpieces (the DL model's limit).
    word_count, length_bin    whitespace-separated words of the original text
    neg_cue_strict            not, never, cannot, can not, no longer, or an n't contraction (don't, isn't, ...)
    neg_of_negative           a strict cue followed within 3 words, in the same sentence, by a NEGATIVE
                              lexicon word ("not bad", leans positive)
    neg_of_positive           the same with a POSITIVE lexicon word ("not good", leans negative)
    negated_sentiment_any     neg_of_negative OR neg_of_positive
    has_contrast_cue          but, however, although, ...
    mixed_candidate           contrast cue + at least one positive AND one negative lexicon word
    duplicate_text            the original text occurs more than once in the test set

IMPORTANT: the neg_*, negated_sentiment_any, has_contrast_cue and mixed_candidate tags are crude
keyword heuristics on the lowercased text_clean (HTML removed). They are NOT gold labels.
Sarcasm cannot be tagged automatically; it is labelled by hand in manual_annotation_sample.csv.

Usage (from the repository root):
    python eval/tag_errors.py
"""
import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
import transformers
from transformers import AutoTokenizer

# Works because Python puts this script's folder (eval/) on the import path.
from evaluate import check_same_test_set, load_preds, model_name

ROOT = Path(__file__).resolve().parents[1]

# ---- Word lists (edit here). A trailing * matches any ending: disappoint* = disappointing, disappointed, ...
POSITIVE_WORDS = ["good", "great", "best", "excellent", "amazing", "brilliant", "wonderful", "perfect",
                  "fun", "funny", "worth", "love", "loved", "enjoy*", "recommend*"]
NEGATIVE_WORDS = ["bad", "worse", "worst", "terrible", "awful", "horrible", "poor", "boring", "dull",
                  "stupid", "annoying", "mess", "waste*", "hate*", "disappoint*"]
NEGATION_CUES_STRICT = ["not", "never", "cannot", "can not", "no longer"]  # plus any n't contraction
CONTRAST_CUES = ["but", "however", "although", "though", "yet", "despite", "nevertheless",
                 "on the other hand", "in spite of"]

NEGATION_WINDOW = 3     # words after a negation cue (same sentence) that are checked for a lexicon word
MAX_WORDPIECES = 512    # DistilBERT input limit used by the DL model
SMALL_N = 30            # groups with fewer reviews are marked small_n
LENGTH_BINS = [0, 100, 200, 300, 400, float("inf")]
LENGTH_LABELS = ["<=100", "101-200", "201-300", "301-400", ">400"]
CATEGORIES = ["truncated", "length_bin", "neg_cue_strict", "neg_of_negative", "neg_of_positive",
              "negated_sentiment_any", "has_contrast_cue", "mixed_candidate"]
N_SAMPLE_DL_WRONG = 60   # annotation sample: wrong for DL, but not wrong for all models
N_SAMPLE_ALL_WRONG = 40  # annotation sample: wrong for all models


def words_pattern(words):
    """Regex text matching any listed word or phrase; 'x*' also matches longer words starting with x."""
    return "|".join(re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) for w in words)


NEG_CUE_STRICT_RE = re.compile(r"\b(?:" + words_pattern(NEGATION_CUES_STRICT) + r"|\w+n['’]t)\b")
CONTRAST_RE = re.compile(r"\b(?:" + words_pattern(CONTRAST_CUES) + r")\b")
POSITIVE_RE = re.compile(r"\b(?:" + words_pattern(POSITIVE_WORDS) + r")\b")
NEGATIVE_RE = re.compile(r"\b(?:" + words_pattern(NEGATIVE_WORDS) + r")\b")
WORD_RE = re.compile(r"[a-z]+(?:['’][a-z]+)*")  # words, keeping "isn't" as one word
SENTENCE_RE = re.compile(r"[^.!?;:\n]+")       # sentence pieces: stop at . ! ? ; : or a line break


def find_negated(text, lexicon_re):
    """First strict negation cue followed within NEGATION_WINDOW words, in the same sentence,
    by a word matching lexicon_re. Returns (cue match, word match), or None if there is none."""
    for sentence in SENTENCE_RE.finditer(text):
        for cue in NEG_CUE_STRICT_RE.finditer(text, sentence.start(), sentence.end()):
            following = list(WORD_RE.finditer(text, cue.end(), sentence.end()))[:NEGATION_WINDOW]
            for word in following:
                if lexicon_re.fullmatch(word.group()):
                    return cue, word
    return None


def count_wordpieces(texts):
    """Number of DistilBERT wordpieces per text, special tokens included, no truncation."""
    try:
        # Cache inside the repo (models/ is gitignored) instead of the user's home folder.
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=ROOT / "models/hf_cache")
    except OSError as e:
        sys.exit(f"ERROR: could not load the distilbert-base-uncased tokenizer "
                 f"(internet is needed the first time): {e}")
    transformers.logging.set_verbosity_error()  # hide the expected "longer than 512" warning
    ids = tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    return [len(x) for x in ids]


def tag_reviews(test):
    """One row per test review with all tags (0/1 for yes/no tags)."""
    lower = test["text_clean"].str.lower()
    tags = pd.DataFrame({"id": test["id"], "label": test["label"]})
    tags["n_wordpieces"] = count_wordpieces(test["text_clean"])
    tags["truncated"] = (tags["n_wordpieces"] > MAX_WORDPIECES).astype(int)
    tags["word_count"] = test["text"].str.split().str.len()
    tags["length_bin"] = pd.cut(tags["word_count"], bins=LENGTH_BINS, labels=LENGTH_LABELS).astype(str)
    tags["neg_cue_strict"] = [int(bool(NEG_CUE_STRICT_RE.search(t))) for t in lower]
    tags["neg_of_negative"] = [int(find_negated(t, NEGATIVE_RE) is not None) for t in lower]
    tags["neg_of_positive"] = [int(find_negated(t, POSITIVE_RE) is not None) for t in lower]
    tags["negated_sentiment_any"] = tags["neg_of_negative"] | tags["neg_of_positive"]
    tags["has_contrast_cue"] = [int(bool(CONTRAST_RE.search(t))) for t in lower]
    tags["mixed_candidate"] = [int(bool(CONTRAST_RE.search(t) and POSITIVE_RE.search(t) and NEGATIVE_RE.search(t)))
                               for t in lower]
    tags["duplicate_text"] = test["text"].duplicated(keep=False).astype(int)
    return tags


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k errors out of n reviews."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)  # clamp tiny float noise at 0 / 1


def error_rates(tags, models):
    """Error rate of each model for each category value, over all test reviews."""
    rows = []
    for model in models:
        err = tags[f"err_{model}"]
        for col in CATEGORIES:
            values = LENGTH_LABELS if col == "length_bin" else [1, 0]
            for value in values:
                in_group = tags[col] == value
                n, k = int(in_group.sum()), int(err[in_group].sum())
                low, high = wilson_ci(k, n)
                rows.append({
                    "model": model,
                    "category": col,
                    "value": {1: "yes", 0: "no"}.get(value, value),
                    "n_reviews": n,
                    "n_errors": k,
                    "error_rate": k / n if n else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                    "overall_error_rate": err.mean(),
                    "small_n": "yes" if n < SMALL_N else "no",
                })
    table = pd.DataFrame(rows)
    table["relative_risk"] = table["error_rate"] / table["overall_error_rate"]
    return table


def to_markdown(table):
    lines = ["| model | category | value | n_reviews | n_errors | error_rate | 95% CI | overall_error_rate "
             "| relative_risk | small_n |",
             "|---|---|---|---:|---:|---:|---|---:|---:|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.model} | {r.category} | {r.value} | {r.n_reviews:,} | {r.n_errors:,} | {r.error_rate:.4f} "
                     f"| [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.overall_error_rate:.4f} | {r.relative_risk:.2f} "
                     f"| {r.small_n} |")
    return "\n".join(lines) + "\n"


def sanity_checks(tags, models):
    """Stop if the per-group error counts do not add up; print the truncated share."""
    metrics_path = ROOT / "results/eval/metrics_comparison.csv"
    if not metrics_path.is_file():
        sys.exit(f"ERROR: {metrics_path} not found. Run python eval/evaluate.py first.")
    metrics = pd.read_csv(metrics_path, encoding="utf-8").set_index("model")

    print("Sanity check: errors in truncated=yes + truncated=no vs. n_errors in metrics_comparison.csv")
    broken = []
    for model in models:
        if model not in metrics.index:
            sys.exit(f"ERROR: model '{model}' is missing from {metrics_path.name}. Run python eval/evaluate.py first.")
        err = tags[f"err_{model}"]
        yes, no = int(err[tags["truncated"] == 1].sum()), int(err[tags["truncated"] == 0].sum())
        expected = int(metrics.loc[model, "n_errors"])
        status = "ok" if yes + no == expected else "MISMATCH"
        print(f"  {model:<18} {yes:>5,} + {no:>5,} = {yes + no:>5,}   expected {expected:>5,}   {status}")
        if status != "ok":
            broken.append(model)
    if broken:
        sys.exit(f"ERROR: error counts do not add up for {broken}. Is metrics_comparison.csv out of date?")

    n_trunc = int(tags["truncated"].sum())
    print(f"\nTruncated test reviews (> {MAX_WORDPIECES} wordpieces): "
          f"{n_trunc:,} of {len(tags):,} = {n_trunc / len(tags):.1%}")
    dl_config = ROOT / "results/dl_config.csv"
    if dl_config.is_file():
        frac = pd.read_csv(dl_config, encoding="utf-8")["truncated_frac_train"].iloc[0]
        print(f"DL teammate's truncated_frac_train (results/dl_config.csv, TRAIN split): {frac:.1%} "
              "(different split, shown for comparison only)")


def make_annotation_sample(tags, test, dfs, path, seed=42):
    """100 wrong reviews for hand labelling: 60 wrong for DL but not for all models, 40 wrong for all models."""
    if path.exists():
        print(f"\n{path.name} already exists and was NOT overwritten (it may contain hand labels). "
              "Delete it to draw a new sample.")
        return
    if "distilbert_lora" not in dfs:
        sys.exit("ERROR: preds_distilbert_lora.csv is needed for the annotation sample.")
    err_cols = [f"err_{m}" for m in dfs]
    pool = tags.assign(text=test["text"].values, n_models_wrong=tags[err_cols].sum(axis=1))
    pool = pool[~pool["text"].duplicated(keep="first")]  # at most one copy of a duplicated text

    wrong_all = pool[pool["n_models_wrong"] == len(dfs)]
    wrong_dl_not_all = pool[(pool["err_distilbert_lora"] == 1) & (pool["n_models_wrong"] < len(dfs))]
    sample = pd.concat([wrong_dl_not_all.sample(N_SAMPLE_DL_WRONG, random_state=seed),
                        wrong_all.sample(N_SAMPLE_ALL_WRONG, random_state=seed)])
    sample = sample.sample(frac=1, random_state=seed)  # shuffle so the two groups are mixed
    assert len(sample) == 100 and sample["id"].is_unique and sample["text"].is_unique

    dl = dfs["distilbert_lora"].set_index("id")
    sample["pred_dl"] = sample["id"].map(dl["pred"])
    sample["prob_dl"] = sample["id"].map(dl["prob"])
    for col in ["category", "annotator", "note"]:  # filled in by hand
        sample[col] = ""
    cols = ["id", "text", "label", "pred_dl", "prob_dl", "n_models_wrong", "truncated", "neg_cue_strict",
            "neg_of_negative", "neg_of_positive", "negated_sentiment_any", "has_contrast_cue", "mixed_candidate",
            "category", "annotator", "note"]
    sample[cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(sample)} rows to {path} "
          f"({N_SAMPLE_DL_WRONG} wrong for DL but not all models, {N_SAMPLE_ALL_WRONG} wrong for all models)")


def main():
    parser = argparse.ArgumentParser(description="Tag test reviews and compute error rates per category.")
    parser.add_argument("files", nargs="*", type=Path,
                        help="prediction files (default: all results/predictions/preds_*.csv)")
    parser.add_argument("--test", type=Path, default=ROOT / "imdb_outputs/imdb_test_prepared.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/eval", help="output folder (default: results/eval)")
    args = parser.parse_args()

    files = args.files or sorted((ROOT / "results/predictions").glob("preds_*.csv"))
    if not files:
        sys.exit("ERROR: no results/predictions/preds_*.csv files found.")
    dfs = {model_name(path): load_preds(path) for path in files}

    test = pd.read_json(args.test, lines=True, encoding="utf-8")
    test = test.rename(columns={"review_id": "id"})[["id", "text", "text_clean", "label"]]
    check_same_test_set(dfs, test)

    tags = tag_reviews(test)
    for name, df in dfs.items():
        d = df.set_index("id")
        tags[f"err_{name}"] = (d["label"] != d["pred"]).astype(int).reindex(tags["id"]).values

    sanity_checks(tags, list(dfs))

    args.out.mkdir(parents=True, exist_ok=True)
    tags.to_csv(args.out / "tagged_test.csv", index=False, encoding="utf-8")
    table = error_rates(tags, list(dfs))
    table.to_csv(args.out / "error_rate_by_category.csv", index=False, encoding="utf-8")
    (args.out / "error_rate_by_category.md").write_text(to_markdown(table), encoding="utf-8")
    print(f"\nWrote tagged_test.csv and error_rate_by_category.csv/.md to {args.out}")

    print(f"\nHighest-risk categories per model (relative_risk = error_rate / overall_error_rate; "
          f"groups under {SMALL_N} reviews left out):")
    for model, group in table[table["small_n"] == "no"].groupby("model", sort=False):
        print(f"  {model} (overall error rate {group['overall_error_rate'].iloc[0]:.4f})")
        for r in group.nlargest(5, "relative_risk").itertuples():
            print(f"    {r.category + ' = ' + r.value:<30} n={r.n_reviews:>6,}  error_rate={r.error_rate:.4f} "
                  f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]  relative_risk={r.relative_risk:.2f}")

    make_annotation_sample(tags, test, dfs, args.out / "manual_annotation_sample.csv")


if __name__ == "__main__":
    main()
