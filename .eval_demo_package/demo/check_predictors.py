"""Check that the demo predictors reproduce results/predictions/preds_<model>.csv.

Runs every available predictor on the RAW text of N random test reviews and compares the label and
P(positive) with the saved prediction files. Unavailable models (e.g. the DL model before its
adapter arrives) are skipped with a message.

Pass rules: ML models need 100% label agreement and a tiny |prob diff| (TOLERANCE). The DL model needs
max |prob diff| <= 0.01; a label may differ only when the stored prob is within 0.01 of 0.5.

Usage (from the repository root):
    python demo/check_predictors.py
    python demo/check_predictors.py --n 500 --seed 1
Exit code 1 if an available model fails these rules.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from predictors import get_predictors

ROOT = Path(__file__).resolve().parents[1]
# Max allowed |prob difference| per model.
TOLERANCE = {"tfidf_logreg": 1e-6, "w2v_logreg": 1e-6, "tfidf_svm": 1e-4, "w2v_svm": 1e-4, "distilbert_lora": 0.01}
# DL only: the label may differ when the stored prob is this close to 0.5 (CPU fp32 here, GPU fp16 in training).
DL_NEAR_HALF = 0.01


def main():
    parser = argparse.ArgumentParser(description="Compare the demo predictors with the saved test predictions.")
    parser.add_argument("--n", type=int, default=200, help="number of random test reviews")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the sample")
    args = parser.parse_args()

    test = pd.read_json(ROOT / "imdb_outputs/imdb_test_prepared.jsonl", lines=True, encoding="utf-8")
    sample = test.sample(n=args.n, random_state=args.seed)
    print(f"{len(sample)} random test reviews (seed {args.seed})")

    failed = []
    for name, (predict, status) in get_predictors().items():
        print(f"\n== {name}: {status}")
        if predict is None:
            print("   SKIPPED: model not available")
            continue
        saved = pd.read_csv(ROOT / f"results/predictions/preds_{name}.csv", encoding="utf-8-sig").set_index("id")
        rows = []
        for r in sample.itertuples():
            assert saved.at[r.review_id, "text"] == r.text, f"{r.review_id}: text differs from the prediction file"
            label, prob = predict(r.text)
            rows.append((r.review_id, label, prob, saved.at[r.review_id, "pred"], saved.at[r.review_id, "prob"]))
        df = pd.DataFrame(rows, columns=["id", "pred_demo", "prob_demo", "pred_file", "prob_file"])
        df["abs_diff"] = (df["prob_demo"] - df["prob_file"]).abs()
        df["allowed"] = (name == "distilbert_lora") & ((df["prob_file"] - 0.5).abs() <= DL_NEAR_HALF)
        differ = df["pred_demo"] != df["pred_file"]
        bad = differ & ~df["allowed"]
        max_diff = df["abs_diff"].max()
        tolerance = TOLERANCE[name]

        reasons = []
        if bad.any():
            reasons.append(f"{int(bad.sum())} label disagreement(s)"
                           + (f" with stored prob farther than {DL_NEAR_HALF} from 0.5" if name == "distilbert_lora" else ""))
        if max_diff > tolerance:
            reasons.append(f"max |prob diff| {max_diff:.3e} > allowed {tolerance}")

        print(f"   label agreement: {int((~differ).sum())}/{len(df)} = {(~differ).mean():.2%}")
        if name == "distilbert_lora":
            print(f"   disagreements allowed because the stored prob is within {DL_NEAR_HALF} of 0.5: "
                  f"{int((differ & df['allowed']).sum())}")
        print(f"   max |prob diff|: {max_diff:.3e} (review {df.loc[df['abs_diff'].idxmax(), 'id']}), allowed: {tolerance}")
        if differ.any():
            print(f"   label disagreements (first 10 of {int(differ.sum())}):")
            print(df[differ].head(10).to_string(index=False))
        print(f"   result: {'OK' if not reasons else 'FAIL: ' + '; '.join(reasons)}")
        if reasons:
            failed.append(name)

    if failed:
        sys.exit(f"\nFAIL: {', '.join(failed)}")
    print("\nAll available models match their prediction files.")


if __name__ == "__main__":
    main()
