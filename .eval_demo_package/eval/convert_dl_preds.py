"""Convert the DistilBERT + LoRA predictions to the team prediction format.

Reads   results/predictions/distilbert_lora.csv        (review_id, split, label, pred, score)
Writes  results/predictions/preds_distilbert_lora.csv  (id, text, label, pred, prob)

Only the test split is kept. The original review text is taken from the test jsonl.

Usage (from the repository root):
    python eval/convert_dl_preds.py
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Convert DL predictions to the team format.")
    parser.add_argument("--dl-preds", type=Path, default=ROOT / "results/predictions/distilbert_lora.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "imdb_outputs/imdb_test_prepared.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/predictions/preds_distilbert_lora.csv")
    args = parser.parse_args()

    dl = pd.read_csv(args.dl_preds)
    dl = dl[dl["split"] == "test"]
    test = pd.read_json(args.test, lines=True)[["review_id", "text", "label"]]

    assert len(dl) == 25_000, f"expected 25,000 test rows, got {len(dl)}"
    assert dl["review_id"].is_unique, "duplicate review_id in the DL test predictions"
    assert set(dl["review_id"]) == set(test["review_id"]), "DL test ids differ from the test jsonl ids"

    # Start from the jsonl so the output keeps the test file's row order.
    df = test.merge(dl, on="review_id", suffixes=("", "_dl"))
    n_bad = (df["label"] != df["label_dl"]).sum()
    assert n_bad == 0, f"{n_bad} DL labels differ from the test jsonl labels"
    n_bad = (df["pred"] != (df["score"] > 0.5).astype(int)).sum()
    assert n_bad == 0, f"{n_bad} rows where pred != (score > 0.5)"

    out = df.rename(columns={"review_id": "id", "score": "prob"})[["id", "text", "label", "pred", "prob"]]
    # utf-8-sig adds a BOM so Excel on Windows shows non-ASCII characters correctly
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
