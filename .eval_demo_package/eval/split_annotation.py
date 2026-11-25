"""Split the 100-review annotation sample into two disjoint halves of 50, one per annotator.

Input:  results/eval/manual_annotation_sample.csv   master sample (only read, never changed)
Output (results/eval/annotation/):
    annotator_A_*.csv         50 reviews for annotator A
    annotator_B_*.csv         50 reviews for annotator B
        If one file matches a pattern (e.g. annotator_B_Phat.csv), that file is used; if none matches, the
        default names annotator_A_kiet.csv / annotator_B_person1.csv are written; more than one match: stop.
        columns: id, text, label, pred_dl, prob_dl, category, unsure, note (the last three empty).
        No automatic tags and no n_models_wrong, so the annotators are not primed.
    _split_key.csv            id, annotator (A/B), group (all5/other): used later to join results back

Split (seed 42): group "all5" = reviews wrong for all 5 models (n_models_wrong == 5), group "other"
= the rest. Each half gets half of each group (20 all5 + 30 other), and inside each group the true
labels are spread over the two halves as evenly as possible.

Hand labels are never lost: if an annotator file already has text in category/unsure/note,
the script stops and writes nothing.

Usage (from the repository root):
    python eval/split_annotation.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ANNOTATOR_PATTERNS = {"A": "annotator_A_*.csv", "B": "annotator_B_*.csv"}  # same as eval/merge_annotations.py
DEFAULT_FILES = {"A": "annotator_A_kiet.csv", "B": "annotator_B_person1.csv"}  # only when no file matches yet
KEY_FILE = "_split_key.csv"
OUT_COLUMNS = ["id", "text", "label", "pred_dl", "prob_dl", "category", "unsure", "note"]
HAND_COLUMNS = ["category", "unsure", "note"]
EXPECTED_SIZES = {("A", "all5"): 20, ("A", "other"): 30, ("B", "all5"): 20, ("B", "other"): 30}


def has_hand_labels(path):
    """True if the file exists and any category/unsure/note cell is filled (or its columns look wrong)."""
    if not path.exists():
        return False
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if not set(HAND_COLUMNS) <= set(df.columns):
        return True  # unexpected format: safer not to overwrite
    return bool((df[HAND_COLUMNS].apply(lambda col: col.str.strip()) != "").any().any())


def annotator_paths(folder):
    """{'A': path, 'B': path}: the one existing file per pattern, or the default name if none exists yet.
    Stops and lists the files if a pattern matches more than one file."""
    paths, problems = {}, []
    for annotator, pattern in ANNOTATOR_PATTERNS.items():
        found = sorted(folder.glob(pattern))
        if len(found) > 1:
            problems.append(f"{pattern}: found {len(found)} files {[p.name for p in found]}")
        paths[annotator] = found[0] if found else folder / DEFAULT_FILES[annotator]
    if problems:
        sys.exit(f"STOP: more than one file per annotator pattern in {folder}. Nothing was written.\n  - "
                 + "\n  - ".join(problems))
    return paths


def split_halves(master):
    """Return a Series id -> "A" or "B"."""
    assignment = {}
    for _, rows in master.groupby("group"):
        # shuffle inside each label, then put the label-0 rows before the label-1 rows
        ordered = pd.concat(part.sample(frac=1, random_state=SEED) for _, part in rows.groupby("label"))
        # alternate A, B, A, B, ...: the group and each label in it are split as evenly as possible
        for i, review_id in enumerate(ordered["id"]):
            assignment[review_id] = "AB"[i % 2]
    return pd.Series(assignment)


def main():
    parser = argparse.ArgumentParser(description="Split the annotation sample between two annotators.")
    parser.add_argument("--master", type=Path, default=ROOT / "results/eval/manual_annotation_sample.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/eval/annotation", help="output folder")
    args = parser.parse_args()

    paths = annotator_paths(args.out)
    protected = [path.name for path in paths.values() if has_hand_labels(path)]
    if protected:
        sys.exit(f"STOP: these files already have hand labels in category/unsure/note (or unexpected columns): "
                 f"{', '.join(protected)}. Nothing was written, so no labels were lost.")

    master = pd.read_csv(args.master, encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)
    if not master["id"].is_unique:
        sys.exit("ERROR: the master sample has duplicate ids.")
    master["group"] = master["n_models_wrong"].map(lambda n: "all5" if n == 5 else "other")
    master["annotator"] = master["id"].map(split_halves(master))

    print("Reviews per half: group x true label")
    print(pd.crosstab([master["annotator"], master["group"]], master["label"], margins=True).to_string())

    # checks: 20 all5 + 30 other per half, halves disjoint, together exactly the master ids
    sizes = master.groupby(["annotator", "group"]).size().to_dict()
    if sizes != EXPECTED_SIZES:
        sys.exit(f"ERROR: unexpected half sizes {sizes}, expected {EXPECTED_SIZES}.")
    ids_a = set(master.loc[master["annotator"] == "A", "id"])
    ids_b = set(master.loc[master["annotator"] == "B", "id"])
    if ids_a & ids_b or ids_a | ids_b != set(master["id"]):
        sys.exit("ERROR: the halves overlap or do not cover the master sample.")
    print(f"\nCheck: A has {len(ids_a)} ids, B has {len(ids_b)}, shared {len(ids_a & ids_b)}, "
          f"union equals the {len(master)} master ids: {ids_a | ids_b == set(master['id'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    for annotator, path in paths.items():
        half = master[master["annotator"] == annotator].sample(frac=1, random_state=SEED)  # shuffle rows
        half = half.assign(category="", unsure="", note="")[OUT_COLUMNS]
        half.to_csv(path, index=False, encoding="utf-8-sig")
    key = master[["id", "annotator", "group"]].sort_values(["annotator", "id"])
    key.to_csv(args.out / KEY_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {', '.join(p.name for p in paths.values())} and {KEY_FILE} to {args.out}")


if __name__ == "__main__":
    main()
