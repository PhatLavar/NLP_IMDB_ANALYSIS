"""Merge the two annotators' files and summarise the error categories.

Inputs (results/eval/annotation/):
    annotator_A_*.csv, annotator_B_*.csv             filled in by the annotators (see annotation_guide.md);
                                                     exactly one file per pattern, e.g. annotator_B_Phat.csv
    _split_key.csv                                   id, annotator, group (from eval/split_annotation.py)
    discussion.csv  (optional)                       a copy of discussion_queue.csv with final_category
                                                     filled in during the meeting
Outputs (same folder):
    discussion_queue.csv     reviews marked unsure = yes, with an empty final_category for the meeting
    annotations_final.csv    all reviews with their final category
    annotation_summary.md    category counts and percentages, overall and by group (all5 / other)

All problems in the inputs are listed together and nothing is written until they are fixed.

Usage (from the repository root):
    python eval/merge_annotations.py
    python eval/merge_annotations.py --dir <folder>   (another folder with the same files, e.g. a test)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["sarcasm", "negation", "mixed", "long_truncated", "label_noise", "other"]
ANNOTATOR_PATTERNS = {"A": "annotator_A_*.csv", "B": "annotator_B_*.csv"}  # same as eval/split_annotation.py
REQUIRED_COLUMNS = ["id", "text", "label", "pred_dl", "category", "unsure", "note"]
QUEUE_COLUMNS = ["id", "text", "label", "pred_dl", "annotator", "category", "note", "final_category"]
FINAL_COLUMNS = ["id", "text", "label", "pred_dl", "annotator", "group", "category", "unsure", "note",
                 "final_category", "main_category", "changed_in_discussion"]


def read_csv(path):
    """Read a CSV with every cell as text; empty cells become ''. Stops with a clear message if not UTF-8."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        sys.exit(f"ERROR: {path.name} is not valid UTF-8. Please save it as CSV UTF-8 (in Excel: Save As > "
                 "CSV UTF-8 (Comma delimited); in Google Sheets: File > Download > CSV), then run again. "
                 "Nothing was written.")


def category_problem(value):
    """None if value is a valid category list such as 'Negation ; mixed' (any case), else a short reason."""
    parts = [p.strip().lower() for p in value.split(";")]
    if value.strip() == "":
        return "category is empty"
    unknown = [p for p in parts if p not in CATEGORIES]
    if unknown:
        return f"unknown value(s) {unknown}; allowed: {' | '.join(CATEGORIES)}"
    if len(set(parts)) != len(parts):
        return "a value is repeated"
    return None


def normalize(value):
    """'Negation ; Mixed' -> 'negation;mixed'"""
    return ";".join(p.strip().lower() for p in value.split(";"))


def find_annotator_files(folder):
    """{'A': path, 'B': path}: exactly one file per pattern, otherwise stop and list the files found."""
    found = {annotator: sorted(folder.glob(pattern)) for annotator, pattern in ANNOTATOR_PATTERNS.items()}
    problems = [f"{ANNOTATOR_PATTERNS[annotator]}: found {len(paths)} file(s) {[p.name for p in paths]}"
                for annotator, paths in found.items() if len(paths) != 1]
    if problems:
        sys.exit(f"ERROR: expected exactly one file per annotator pattern in {folder}. Nothing was written.\n  - "
                 + "\n  - ".join(problems)
                 + f"\n  CSV files in the folder: {sorted(p.name for p in folder.glob('*.csv'))}")
    return {annotator: paths[0] for annotator, paths in found.items()}


def load_annotations(folder, key):
    """Read and check both annotator files; return one table with an annotator and a group column."""
    problems, frames = [], []
    for annotator, path in find_annotator_files(folder).items():
        filename = path.name
        df = read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            problems.append(f"{filename}: missing columns {missing}")
            continue
        for r in df.itertuples():
            reason = category_problem(r.category)
            if reason:
                problems.append(f"{filename}, id {r.id}: {reason} (category = {r.category!r})")
            if r.unsure.strip() not in ("", "yes"):
                problems.append(f"{filename}, id {r.id}: unsure must be empty or 'yes', found {r.unsure!r}")
        expected = set(key.loc[key["annotator"] == annotator, "id"])
        dupes = sorted(df.loc[df["id"].duplicated(), "id"].unique())
        if dupes:
            problems.append(f"{filename}: duplicate ids {dupes}")
        if sorted(expected - set(df["id"])):
            problems.append(f"{filename}: ids from the key are missing: {sorted(expected - set(df['id']))}")
        if sorted(set(df["id"]) - expected):
            problems.append(f"{filename}: ids not assigned to annotator {annotator} in the key: "
                            f"{sorted(set(df['id']) - expected)}")
        frames.append(df.assign(annotator=annotator))
    if problems:
        sys.exit("ERROR: problems in the annotation files. Nothing was written.\n  - " + "\n  - ".join(problems))

    merged = pd.concat(frames, ignore_index=True)
    merged["category"] = merged["category"].map(normalize)
    merged["unsure"] = merged["unsure"].str.strip()
    return merged.merge(key[["id", "group"]], on="id", validate="one_to_one")


def apply_discussion(df, path):
    """final_category = agreed value from discussion.csv for its ids, else the annotator's category."""
    df["final_category"] = df["category"]
    if not path.is_file():
        return df, None
    disc = read_csv(path)
    if not {"id", "final_category"} <= set(disc.columns):
        sys.exit(f"ERROR: {path.name} needs the columns id and final_category. Nothing was written.")
    problems = []
    for r in disc.itertuples():
        if r.id not in set(df["id"]):
            problems.append(f"{path.name}, id {r.id}: id is not in the annotation files")
        reason = category_problem(r.final_category)
        if reason:
            problems.append(f"{path.name}, id {r.id}: final_category: {reason} (= {r.final_category!r})")
    if disc["id"].duplicated().any():
        problems.append(f"{path.name}: duplicate ids {sorted(disc.loc[disc['id'].duplicated(), 'id'].unique())}")
    if problems:
        sys.exit(f"ERROR: problems in {path.name}. Nothing was written.\n  - " + "\n  - ".join(problems))
    agreed = disc.set_index("id")["final_category"].map(normalize)
    decided = df["id"].isin(agreed.index)
    df.loc[decided, "final_category"] = df.loc[decided, "id"].map(agreed)
    return df, len(agreed)


def category_table(df, column):
    """Markdown table: number and % of reviews per category value, overall and per group."""
    exploded = df[["group"]].assign(value=df[column].str.split(";")).explode("value")
    groups = {"all": df, "all5": df[df["group"] == "all5"], "other": df[df["group"] == "other"]}
    header = "| category | " + " | ".join(f"{g} (n={len(part)})" for g, part in groups.items()) + " |"
    lines = [header, "|---|" + "---:|" * len(groups)]
    for category in CATEGORIES:
        cells = []
        for g, part in groups.items():
            rows = exploded if g == "all" else exploded[exploded["group"] == g]
            n = int((rows["value"] == category).sum())
            cells.append(f"{n} ({n / len(part):.0%})" if len(part) else "0")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Merge the two annotators' files and summarise the categories.")
    parser.add_argument("--dir", type=Path, default=ROOT / "results/eval/annotation",
                        help="folder with the annotator files, _split_key.csv and optional discussion.csv")
    args = parser.parse_args()

    key_path = args.dir / "_split_key.csv"
    if not key_path.is_file():
        sys.exit(f"ERROR: {key_path} not found. Run python eval/split_annotation.py first.")
    key = read_csv(key_path)
    df = load_annotations(args.dir, key)
    df, n_discussed = apply_discussion(df, args.dir / "discussion.csv")

    # never overwrite a queue in which someone already typed final decisions
    queue_path = args.dir / "discussion_queue.csv"
    if queue_path.is_file():
        old = read_csv(queue_path)
        if "final_category" in old.columns and (old["final_category"].str.strip() != "").any():
            sys.exit(f"STOP: {queue_path.name} already has final_category values. Save it as discussion.csv "
                     "(that is the file this script reads), then run again. Nothing was written.")

    df["main_category"] = df["final_category"].str.split(";").str[0]
    df["changed_in_discussion"] = (df["final_category"] != df["category"]).map({True: "yes", False: "no"})
    df = df.sort_values("id")

    unsure = df[df["unsure"] == "yes"]
    unsure.assign(final_category="")[QUEUE_COLUMNS].to_csv(queue_path, index=False, encoding="utf-8-sig")
    df[FINAL_COLUMNS].to_csv(args.dir / "annotations_final.csv", index=False, encoding="utf-8-sig")

    n_all5, n_other = int((df["group"] == "all5").sum()), int((df["group"] == "other").sum())
    discussion_line = ("discussion.csv not found: final_category = the annotator's category" if n_discussed is None
                       else f"Reviews with an agreed category in discussion.csv: {n_discussed}")
    summary = "\n".join([
        "# Annotation summary",
        "",
        f"**These percentages come from a sample of {len(df)} misclassified reviews, not from all errors.** "
        f"The sample has {n_all5} reviews wrong for all 5 models (group all5) and {n_other} reviews wrong for "
        "DistilBERT but not for all 5 models (group other).",
        "",
        f"- Reviews: {len(df)} (annotator A: {int((df['annotator'] == 'A').sum())}, "
        f"annotator B: {int((df['annotator'] == 'B').sum())})",
        f"- Unsure reviews: {len(unsure)}",
        f"- {discussion_line}",
        f"- Reviews whose category changed after the discussion: {int((df['changed_in_discussion'] == 'yes').sum())}",
        "",
        "## Main category (the first value of final_category)",
        "",
        category_table(df, "main_category"),
        "",
        "## Every category value",
        "",
        "A review with several values is counted once for each value, so a column can add up to more than 100%.",
        "",
        category_table(df, "final_category"),
        "",
    ])
    (args.dir / "annotation_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote discussion_queue.csv ({len(unsure)} rows), annotations_final.csv and annotation_summary.md "
          f"to {args.dir}")


if __name__ == "__main__":
    main()
