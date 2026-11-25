"""Local hand-labelling tool for the error-analysis sample: one review per screen.

Run (from the repository root, PowerShell):
    streamlit run eval/label_app.py --server.address localhost --server.port 8502
        labels both files of results/eval/annotation/ (annotator_A_*.csv and annotator_B_*.csv; switch in the
        sidebar). It opens the first file that still has an unlabelled review.
    streamlit run eval/label_app.py --server.address localhost --server.port 8502 -- --file <path to annotator csv>
        labels only that file.
More options (after "--"):
    --suggestions <csv>  AI suggestions (default: results/eval/suggestions/llm_suggestions.csv if it exists;
                         without a suggestions file there is no suggestion box)
    --log <csv>          decision log (default: results/eval/suggestions/review_log.csv)
    --dir <folder>       folder with the two annotator files (default: results/eval/annotation)

What it changes: only the columns category, unsure and note of an annotator file; every other cell and the row
order stay as they are. Before the first save to each file in a run, that file is copied to
reports/label_backups/<name>_<time>.csv. Each save writes a temp file in the same folder and then replaces the
file (os.replace), as CSV UTF-8 (utf-8-sig). Each save also appends one row to the decision log
(accepted / edited / no_suggestion); the last row of an id is its final decision.

What it shows: the true label, the DL model's prediction, the AI suggestion (clearly marked as not ground truth),
the cleaned review text and the part of a long review the DL model never saw. A review without a saved category
is pre-filled from the AI suggestion; nothing is written until "Save and next". On purpose it does NOT show
heuristic tags or other models' predictions (the hand labels are used to check those heuristics).
"""
import argparse
import csv
import fnmatch
import html
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from merge_annotations import ANNOTATOR_PATTERNS, CATEGORIES, category_problem, normalize  # noqa: E402  (read-only)
from preprocess import clean_review  # noqa: E402  (the minimal cleanup that gives the DL model's text_clean)

ANNOTATION_DIR = ROOT / "results/eval/annotation"
BACKUP_DIR = ROOT / "reports/label_backups"  # not inside the annotation folder: merge_annotations.py globs *.csv there
SUGGESTIONS_DEFAULT = ROOT / "results/eval/suggestions/llm_suggestions.csv"
LOG_DEFAULT = ROOT / "results/eval/suggestions/review_log.csv"
COLUMNS = ["id", "text", "label", "pred_dl", "prob_dl", "category", "unsure", "note"]
SUGGESTION_COLUMNS = ["id", "source", "main_category", "also_applies", "unsure", "alt_category", "reason",
                      "suggested_note"]
LOG_COLUMNS = ["id", "source", "saved_at", "suggested_main", "suggested_also", "suggested_unsure",
               "final_main", "final_also", "final_unsure", "decision"]
LABELS = {"0": "Negative", "1": "Positive"}

# Copied from results/eval/annotation_guide.md (the column rules and "How to label").
GUIDE_RULES = """\
- `category`: one or more of `sarcasm`, `negation`, `mixed`, `long_truncated`, `label_noise`, `other`. \
Separate several values with `;` and put the main reason first, for example `negation;mixed`.
- `unsure`: write `yes` if you cannot decide between categories or you doubt your choice. Otherwise leave it \
empty. Fill in `category` with your best guess either way.
- `note`: optional and short, for example the key phrase.

1. Read the true `label` first (0 = negative, 1 = positive), then read the full `text`.
2. Decide whether the text clearly supports the model's prediction (`pred_dl`) instead of the label. \
If it does, the label is probably wrong: use `label_noise`.
3. Otherwise, choose the reason or reasons the text was hard to classify. Label from the text only.
"""
# Copied from the "Categories" table of results/eval/annotation_guide.md (definition column).
DEFINITIONS = {
    "sarcasm": "The literal words say the opposite of what the writer means.",
    "negation": "A negation flips a sentiment word, and the model probably missed it.",
    "mixed": "The review praises some parts and criticises others, and the overall verdict is subtle or comes late.",
    "long_truncated": "The review is longer than the model's 512-token limit, and the decisive opinion sits in the "
                      "part that was cut. The DL model keeps the beginning (about 25%) and the end of a long review, "
                      "and drops the middle (`truncation = head_tail` in `results/dl_config.csv`).",
    "label_noise": "The text clearly agrees with the model's prediction, not with the true label.",
    "other": "None of the above. Explain the reason in `note`.",
}


# ---------------------------------------------------------------- file handling (no Streamlit needed)
def annotator_files(folder):
    """{'A': path, 'B': path}: exactly one annotator_A_*.csv and one annotator_B_*.csv (as in merge_annotations.py)."""
    found = {letter: sorted(folder.glob(pattern)) for letter, pattern in ANNOTATOR_PATTERNS.items()}
    problems = [f"{ANNOTATOR_PATTERNS[letter]}: found {len(paths)} {[p.name for p in paths]}"
                for letter, paths in found.items() if len(paths) != 1]
    if problems:
        raise ValueError(f"expected exactly one file per pattern in {folder}: {'; '.join(problems)}. "
                         "Pass --file <path> or --dir <folder>.")
    return {letter: paths[0] for letter, paths in found.items()}


def file_letter(path):
    """'A' or 'B' from the file name (annotator_A_*.csv / annotator_B_*.csv), else the file name itself."""
    return next((letter for letter, pattern in ANNOTATOR_PATTERNS.items() if fnmatch.fnmatch(path.name, pattern)),
                path.name)


def load_file(path):
    """Read every cell as text (empty cells stay ''); refuse unexpected columns or duplicate ids."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        raise ValueError(f"{path.name} is not valid UTF-8. Save it as CSV UTF-8 (in Excel: Save As > CSV UTF-8 "
                         "(Comma delimited); in Google Sheets: File > Download > CSV).") from None
    if list(df.columns) != COLUMNS:
        raise ValueError(f"{path.name}: the columns must be exactly {COLUMNS}, found {list(df.columns)}")
    dupes = sorted(df.loc[df["id"].duplicated(), "id"].unique())
    if dupes:
        raise ValueError(f"{path.name}: duplicate ids {dupes}")
    return df


def make_category(main, extras):
    """'main;extra1;extra2', checked with the rules of eval/merge_annotations.py."""
    if not main:
        raise ValueError("choose a main category first")
    category = ";".join([main, *extras])
    problem = category_problem(category)
    if problem:
        raise ValueError(f"category {category!r}: {problem}")
    return normalize(category)


def backup(path):
    """Copy the file to reports/label_backups/<name>_<time>.csv and return the copy's path."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(path, target)
    return target


def save_label(path, review_id, main, extras, unsure, note):
    """Set category/unsure/note of one review and rewrite the file atomically. Returns the saved category."""
    category = make_category(main, extras)
    df = load_file(path)
    match = df.index[df["id"] == review_id]
    if len(match) != 1:
        raise ValueError(f"id {review_id} is not in {path.name}")
    df.loc[match[0], ["category", "unsure", "note"]] = [category, "yes" if unsure else "", note.strip()]
    tmp = path.with_name(path.name + ".tmp")  # same folder, so os.replace is atomic; not *.csv, so no glob sees it
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # only left over if the replace failed (e.g. the file is open in Excel)
    return category


def next_position(labelled, pos):
    """Next review without a category after pos (then from the top), else simply the next review."""
    n = len(labelled)
    for i in [*range(pos + 1, n), *range(pos)]:
        if not labelled[i]:
            return i
    return min(pos + 1, n - 1)


# ---------------------------------------------------------------- AI suggestions and the decision log
def resolve_suggestions(arg):
    """--suggestions if given, else the default file if it exists, else None (no suggestions)."""
    if arg is not None:
        return arg
    return SUGGESTIONS_DEFAULT if SUGGESTIONS_DEFAULT.is_file() else None


def load_suggestions(path, ids_by_source):
    """Valid suggestions as {id: dict} and the skipped rows as [(id, reason)]; a bad row is skipped, never fatal.

    ids_by_source = {'A': ids of the open A file, 'B': ...}. Rows for an annotator whose file is not open
    (--file mode) are ignored, not counted as skipped.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [c for c in SUGGESTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    valid, skipped = {}, []
    for r in df.itertuples(index=False):
        source = r.source.strip()
        if source in ANNOTATOR_PATTERNS and source not in ids_by_source:
            continue  # a suggestion for the other annotator's file, which is not open
        main = r.main_category.strip().lower()
        also = [p.strip().lower() for p in r.also_applies.split(";")] if r.also_applies.strip() else []
        unsure, alt = r.unsure.strip().lower(), r.alt_category.strip().lower()
        if source not in ANNOTATOR_PATTERNS:
            problem = f"source must be A or B, found {r.source!r}"
        elif r.id not in ids_by_source[source]:
            problem = "unknown id"
        elif main in also:
            problem = "main category repeated in also_applies"
        elif category_problem(";".join([main, *also])):
            problem = "invalid category: " + category_problem(";".join([main, *also]))
        elif unsure not in ("", "yes"):
            problem = f"unsure must be yes or empty, found {r.unsure!r}"
        elif unsure == "yes" and not alt:
            problem = "unsure without alt_category"
        elif alt and alt not in CATEGORIES:
            problem = f"invalid alt_category {r.alt_category!r}"
        elif not r.reason.strip():
            problem = "empty reason"
        elif r.id in valid:
            problem = "duplicate id"
        else:
            problem = None
        if problem:
            skipped.append((r.id, problem))
        else:
            valid[r.id] = {"main": main, "also": also, "unsure": unsure == "yes", "alt": alt,
                           "reason": r.reason.strip(), "note": r.suggested_note.strip()}
    return valid, skipped


def decision(suggestion, main, extras, unsure):
    """accepted / edited / no_suggestion: compares main, the extra categories (as a set) and unsure."""
    if suggestion is None:
        return "no_suggestion"
    same = (main == suggestion["main"] and set(extras) == set(suggestion["also"])
            and bool(unsure) == suggestion["unsure"])
    return "accepted" if same else "edited"


def append_log(log_path, review_id, source, suggestion, main, extras, unsure):
    """Append one row to the decision log (created with its header on the first save), flushed to disk."""
    row = {"id": review_id, "source": source, "saved_at": datetime.now().isoformat(timespec="seconds"),
           "suggested_main": suggestion["main"] if suggestion else "",
           "suggested_also": ";".join(suggestion["also"]) if suggestion else "",
           "suggested_unsure": "yes" if suggestion and suggestion["unsure"] else "",
           "final_main": main, "final_also": ";".join(extras), "final_unsure": "yes" if unsure else "",
           "decision": decision(suggestion, main, extras, unsure)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    # utf-8-sig writes the BOM only at the start of a new file; appending to a non-empty file adds none
    with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    return row["decision"]


def decision_counts(log_path):
    """{decision: count} over the last log row of each id ({} if there is no log yet)."""
    if not log_path.is_file():
        return {}
    log = pd.read_csv(log_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return log.drop_duplicates("id", keep="last")["decision"].value_counts().to_dict()


# ---------------------------------------------------------------- what the DL model sees
def load_dl_settings():
    """model_name, max_length, head_frac of the final DL run (results/dl_config.csv); only head_tail is supported."""
    cfg = pd.read_csv(ROOT / "results/dl_config.csv", encoding="utf-8").iloc[0]
    if cfg["truncation"] != "head_tail":
        raise ValueError(f"results/dl_config.csv has truncation = {cfg['truncation']!r}; "
                         "this tool can only show head_tail truncation")
    return cfg["model_name"], int(cfg["max_length"]), float(cfg["head_frac"])


@st.cache_resource
def load_tokenizer(model_name):
    """The DistilBERT tokenizer from models/hf_cache (the cache demo/predictors.py uses), offline."""
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()  # hide the expected "longer than 512" warning
    return AutoTokenizer.from_pretrained(model_name, cache_dir=ROOT / "models/hf_cache", local_files_only=True)


def dl_view(text_clean, tokenizer, max_length, head_frac):
    """What the DL model reads of text_clean: head_tail as in build_inputs, dl/train_lora.py:193-205.

    Returns (ids, kept_ids, dropped): all wordpiece ids (no special tokens), the ids the model keeps, and
    (start, end) character positions of the dropped middle in text_clean, or None if nothing is dropped.
    """
    enc = tokenizer(text_clean, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    budget = max_length - 2  # [CLS] and [SEP]
    if len(ids) <= budget:
        return ids, ids, None
    head = int(round(budget * head_frac))
    tail = budget - head
    kept = ids[:head] + ids[len(ids) - tail:]
    return ids, kept, (offsets[head][0], offsets[len(ids) - tail - 1][1])


def review_html(text_clean, dropped, n_dropped):
    """The review as HTML (text escaped); the part the model did not see is grey, between two marker lines."""
    if dropped is None:
        return f'<div style="white-space: pre-wrap">{html.escape(text_clean)}</div>'
    start, end = dropped
    grey = "color: #888888"
    return (f'<div style="white-space: pre-wrap">{html.escape(text_clean[:start])}'
            f'<div style="{grey}; font-weight: bold">[... {n_dropped} wordpieces not seen by the model ...]</div>'
            f'<span style="{grey}">{html.escape(text_clean[start:end])}</span>'
            f'<div style="{grey}; font-weight: bold">[... end of the part not seen by the model ...]</div>'
            f'{html.escape(text_clean[end:])}</div>')


# ---------------------------------------------------------------- the screen
def main():
    parser = argparse.ArgumentParser(description="Hand-label the annotator files, one review per screen.")
    parser.add_argument("--file", type=Path, default=None, help="label only this annotator CSV")
    parser.add_argument("--dir", type=Path, default=ANNOTATION_DIR,
                        help="folder with annotator_A_*.csv and annotator_B_*.csv (default: results/eval/annotation)")
    parser.add_argument("--suggestions", type=Path, default=None,
                        help="AI suggestions CSV (default: results/eval/suggestions/llm_suggestions.csv if it exists)")
    parser.add_argument("--log", type=Path, default=LOG_DEFAULT,
                        help="decision log CSV (default: results/eval/suggestions/review_log.csv)")
    args = parser.parse_args()

    st.set_page_config(page_title="Labelling tool")
    try:
        files = {file_letter(args.file): args.file.resolve()} if args.file else annotator_files(args.dir.resolve())
        log_path = args.log.resolve()
        if log_path.parent in {p.parent for p in files.values()} | {args.dir.resolve()}:
            raise ValueError(f"--log must not be in the annotation folder {log_path.parent} "
                             "(merge_annotations.py reads the *.csv files there)")
        frames = {letter: load_file(path) for letter, path in files.items()}
        model_name, max_length, head_frac = load_dl_settings()
        tokenizer = load_tokenizer(model_name)
    except (ValueError, OSError) as e:
        st.error(f"Cannot start: {e}")
        st.stop()

    suggestions, skipped, suggestions_error = {}, [], None
    suggestions_path = resolve_suggestions(args.suggestions)
    if suggestions_path is not None:
        try:
            suggestions, skipped = load_suggestions(suggestions_path, {k: set(v["id"]) for k, v in frames.items()})
        except (ValueError, OSError) as e:  # missing file, not UTF-8, wrong columns: work without suggestions
            suggestions_error = f"Suggestions not loaded from {suggestions_path}: {e}"

    # which file: the sidebar choice (both files), else the only one (--file)
    letters = list(files)
    if "file_choice" not in st.session_state:  # first run: the first file that still has an unlabelled review
        st.session_state.file_choice = next(
            (k for k in letters if (frames[k]["category"].str.strip() == "").any()), letters[0])
    if len(letters) > 1:
        st.sidebar.selectbox("Annotator file", letters, key="file_choice", format_func=lambda k: f"{k}: {files[k].name}")
    letter = st.session_state.file_choice
    path, df = files[letter], frames[letter]

    ids = df["id"].tolist()
    labelled = (df["category"].str.strip() != "").tolist()
    pos_key, jump_key = f"pos_{letter}", f"jump_{letter}"
    if pos_key not in st.session_state:  # first visit of this file: open at its first review without a category
        st.session_state[pos_key] = next_position(labelled, -1)
    if jump_key not in st.session_state:
        st.session_state[jump_key] = ids[st.session_state[pos_key]]

    def go(pos):
        st.session_state[pos_key] = pos
        st.session_state[jump_key] = ids[pos]

    def on_jump():
        st.session_state[pos_key] = ids.index(st.session_state[jump_key])

    def on_save(pos):
        rid = ids[pos]
        main_value = st.session_state.get(f"main_{rid}")
        extras = st.session_state.get(f"extra_{rid}_{main_value}", [])
        unsure = st.session_state[f"unsure_{rid}"]
        try:
            make_category(main_value, extras)  # check before the backup is made
            if f"backup_{letter}" not in st.session_state:  # first save to this file in this run
                st.session_state[f"backup_{letter}"] = str(backup(path))
            category = save_label(path, rid, main_value, extras, unsure, st.session_state[f"note_{rid}"])
        except (ValueError, OSError) as e:
            st.session_state.message = ("error", f"Not saved: {e}")
            return
        try:
            result = append_log(log_path, rid, letter, suggestions.get(rid), main_value, extras, unsure)
            st.session_state.message = ("success", f"Saved {rid}: category = {category} ({result})")
        except OSError as e:
            st.session_state.message = ("error", f"Saved {rid}, but the decision log was not written: {e}")
        go(next_position((load_file(path)["category"].str.strip() != "").tolist(), pos))

    pos = st.session_state[pos_key]
    row = df.iloc[pos]
    rid = row["id"]
    suggestion = suggestions.get(rid)

    # 1. progress
    st.subheader(f"{sum(labelled)} of {len(df)} labelled")
    st.progress(sum(labelled) / len(df))
    if len(frames) > 1:
        done = sum(int((f["category"].str.strip() != "").sum()) for f in frames.values())
        st.caption(f"{done} of {sum(len(f) for f in frames.values())} labelled in total (A + B)")
    st.caption(f"Review {pos + 1} of {len(df)} · id {rid} · {path}")
    if "message" in st.session_state:
        kind, message = st.session_state.pop("message")
        (st.error if kind == "error" else st.success)(message)

    # 2. true label and the DL prediction, then the AI suggestion
    st.markdown(f"True label: **{LABELS.get(row['label'], row['label'])}** · DistilBERT prediction: "
                f"**{LABELS.get(row['pred_dl'], row['pred_dl'])}** · P(positive) = {float(row['prob_dl']):.2f}")
    if suggestion:
        with st.container(border=True):
            st.markdown("**AI suggestion (not ground truth)**")
            unsure_text = f"yes (alternative: `{suggestion['alt']}`)" if suggestion["unsure"] else "no"
            st.markdown(f"Main category: `{suggestion['main']}` · Also applies: "
                        f"{', '.join(f'`{c}`' for c in suggestion['also']) or 'none'} · Unsure: {unsure_text}")
            st.markdown(f"Reason: {suggestion['reason']}")
            st.markdown(f"Suggested note: {suggestion['note'] or '(none)'}")

    # 3. + 4. the cleaned review, with the part the DL model did not see in grey
    text_clean = clean_review(row["text"])
    all_ids, kept, dropped = dl_view(text_clean, tokenizer, max_length, head_frac)
    count = f"{len(all_ids)} wordpieces; the DL model reads at most {max_length - 2} (plus [CLS] and [SEP])"
    st.caption(count + (f": {len(all_ids) - len(kept)} in the middle are not seen (grey)." if dropped else "."))
    st.html(review_html(text_clean, dropped, len(all_ids) - len(kept)))

    # 5. inputs: pre-filled from the saved values, else from the AI suggestion (nothing is written until Save)
    if row["category"].strip():
        parts = [p.strip().lower() for p in row["category"].split(";")]
        main_default, also_default = parts[0], parts[1:]
        unsure_default, note_default = row["unsure"].strip() == "yes", row["note"]
        st.info("Saved by you")
    elif suggestion:
        main_default, also_default = suggestion["main"], suggestion["also"]
        unsure_default, note_default = suggestion["unsure"], suggestion["note"]
        st.warning("Pre-filled from the AI suggestion - not saved yet")
    else:
        main_default, also_default, unsure_default, note_default = None, [], False, ""
    main_value = st.selectbox("Main category", CATEGORIES, key=f"main_{rid}", placeholder="Choose the main reason",
                              index=CATEGORIES.index(main_default) if main_default in CATEGORIES else None)
    options = [c for c in CATEGORIES if c != main_value]
    st.multiselect("Also applies (optional)", options, default=[p for p in also_default if p in options],
                   key=f"extra_{rid}_{main_value}")  # key includes main: options change with it
    st.checkbox("unsure", value=unsure_default, key=f"unsure_{rid}")
    st.text_input("note (optional, short)", value=note_default, key=f"note_{rid}")

    # 6. buttons and jump
    save_col, prev_col, next_col = st.columns(3)
    save_col.button("Save and next", type="primary", on_click=on_save, args=(pos,))
    prev_col.button("Previous", on_click=go, args=(max(pos - 1, 0),), disabled=pos == 0)
    next_col.button("Next", on_click=go, args=(min(pos + 1, len(df) - 1),), disabled=pos == len(df) - 1)
    st.selectbox("Jump to id", ids, key=jump_key, on_change=on_jump)

    # sidebar: suggestions status, decision counters, the guide
    if suggestions_error:
        st.sidebar.error(suggestions_error)
    elif suggestions_path is not None:
        st.sidebar.caption(f"{len(suggestions)} AI suggestions loaded from {suggestions_path}")
        if skipped:
            st.sidebar.warning(f"{len(skipped)} suggestions skipped: "
                               + "; ".join(f"{review_id} ({why})" for review_id, why in skipped))
    try:
        counts = decision_counts(log_path)
        st.sidebar.markdown(f"Decisions (last save per id): accepted {counts.get('accepted', 0)} / "
                            f"edited {counts.get('edited', 0)} / no suggestion {counts.get('no_suggestion', 0)}")
    except (ValueError, OSError, KeyError) as e:
        st.sidebar.error(f"Decision log not readable ({log_path}): {e}")
    st.sidebar.header("How to label")
    st.sidebar.markdown(GUIDE_RULES)
    st.sidebar.header("Categories")
    st.sidebar.markdown("\n".join(f"- **{value}**: {text}" for value, text in DEFINITIONS.items()))


if __name__ == "__main__":
    main()
