# IMDB Sentiment Analysis: Data Preparation

This repository contains the data preparation work for our NLP course project: binary sentiment classification of English IMDB movie reviews. The notebook covers downloading, quality checks, fixed splits, preprocessing, exports, and draft report sections. Model training and evaluation are handled separately.

## Data and preparation

Source: [stanfordnlp/imdb on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb). Labels are **0 = negative** and **1 = positive**. We use the original 25,000 labeled training reviews and 25,000 labeled test reviews; the separate 50,000 unlabeled reviews are excluded.

We preserved the official test set, removed 123 training records whose exact text appeared in test, and removed 96 remaining duplicate training records. The remaining 24,781 reviews were split into 80% training and 20% validation, stratified by label with **seed 42**.

| Fixed split | Reviews | Negative | Positive |
|---|---:|---:|---:|
| Train | 19,824 | 9,869 | 9,955 |
| Validation | 4,957 | 2,468 | 2,489 |
| Test | 25,000 | 12,500 | 12,500 |
| **Total** | **49,781** | **24,837** | **24,944** |

Minimal cleanup removes HTML tags, decodes HTML entities, and normalizes whitespace. The classical pipeline then applies **spaCy tokenization, stop-word removal, and contextual lemmatization**, with punctuation/whitespace filtering and lowercasing. Negation and selected contrast words are protected; contracted negation is normalized to `not`. Original review text is retained.

Verification found no empty processed reviews, no cross-split exact matches in any exported text version, and no conflicting labels for repeated representations. Every exported record was compared with the prepared data and passed verification. Near-duplicates were not ruled out. The official test set retains 199 extra original-text duplicate rows; preprocessing also creates a few within-split representation duplicates, documented in the output tables.

## For teammates training models

Use the three `imdb_*_prepared.jsonl` files and `metadata.json` in `imdb_outputs/`. JSONL stores one review per line, including token lists.

| Field | Purpose |
|---|---|
| `review_id`, `original_split`, `split` | Record identity and fixed split membership |
| `label` | Classification target: 0 negative, 1 positive |
| `text` | Original review for inspection and error analysis |
| `text_clean` | Minimally cleaned natural text for a transformer's own tokenizer |
| `tokens_classical` | Filtered, normalized token list for classical models |
| `text_classical` | The same classical tokens joined with spaces |
| Token-count fields | Preprocessing and filtering diagnostics |

Load a split from the repository root:

```python
import pandas as pd

train = pd.read_json("imdb_outputs/imdb_train_prepared.jsonl", lines=True)
val = pd.read_json("imdb_outputs/imdb_val_prepared.jsonl", lines=True)
test = pd.read_json("imdb_outputs/imdb_test_prepared.jsonl", lines=True)
```

Keep these splits unchanged. Fit TF-IDF vocabulary/IDF weights or other learned representations on training data only; select settings using validation data and evaluate finally on test. For classical models, preserve the prepared tokens and avoid another stop-word filter. For transformers, use `text_clean` with the selected model's tokenizer.

## File guide

Section numbers refer to `imdb_data_preprocessing.ipynb`. All generated files below are in **`imdb_outputs/`**, except the ZIP archive. “Saved in” identifies the section that writes the file, even when the check was calculated earlier.

| File | Saved in notebook section | Purpose |
|---|---|---|
| `split_manifest.csv` | 5: Fixed splits | IDs, source/project split membership, and labels for every retained record; no review text |
| `removed_training_records.csv` | 5: Fixed splits | IDs and removal reasons for the 219 excluded training records |
| `split_summary.csv` | 5: Fixed splits | Final split sizes, label counts, and percentages for reporting |
| `label_distribution.png` | 6: Dataset description | Positive/negative count chart across splits |
| `training_review_length_summary.csv` | 6: Dataset description | Training length statistics measured by whitespace-separated words |
| `training_review_lengths.png` | 6: Dataset description | Training length histogram; the longest 1% is omitted from the plot only |
| `training_sample_examples.csv` | 6: Dataset description | Two full training reviews, one per label, with IDs and word counts |
| `preprocessing_summary.csv` | 11: Batch preprocessing | Per-split mean token counts and empty-output checks |
| `representation_overlap_summary.csv` | 12: Verification/export | Cross-split exact overlap checks for original, cleaned, and classical text |
| `representation_duplicate_summary.csv` | 12: Verification/export | Within-split repeated representations and conflicting-label checks |
| `representation_overlap_records.jsonl` | 12: Verification/export | Investigation records for cleaned/classical cross-split overlap; this run contains no records |
| `imdb_train_prepared.jsonl` | 12: Verification/export | Full prepared training data: 19,824 records |
| `imdb_val_prepared.jsonl` | 12: Verification/export | Full prepared validation data: 4,957 records |
| `imdb_test_prepared.jsonl` | 12: Verification/export | Full prepared official test data: 25,000 records |
| `metadata.json` | 12: Verification/export | Versions, configuration, preprocessing/split policies, results, and export SHA-256 checksums |
| `original_data_quality.csv` | 12; calculated in 4 | Original split checks for invalid text/labels, duplicates, and conflicting labels |
| `original_train_test_overlap.csv` | 12; calculated in 4 | Original train/test overlap findings before removal |
| `training_removal_summary.csv` | 12; calculated in 5 | Row counts after removal stages: 25,000 → 24,877 → 24,781 |
| `processing_times.csv` | 12; calculated in 11 | Batch preprocessing time for each split |
| `imdb_outputs.zip` (repository root) | 12: Archive/download | Downloaded archive of the output files; duplicates the extracted folder. The supplied notebook creates the archive as `imdb_prepared_outputs.zip` |

Sections 1–4 and 7–10 do not directly save output files. Section 13 contains report drafts in Markdown, with a provisional conclusion awaiting model results.

Other repository files:

| File | Purpose |
|---|---|
| `imdb_data_preprocessing.ipynb` | Main notebook with code, explanations, outputs, and report drafts |
| `nlp_Assignments_v1.1_english.pdf` | Course assignment reference |
| `README.md` | Workflow summary and teammate/file guide |

## Reproducing and sharing

The notebook was executed in Google Colab on CPU. Run its cells in order to reproduce preparation. It installs `datasets`, `pandas`, `matplotlib`, `spacy`, and the `en_core_web_sm` model; the recorded versions are in `metadata.json`. The notebook defaults to `outputs/`, while this repository stores the downloaded files in `imdb_outputs/`. Adjust `OUTPUT_DIR` if rerunning locally into the existing folder.

The original dataset download did not pin a Hugging Face revision. Reuse the verified exports and checksums for consistent team experiments.

**GitHub file size:** `imdb_test_prepared.jsonl` is approximately 108 MiB, above GitHub's 100 MiB limit for regular Git files. Store it with Git LFS or distribute the ZIP through a GitHub Release. If using LFS, ensure teammates download the actual data rather than a pointer file. See [GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Reference

Maas et al. (2011). [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/).
