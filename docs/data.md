# Data preparation and exports

[Back to the project README](../README.md) · [Reproduction guide](reproduction.md)

## Source and fixed splits

The [preprocessing notebook](../imdb_data_preprocessing.ipynb) loads [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb): 25,000 labeled training reviews, 25,000 labeled test reviews and a separate unlabeled partition. Only the labeled data are used. Labels are **0 = negative** and **1 = positive**.

The split policy is:

1. Preserve the official 25,000-review test set.
2. Remove 123 original training records whose exact review text also occurs in test.
3. Remove 96 remaining duplicate training records.
4. Split the remaining 24,781 reviews into 80% training and 20% validation, stratified by label with seed 42.

| Split | Reviews | Negative | Positive |
|---|---:|---:|---:|
| Train | 19,824 | 9,869 | 9,955 |
| Validation | 4,957 | 2,468 | 2,489 |
| Test | 25,000 | 12,500 | 12,500 |
| **Total** | **49,781** | **24,837** | **24,944** |

Reuse the exported splits for all experiments. Fit learned representations on training records only. The original dataset download did not pin a Hugging Face revision; the archived exports and their SHA-256 checksums provide the reproducible data reference.

## Text representations

The notebook retains the original text and produces separate inputs for the classical and transformer branches.

| Representation | Processing | Consumer |
|---|---|---|
| `text` | Original review | Inspection, traceability and error analysis |
| `text_clean` | Remove HTML, decode entities, normalize whitespace | DistilBERT tokenizer |
| `tokens_classical` | spaCy tokenization, punctuation/whitespace filtering, stop-word removal, contextual lemmatization and lowercasing | Word2Vec |
| `text_classical` | Classical token list joined with spaces | TF-IDF |

Negation and selected contrast words are protected from stop-word filtering. Contracted negation is normalized to `not`. Avoid applying an additional stop-word filter to the prepared classical inputs.

## Prepared record schema

The archive contains one JSON object per line in each `imdb_*_prepared.jsonl` file.

| Field | Meaning |
|---|---|
| `review_id` | Stable review identifier, used to join predictions and annotations |
| `original_split` | Source dataset split |
| `split` | Project split: train, val or test |
| `label` | Target: 0 negative, 1 positive |
| `text`, `text_clean` | Original and minimally cleaned review |
| `tokens_classical`, `text_classical` | Prepared classical tokens and joined text |
| `token_count_original`, `token_count_classical` | Token counts before and after classical filtering |
| `stopword_count_removed`, `punctuation_count_removed`, `whitespace_count_removed`, `protected_token_count` | Preprocessing diagnostics |

After unpacking the archive into `imdb_outputs/`, load it from the repository root:

```python
import pandas as pd

train = pd.read_json("imdb_outputs/imdb_train_prepared.jsonl", lines=True)
val = pd.read_json("imdb_outputs/imdb_val_prepared.jsonl", lines=True)
test = pd.read_json("imdb_outputs/imdb_test_prepared.jsonl", lines=True)
```

For package-local evaluation and demo setup, the scripts expect their own `imdb_outputs/` directory under `.eval_demo_package/`; see the [reproduction guide](reproduction.md).

## Quality checks

The recorded preparation checks found:

- No empty processed reviews and no cross-split exact matches in `text`, `text_clean` or `text_classical`.
- No conflicting labels among repeated representations.
- All exported records passed comparison with the prepared in-memory data.
- The unchanged official test set retains 199 extra duplicate rows by original text. Cleaning and classical preprocessing introduce additional within-split representation duplicates, recorded in the diagnostics.

These checks concern exact text representations. They do not rule out near-duplicates or establish that every dataset label is correct.

## Archive contents

[`imdb_outputs.zip`](../imdb_outputs.zip) contains the following files directly at the archive root. Paths below are relative to the extraction directory.

| File(s) | Purpose |
|---|---|
| `imdb_train_prepared.jsonl`, `imdb_val_prepared.jsonl`, `imdb_test_prepared.jsonl` | Full prepared fixed splits |
| `metadata.json` | Package/model versions, split and preprocessing policies, checks and export SHA-256 checksums |
| `split_manifest.csv`, `split_summary.csv` | Retained record IDs, source/project membership, labels and split counts |
| `removed_training_records.csv`, `training_removal_summary.csv` | IDs/reasons for exclusions and row counts at each removal stage |
| `original_data_quality.csv`, `original_train_test_overlap.csv` | Original-data validation, duplicate and overlap findings |
| `preprocessing_summary.csv`, `processing_times.csv` | Per-split processing diagnostics and timings |
| `representation_overlap_summary.csv`, `representation_duplicate_summary.csv` | Cross-split exact overlap and within-split duplicate checks |
| `representation_overlap_records.jsonl` | Investigated cleaned/classical cross-split overlaps; empty for the recorded run |
| `training_review_length_summary.csv`, `training_sample_examples.csv` | Training length statistics and one full sample per label |
| `label_distribution.png`, `training_review_lengths.png` | Dataset figures; the length plot omits the longest 1% for display only |

Notebook Sections 5–6 create split/data-description artifacts, Section 11 performs batch preprocessing, and Section 12 verifies and exports the prepared data. Section 13 contains early report drafts; the [final report](../.report/NLP_CC01_Group04_Report.pdf) provides the completed findings.

## Rerunning preparation

The notebook was executed in Google Colab on CPU. Run its cells in order; setup installs `datasets`, `pandas`, `matplotlib`, `spacy` and `en_core_web_sm`. Recorded preprocessing uses spaCy 3.8.16 with `en_core_web_sm` 3.8.0; all recorded versions are in `metadata.json`.

The notebook defaults to `outputs/` and creates `imdb_prepared_outputs.zip`. The submitted archive is named `imdb_outputs.zip`. Set `OUTPUT_DIR` deliberately when rerunning locally, and retain the supplied exports when reproducing the reported model results.
