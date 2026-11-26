# Reproduction and demo setup

[Back to the project README](../README.md) · [Data guide](data.md)

Commands below use **Windows PowerShell**. Each workflow states its starting directory. The committed CSVs, annotations, screenshots and report describe the completed assignment; running training or analysis again can write generated files and replace recorded outputs.

## Environment

Start at the repository root. The demo's pinned environment was checked with Python 3.13.13. Create a local virtual environment and invoke its Python directly:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .eval_demo_package/demo/requirements.txt
```

The pinned file includes Streamlit, pandas, NumPy, SciPy, scikit-learn, Gensim, joblib, spaCy with `en_core_web_sm`, PyTorch, Transformers and PEFT. It supports the CPU demo and classical rebuild. GPU training needs an environment with a compatible CUDA-enabled PyTorch installation; see [deep-model training](#deep-model-training).

Use the recorded scikit-learn and Gensim versions when creating and loading joblib bundles. The first transformer use requires internet access to download `distilbert-base-uncased`; later loads can use its cache.

## Run the five-model demo

The evaluation/demo package is already included under `.eval_demo_package/`. Its scripts resolve data, models and outputs relative to that folder. Keep it in place; no branch switch or relocation is needed.

### 1. Prepare data and the adapter path

From the repository root, after environment setup:

```powershell
Expand-Archive -LiteralPath imdb_outputs.zip -DestinationPath .eval_demo_package/imdb_outputs
$env:DL_ADAPTER_DIR = (Resolve-Path dl/checkpoints/distilbert_lora).Path
$env:PYTHONHASHSEED = "42"
Set-Location .eval_demo_package
```

Extract once into a fresh destination. The archive's JSONL files and `metadata.json` should be directly inside `.eval_demo_package/imdb_outputs/`.

`DL_ADAPTER_DIR` points to the committed adapter and classification head. Set it before entering the package, so the path is absolute. The pretrained base model is downloaded separately on first use.

### 2. Rebuild classical models and prepare prediction files

Run inside `.eval_demo_package/`:

```powershell
..\.venv\Scripts\python.exe eval/make_ml_preds.py
..\.venv\Scripts\python.exe eval/convert_dl_preds.py --dl-preds ../results/predictions/distilbert_lora.csv
```

`make_ml_preds.py` retrains all four classical pipelines on the fixed training split. It saves feature/classifier bundles in `models/ml/` and test predictions in `results/predictions/preds_<model>.csv`. It compares reproduced test accuracy and F1 with the package's recorded `results/ml_results.csv`, allowing a difference of 0.001. If verification fails, predictions go to `_unverified/` and the script stops; resolve that difference before continuing.

`convert_dl_preds.py` joins the root's saved DL predictions to the prepared test data and produces the evaluation-format `preds_distilbert_lora.csv`. This step uses saved predictions and does not retrain DistilBERT.

### 3. Check predictors and launch

Still inside `.eval_demo_package/`:

```powershell
..\.venv\Scripts\python.exe demo/check_predictors.py --n 50
..\.venv\Scripts\python.exe -m streamlit run demo/app.py --server.address localhost
```

Open [http://localhost:8501](http://localhost:8501). The sidebar should show all five models as loaded. If the port is busy, append `--server.port 8511`. Stop Streamlit with `Ctrl+C`; return to the repository root with `Set-Location ..` when finished.

The app accepts a review, offers preset examples and a hard case with optional gold-label reveal, and compares model labels and positive-class outputs. It also displays model availability, recorded test scores and a warning for long transformer inputs. [Screenshots](../.eval_demo_package/docs/screenshots/README.md) document the completed demo.

Interpret the scores with these model limits:

- LinearSVC outputs are sigmoid-transformed margins, not calibrated probabilities; confidence magnitudes are not directly comparable across models.
- For long reviews, DistilBERT retains 128 beginning and 382 ending wordpieces plus two special tokens, dropping the middle.
- Classical models ignore unseen vocabulary. Word2Vec falls back to a zero vector when no review token has a learned embedding.

## Regenerate evaluation and error analysis

Complete demo setup steps 1–2 first. Evaluation and figure generation also require `matplotlib`, which is not listed in the demo requirements. With Streamlit stopped, run from `.eval_demo_package/`:

```powershell
..\.venv\Scripts\python.exe -m pip install matplotlib
..\.venv\Scripts\python.exe eval/evaluate.py
..\.venv\Scripts\python.exe eval/tag_errors.py
..\.venv\Scripts\python.exe eval/error_analysis.py
..\.venv\Scripts\python.exe eval/make_figures.py
```

| Script | Outputs / role |
|---|---|
| `evaluate.py` | Common model comparison, confusion matrices, misclassified reviews and shared hard cases |
| `tag_errors.py` | Token-length and linguistic heuristic tags; caches the tokenizer needed by error analysis |
| `error_analysis.py` | Cross-checked metrics, final annotation summaries, stratified estimates and AI-assistance analysis |
| `make_figures.py` | Combined confusion-matrix and error-category figures |

Outputs are written under `.eval_demo_package/results/eval/`, where recorded tables, figures and final annotations are already included. Run `tag_errors.py` before `error_analysis.py` on a fresh setup because the latter defaults to offline tokenizer loading. The annotation sample is preserved if it already exists. Recreating or relabeling annotations is not required to reproduce the recorded analysis.

The scripts in `eval/` also include `split_annotation.py`, `label_app.py` and `merge_annotations.py` for the annotation workflow. The submitted final labels are in [`annotations_final.csv`](../.eval_demo_package/results/eval/annotation/annotations_final.csv); analysis uses the included pre-suggestion backup and suggestion/review logs to document AI assistance.

## Classical training script

This workflow runs the original [`ml/model.py`](../ml/model.py), separately from the demo bundle builder. Start at the repository root with the environment above installed:

```powershell
Expand-Archive -LiteralPath imdb_outputs.zip -DestinationPath imdb_outputs
$env:PYTHONHASHSEED = "42"
Set-Location ml
..\.venv\Scripts\python.exe model.py
Set-Location ..
```

Skip extraction if the root `imdb_outputs/` is already prepared. Run the script **from `ml/`** because its data and output paths begin with `../`.

It fits TF-IDF/Word2Vec on train, trains LR/LinearSVC and prints validation/test metrics. It writes `results/ml_config.csv`; it does not export the metric table or demo bundles. The committed [`ml_results.csv`](../results/ml_results.csv) records the reported run. Use `eval/make_ml_preds.py` through the package workflow above to save reusable bundles and per-review test predictions.

## Deep-model training

The committed adapter is sufficient for inference after downloading the base model. Retraining is a separate experiment. The original run used one Tesla T4 on Kaggle; [`dl/README.md`](../dl/README.md) gives Kaggle upload and execution details.

For local GPU training, use an environment with suitable CUDA-enabled PyTorch, install [`dl/requirements.txt`](../dl/requirements.txt), and unpack the data to the root `imdb_outputs/`. From the repository root, with that environment's Python selected:

```powershell
python -m pip install -r dl/requirements.txt
python dl/train_lora.py --stage tune --dry_run --results_dir results/reproduction --ckpt_dir dl/checkpoints/reproduction
python dl/train_lora.py --stage tune --results_dir results/reproduction --ckpt_dir dl/checkpoints/reproduction
python dl/train_lora.py --stage final --results_dir results/reproduction --ckpt_dir dl/checkpoints/reproduction
```

These commands write a separate reproduction run. The script's default output paths contain the submitted tuning results and final checkpoint: with those defaults, completed tuning runs are skipped, and `--stage final` refuses to overwrite an existing `dl_results.csv` unless `--force` is explicitly supplied.

- **Tune:** train eight configurations, select each run's best epoch by validation F1, and record the search table. Test data are not loaded.
- **Final:** select the configuration with the highest validation F1, retrain on train, restore the best validation epoch, then evaluate test and export the adapter, logs and predictions.

Ten epochs run in full; restoring the best checkpoint is not early stopping. `--batch_size 16` can reduce memory use, but changes the experiment from the recorded batch size of 32. `--limit_train`, `--limit_val` and `--limit_test` support small debug runs whose scores are not comparable with the reported full-data results.

## Recorded and generated artifacts

| Included in the repository | Generated during setup or reruns |
|---|---|
| `imdb_outputs.zip` | Unpacked JSONL data and diagnostics |
| `results/ml_results.csv`, `results/ml_config.csv` | Classical bundles in `.eval_demo_package/models/ml/` |
| `results/dl_results.csv`, `results/dl_config.csv` | New training outputs if retraining is requested |
| `results/dl_tuning.csv`, `results/dl_training_log.csv` | Separate tuning/training logs for reproduction runs |
| `results/predictions/distilbert_lora.csv` | Package-format `preds_*.csv` and misclassified-review files |
| `dl/checkpoints/distilbert_lora/` | Downloaded base model and tokenizer cache |
| `.eval_demo_package/results/eval/` tables, annotations and figures | Regenerated evaluation/analysis outputs |

The root `results/` and the package's `results/` are distinct. Package scripts use their own copies of the recorded configuration/metric files. Numerical reproduction can vary with library versions, hardware and nondeterministic operations; the finalized report and committed results remain the reference for the submitted assignment.
