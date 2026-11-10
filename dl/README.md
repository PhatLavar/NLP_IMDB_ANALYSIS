# DL model: DistilBERT + LoRA

`train_lora.py` fine-tunes `distilbert-base-uncased` with LoRA on the fixed IMDB
splits (input column: `text_clean`, target: `label`, 1 = positive).

## Protocol

1. `--stage tune` trains each configuration on **train** and scores it on **val**
   only. The test set is not loaded. The best epoch of every run is picked by
   validation F1. Results: `results/dl_tuning.csv`, `results/dl_training_log.csv`.
2. `--stage final` retrains the configuration with the highest validation F1 (or
   `--run_id`), then evaluates **test once**. It refuses to overwrite an existing
   `results/dl_results.csv` unless `--force` is given.

The search is one-factor-at-a-time around r=8, q/v, lr=3e-4, 256 tokens (see
`default_runs()` in the script; edit it to change the grid).

## Run on Kaggle

1. Upload `imdb_outputs.zip` as a private Kaggle Dataset (Kaggle unzips it) and add
   it to the notebook. Upload `dl/train_lora.py` the same way (or paste it).
2. Notebook settings: Accelerator = GPU (T4 or P100), Internet = On (needed to
   download the model from Hugging Face and to install `peft`).
3. Cells:

   ```python
   !pip install -q peft
   !python /kaggle/input/<your-script-dataset>/train_lora.py --stage tune
   !python /kaggle/input/<your-script-dataset>/train_lora.py --stage final
   ```

   The data directory is detected automatically under `/kaggle/input`.
   Outputs are written to `/kaggle/working/results` and `/kaggle/working/checkpoints`.
   Tuning is resumable: rerun the same command after an interruption and finished
   runs are skipped.
4. Download `/kaggle/working/results/*` (including `predictions/`) and copy it
   into this repository's `results/` folder. Do not commit `checkpoints/`.

## Run locally

```bash
unzip imdb_outputs.zip -d imdb_outputs   # once, from the repository root
pip install -r dl/requirements.txt
python dl/train_lora.py --stage tune --dry_run   # list the planned runs
python dl/train_lora.py --stage tune
python dl/train_lora.py --stage final
```

Useful flags: `--batch_size 16` (if a 512-token run runs out of memory),
`--runs <id1,id2>` (subset of runs), `--limit_train/--limit_val/--limit_test N`
(quick debug runs on a subsample).

## Output files (`results/`)

| File | Content |
|---|---|
| `dl_tuning.csv` | One row per tuning run: best epoch, validation metrics, time, memory, config JSON |
| `dl_training_log.csv` | Per-epoch `run_id, stage, epoch, train_loss, val_loss, val_accuracy, val_f1, epoch_time_seconds` |
| `dl_results.csv` | Final model; same columns as `ml_results.csv` |
| `dl_config.csv` | One-row configuration of the final model (hyperparameters, parameter counts, GPU, library versions) |
| `predictions/distilbert_lora.csv` | `review_id, split, label, pred, score` for val and test; `score` = P(positive) |

`training_time_seconds` is the sum of the training epochs of the final run
(validation passes excluded). Metrics use the positive class 1 with binary
averaging, like `ml/model.py`.
