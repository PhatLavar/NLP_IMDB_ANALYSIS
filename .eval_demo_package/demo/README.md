# IMDB sentiment demo

Type a movie review and see, for each model, the predicted label (Positive / Negative), the confidence and P(positive).

## Setup (Windows PowerShell, from the repository root)

The trained ML models are not in the repository (`models/` is gitignored). Do these steps once.

1. **Python and packages.** Python 3.13 (the checks used 3.13.13):

   ```powershell
   py -3.13 -m venv ..\.venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # only if Activate.ps1 says running scripts is disabled
   ..\.venv\Scripts\Activate.ps1
   pip install -r demo/requirements.txt
   ```

   `..\.venv` is the folder ABOVE the repository root, not inside it. The virtual environment can live anywhere;
   change the path if yours is elsewhere (in cmd.exe activate with
   `..\.venv\Scripts\activate.bat`). `requirements.txt` pins the versions used for the checks: streamlit 1.64.0,
   altair 6.3.0, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, gensim 4.4.0, joblib 1.6.0,
   transformers 5.20.2, tokenizers 0.23.3, spacy 3.8.16 with en_core_web_sm 3.8.0, torch 2.11.0 (the CPU build from
   PyPI, about 115 MB to download and about 600 MB on disk with its dependencies) and peft 0.21.2. The ML models are
   joblib files, so load them with the same scikit-learn and gensim versions that wrote them.

   About 330 MB of wheels are downloaded in total. Two messages are harmless and can be ignored: torch's
   "Redirects are currently not supported in Windows or MacOs" and the Hugging Face "unauthenticated requests"
   warning.

2. **Prepared data.** Unzip the committed archive into `imdb_outputs/` (gitignored):

   ```powershell
   Expand-Archive imdb_outputs.zip -DestinationPath imdb_outputs
   ```

3. **The 4 ML models** (`models/ml/tfidf_logreg.joblib`, `tfidf_svm.joblib`, `w2v_logreg.joblib`, `w2v_svm.joblib`):

   ```powershell
   $env:PYTHONHASHSEED = "42"
   python eval/make_ml_preds.py
   ```

   The script uses the settings of `ml/model.py`, fits on the train split only and saves `models/ml/<name>.joblib`.
   It then compares its test accuracy and F1 with `results/ml_results.csv` (allowed difference 0.001) and writes
   `results/predictions/preds_<name>.csv`. If the numbers differ, it stops and puts the CSV files in
   `results/predictions/_unverified/` instead.

4. **DistilBERT + LoRA adapter.** Nothing to copy. The demo uses the first adapter folder it finds:
   1. the folder in `$env:DL_ADAPTER_DIR`, if this variable is set;
   2. `models/dl/distilbert_lora/`, if it exists (a local copy);
   3. `dl/checkpoints/distilbert_lora/`, the adapter committed by the DL teammate. It is what `dl/train_lora.py`
      saves with `model.save_pretrained(...)`: `adapter_config.json` and `adapter_model.safetensors` (the LoRA
      weights plus the `pre_classifier`/`classifier` head).

5. **First run needs internet.** The DL model and the length warning use `distilbert-base-uncased` from the Hugging
   Face Hub. The first start downloads it into `models/hf_cache/` (about 260 MB on disk); later starts use the cache.

## Run it

```powershell
..\.venv\Scripts\Activate.ps1
streamlit run demo/app.py --server.address localhost
```

The app opens at http://localhost:8501 (add `--server.port 8511` if that port is busy). With
`--server.address localhost` the app is reachable only from this computer, not from other machines on the network.
The sidebar should show all five models as "loaded"; a missing model shows "not available" and the reason.

## Check the predictors

```powershell
python demo/check_predictors.py --n 50
```

This compares the demo predictors with the saved test predictions on 50 random test reviews. It needs local files
that are not in the repository: `imdb_outputs/` (step 2) and `results/predictions/preds_<model>.csv` for every
loaded model. Step 3 writes the four ML files. Create the DL file from the committed
`results/predictions/distilbert_lora.csv` (needs `imdb_outputs/` from step 2):

```powershell
python eval/convert_dl_preds.py
```

It writes `results/predictions/preds_distilbert_lora.csv`. If you skip this step, the check stops with
`FileNotFoundError` when it reaches the DL model.

## Files

- `preprocess.py`: copy of the notebook's preprocessing (`text_clean`, `tokens_classical`, `text_classical`).
- `predictors.py`: one `predict(text) -> (label, prob)` per model, plus `n_wordpieces(text)`.
- `check_predictors.py`: compares the predictors with the saved prediction files.
- `app.py`: the Streamlit app. It reads `results/eval/metrics_comparison.csv` (sidebar scores) and
  `results/eval/hard_cases.csv` (the hard example).

## Known limits

- If no adapter is found in the three places of step 4, the DL model shows "not available".
- SVM scores are the sigmoid of the margin, not calibrated probabilities. Compare confidence within one model only.
- Reviews longer than 512 DistilBERT wordpieces are cut: the DL model sees only the first 128 and the last 382 wordpieces.
- The ML models ignore words they did not see in training. If a Word2Vec model knows none of the words, it uses a vector of zeros, so its score is not meaningful.
- `preprocess.py` is a copy of the notebook. If the notebook changes, update the copy too, or replace the import in `predictors.py` with Person 1's module.
- The hard example is a random row of `results/eval/hard_cases.csv`. To get another one, choose another option and then the hard example again.
