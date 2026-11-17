"""Regenerate per-review TEST predictions for the 4 ML models of ml/model.py.

Uses exactly the settings of ml/model.py (see also results/ml_config.csv), fits on
TRAIN only and predicts the full TEST set. ml/model.py itself is not changed.

Writes:
    results/predictions/preds_<name>.csv   columns id, text, label, pred, prob
    models/ml/<name>.joblib                fitted features + classifier
for <name> in tfidf_logreg, tfidf_svm, w2v_logreg, w2v_svm.

prob = probability of the POSITIVE class:
    LogisticRegression -> predict_proba(X)[:, 1]
    LinearSVC          -> sigmoid(decision_function(X))  (LinearSVC has no predict_proba)

Safety checks before anything is written:
    1. pred must equal (prob > 0.5) on every row, otherwise the script stops.
    2. My test accuracy / F1 must be within 0.001 of results/ml_results.csv.
       If not, the files go to results/predictions/_unverified/ and the script stops.

Usage (from the repository root, PowerShell):
    $env:PYTHONHASHSEED = "42"
    python eval/make_ml_preds.py
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.001  # max allowed difference from the teammate's numbers

# model name -> (representation, classifier) as written in results/ml_results.csv
MODELS = {
    "tfidf_logreg": ("TF-IDF", "Logistic Regression"),
    "tfidf_svm": ("TF-IDF", "SVM"),
    "w2v_logreg": ("Word2Vec", "Logistic Regression"),
    "w2v_svm": ("Word2Vec", "SVM"),
}


def document_vector(tokens, w2v):
    """Mean of the known word vectors, zeros if none (copied from ml/model.py)."""
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if not vectors:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)


def positive_prob(clf, X):
    """Probability of the positive class (label 1)."""
    if isinstance(clf, LogisticRegression):
        return clf.predict_proba(X)[:, 1]
    return expit(clf.decision_function(X))  # LinearSVC: sigmoid of the signed distance


def main():
    parser = argparse.ArgumentParser(description="Regenerate test predictions for the 4 ML models.")
    parser.add_argument("--data", type=Path, default=ROOT / "imdb_outputs", help="folder with the prepared jsonl files")
    parser.add_argument("--out", type=Path, default=ROOT / "results/predictions", help="where to write preds_*.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/ml", help="where to save the fitted models")
    args = parser.parse_args()

    hash_seed = os.environ.get("PYTHONHASHSEED")
    print(f"PYTHONHASHSEED = {hash_seed}")
    if hash_seed != "42":
        print('WARNING: PYTHONHASHSEED is not "42". In PowerShell run: $env:PYTHONHASHSEED = "42"')

    train = pd.read_json(args.data / "imdb_train_prepared.jsonl", lines=True, encoding="utf-8")
    test = pd.read_json(args.data / "imdb_test_prepared.jsonl", lines=True, encoding="utf-8")
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    # TF-IDF, fitted on train only
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(train["text_classical"])
    X_test_tfidf = tfidf.transform(test["text_classical"])
    print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_):,}")

    # Word2Vec, trained on train tokens only (workers=1 + seed=42 for a repeatable result)
    train_tokens = train["tokens_classical"].tolist()
    w2v = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2,
                   workers=1, sg=1, epochs=10, seed=42)
    X_train_w2v = np.array([document_vector(tokens, w2v) for tokens in train_tokens])
    X_test_w2v = np.array([document_vector(tokens, w2v) for tokens in test["tokens_classical"]])
    print(f"Word2Vec vocabulary size: {len(w2v.wv):,}")

    # name -> (classifier, train features, test features, feature model to save with it)
    setups = {
        "tfidf_logreg": (LogisticRegression(max_iter=1000), X_train_tfidf, X_test_tfidf, {"vectorizer": tfidf}),
        "tfidf_svm": (LinearSVC(), X_train_tfidf, X_test_tfidf, {"vectorizer": tfidf}),
        "w2v_logreg": (LogisticRegression(max_iter=1000), X_train_w2v, X_test_w2v, {"word2vec": w2v}),
        "w2v_svm": (LinearSVC(), X_train_w2v, X_test_w2v, {"word2vec": w2v}),
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for name, (clf, X_train, X_test, features) in setups.items():
        clf.fit(X_train, train["label"])
        assert list(clf.classes_) == [0, 1], f"{name}: unexpected classes {clf.classes_}"
        preds[name] = pd.DataFrame({
            "id": test["review_id"],
            "text": test["text"],  # original review text
            "label": test["label"],
            "pred": clf.predict(X_test),
            "prob": positive_prob(clf, X_test),
        })
        joblib.dump({**features, "classifier": clf, "pythonhashseed": hash_seed},
                    args.models_dir / f"{name}.joblib")
        print(f"Trained {name}, saved {args.models_dir / f'{name}.joblib'}")

    # Check 1: pred must agree with prob > 0.5 (can only fail by float rounding near 0.5)
    has_mismatch = False
    for name, df in preds.items():
        mismatch = df[df["pred"] != (df["prob"] > 0.5).astype(int)]
        print(f"{name}: {len(mismatch)} rows where pred != (prob > 0.5)")
        if len(mismatch):
            has_mismatch = True
            print(mismatch[["id", "label", "pred", "prob"]].to_string(index=False))
    if has_mismatch:
        sys.exit("STOP: pred/prob mismatches found (rows above). No prediction files were written.")

    # Check 2: my test accuracy / F1 vs. the teammate's results/ml_results.csv
    theirs = pd.read_csv(ROOT / "results/ml_results.csv", encoding="utf-8")
    rows = []
    for name, (rep, clf_name) in MODELS.items():
        row = theirs[(theirs["representation"] == rep) & (theirs["classifier"] == clf_name)]
        assert len(row) == 1, f"expected one row for {rep} + {clf_name} in ml_results.csv, found {len(row)}"
        y, p = preds[name]["label"], preds[name]["pred"]
        for metric, mine in [("test_accuracy", accuracy_score(y, p)), ("test_f1", f1_score(y, p))]:
            rows.append({"model": name, "metric": metric, "mine": mine,
                         "theirs": row[metric].iloc[0], "diff": mine - row[metric].iloc[0]})
    check = pd.DataFrame(rows)
    check["status"] = np.where(check["diff"].abs() <= TOLERANCE, "ok", "DIFFERENT")
    print(f"\nCheck against results/ml_results.csv (allowed difference {TOLERANCE}):")
    print(check.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    verified = (check["status"] == "ok").all()
    out_dir = args.out if verified else args.out / "_unverified"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in preds.items():
        # utf-8-sig adds a BOM so Excel on Windows shows non-ASCII characters correctly
        df.to_csv(out_dir / f"preds_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(preds)} prediction files to {out_dir}")
    if not verified:
        sys.exit("STOP: some models differ from results/ml_results.csv by more than "
                 f"{TOLERANCE}. Files are in {out_dir}. Do not run evaluate.py on them.")


if __name__ == "__main__":
    main()
