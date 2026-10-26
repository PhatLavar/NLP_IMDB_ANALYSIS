import os
import time
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_json(
    "../imdb_outputs/imdb_train_prepared.jsonl",
    lines=True
)

val = pd.read_json(
    "../imdb_outputs/imdb_val_prepared.jsonl",
    lines=True
)

test = pd.read_json(
    "../imdb_outputs/imdb_test_prepared.jsonl",
    lines=True
)

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))


y_train = train["label"]
y_val = val["label"]
y_test = test["label"]


# ============================================================
# 2. RESULTS STORAGE
# ============================================================

results = []


def evaluate_model(name, model, X_train, X_val, X_test):
    """Train and evaluate one classifier."""

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    results.append({
        "representation": name.split(" + ")[0],
        "classifier": name.split(" + ")[1],
        "validation_accuracy": accuracy_score(
            y_val, val_predictions
        ),
        "validation_precision": precision_score(
            y_val, val_predictions
        ),
        "validation_recall": recall_score(
            y_val, val_predictions
        ),
        "validation_f1": f1_score(
            y_val, val_predictions
        ),
        "test_accuracy": accuracy_score(
            y_test, test_predictions
        ),
        "test_precision": precision_score(
            y_test, test_predictions
        ),
        "test_recall": recall_score(
            y_test, test_predictions
        ),
        "test_f1": f1_score(
            y_test, test_predictions
        ),
        "training_time_seconds": training_time
    })

    print(f"\n===== {name} =====")
    print(
        f"Validation Accuracy: "
        f"{accuracy_score(y_val, val_predictions):.4f}"
    )
    print(
        f"Validation F1: "
        f"{f1_score(y_val, val_predictions):.4f}"
    )
    print(
        f"Test Accuracy: "
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )
    print(
        f"Test F1: "
        f"{f1_score(y_test, test_predictions):.4f}"
    )
    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# 3. TF-IDF
# ============================================================

print("\n==============================")
print("BUILDING TF-IDF")
print("==============================")

tfidf_start = time.time()

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True)

X_train_tfidf = tfidf.fit_transform(
    train["text_classical"]
)

X_val_tfidf = tfidf.transform(
    val["text_classical"]
)

X_test_tfidf = tfidf.transform(
    test["text_classical"]
)

tfidf_time = time.time() - tfidf_start

print("TF-IDF vocabulary size:", len(tfidf.vocabulary_))
print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF validation shape:", X_val_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)
print(f"TF-IDF creation time: {tfidf_time:.2f} seconds")


# ============================================================
# 4. TF-IDF + LOGISTIC REGRESSION
# ============================================================

evaluate_model(
    "TF-IDF + Logistic Regression",
    LogisticRegression(max_iter=1000),
    X_train_tfidf,
    X_val_tfidf,
    X_test_tfidf
)


# ============================================================
# 5. TF-IDF + SVM
# ============================================================

evaluate_model(
    "TF-IDF + SVM",
    LinearSVC(),
    X_train_tfidf,
    X_val_tfidf,
    X_test_tfidf
)


# ============================================================
# 6. WORD2VEC
# ============================================================

print("\n==============================")
print("BUILDING WORD2VEC")
print("==============================")

train_tokens = train["tokens_classical"].tolist()
val_tokens = val["tokens_classical"].tolist()
test_tokens = test["tokens_classical"].tolist()

w2v_start = time.time()

w2v = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=1,
    sg=1,
    epochs=10,
    seed=42
)

w2v_time = time.time() - w2v_start

print("Word2Vec vocabulary size:", len(w2v.wv))
print(f"Word2Vec training time: {w2v_time:.2f} seconds")


# ============================================================
# 7. DOCUMENT VECTORS
# ============================================================

def document_vector(tokens, model):

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


vector_start = time.time()

X_train_w2v = np.array([
    document_vector(tokens, w2v)
    for tokens in train_tokens
])

X_val_w2v = np.array([
    document_vector(tokens, w2v)
    for tokens in val_tokens
])

X_test_w2v = np.array([
    document_vector(tokens, w2v)
    for tokens in test_tokens
])

vector_time = time.time() - vector_start

print("Word2Vec train shape:", X_train_w2v.shape)
print("Word2Vec validation shape:", X_val_w2v.shape)
print("Word2Vec test shape:", X_test_w2v.shape)
print(f"Document vector creation time: {vector_time:.2f} seconds")


# ============================================================
# 8. WORD2VEC + LOGISTIC REGRESSION
# ============================================================

evaluate_model(
    "Word2Vec + Logistic Regression",
    LogisticRegression(max_iter=1000),
    X_train_w2v,
    X_val_w2v,
    X_test_w2v
)


# ============================================================
# 9. WORD2VEC + SVM
# ============================================================

evaluate_model(
    "Word2Vec + SVM",
    LinearSVC(),
    X_train_w2v,
    X_val_w2v,
    X_test_w2v
)


# ============================================================
# 10. SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {
    "tfidf_ngram_range": "(1, 2)",
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.95,
    "tfidf_sublinear_tf": True,

    "word2vec_vector_size": 100,
    "word2vec_window": 5,
    "word2vec_min_count": 2,
    "word2vec_sg": 1,
    "word2vec_epochs": 10,
    "word2vec_workers": 1,
    "word2vec_seed": 42,

    "logistic_regression_max_iter": 1000,
    "svm": "LinearSVC"
}

config_df = pd.DataFrame(
    [config]
)

config_df.to_csv(
    "../results/ml_config.csv",
    index=False
)