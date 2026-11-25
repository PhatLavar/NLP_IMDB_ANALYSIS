"""One predict(text) -> (label, prob) function per model, for the demo app and check_predictors.py.

    prob  = P(positive), label = int(prob > 0.5)   (0 = negative, 1 = positive)
    text  = a RAW review (HTML such as <br /> is allowed)

ML models: models/ml/<name>.joblib, written by eval/make_ml_preds.py. Prediction is done exactly
like in that script (TF-IDF or Word2Vec mean vector; LogisticRegression.predict_proba,
LinearSVC -> sigmoid of decision_function).

DL model: distilbert-base-uncased + the LoRA adapter from the environment variable DL_ADAPTER_DIR if set,
else models/dl/distilbert_lora/ if it exists, else dl/checkpoints/distilbert_lora/ (in the repository).
Inference copies dl/train_lora.py.
"""
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from preprocess import clean_review, preprocess  # swap this import for Person 1's module if needed

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "models/ml"
ML_MODELS = ["tfidf_logreg", "tfidf_svm", "w2v_logreg", "w2v_svm"]
HF_CACHE = ROOT / "models/hf_cache"  # same cache as eval/tag_errors.py (models/ is gitignored)

# DL settings of the final run (results/dl_config.csv)
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 512
TRUNCATION = "head_tail"
HEAD_FRAC = 0.25


# ---------------------------------------------------------------- ML models
def document_vector(tokens, w2v):
    """Mean of the known word vectors, zeros if none (copied from eval/make_ml_preds.py)."""
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if not vectors:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)


def positive_prob(clf, X):
    """Probability of the positive class (copied from eval/make_ml_preds.py)."""
    if isinstance(clf, LogisticRegression):
        return clf.predict_proba(X)[:, 1]
    return expit(clf.decision_function(X))  # LinearSVC: sigmoid of the signed distance


@lru_cache(maxsize=None)
def load_ml(name):
    """Load models/ml/<name>.joblib once: {'vectorizer' or 'word2vec', 'classifier', ...}."""
    return joblib.load(ML_DIR / f"{name}.joblib")


def make_ml_predictor(name):
    bundle = load_ml(name)

    def predict(text):
        _, tokens, text_classical = preprocess(text)
        if "vectorizer" in bundle:
            X = bundle["vectorizer"].transform([text_classical])
        else:
            X = np.array([document_vector(tokens, bundle["word2vec"])])
        prob = float(positive_prob(bundle["classifier"], X)[0])
        return int(prob > 0.5), prob

    return predict


# ---------------------------------------------------------------- DL model
def adapter_dir():
    """DL_ADAPTER_DIR if set, else models/dl/distilbert_lora/ if present, else the adapter committed in dl/checkpoints/."""
    if "DL_ADAPTER_DIR" in os.environ:
        return Path(os.environ["DL_ADAPTER_DIR"])
    local = ROOT / "models/dl/distilbert_lora"
    return local if local.is_dir() else ROOT / "dl/checkpoints/distilbert_lora"


@lru_cache(maxsize=1)
def get_tokenizer():
    """DistilBERT tokenizer (internet is needed the first time, then it is cached)."""
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()  # hide the expected "longer than 512" warning (dl/train_lora.py:428)
    return AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)


def n_wordpieces(text):
    """Wordpieces the DL model would get without truncation, special tokens included (> 512: truncated)."""
    return len(get_tokenizer()(clean_review(text), add_special_tokens=True, truncation=False)["input_ids"])


def build_inputs(tokenizer, ids, max_length, mode, head_frac):
    """Copied from dl/train_lora.py:193-205 (truncation + [CLS]/[SEP])."""
    budget = max_length - 2  # [CLS] and [SEP]
    if len(ids) <= budget:
        kept = ids
    elif mode == "head":
        kept = ids[:budget]
    elif mode == "head_tail":
        head = int(round(budget * head_frac))
        tail = budget - head
        kept = ids[:head] + ids[len(ids) - tail :]
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")
    return [tokenizer.cls_token_id] + kept + [tokenizer.sep_token_id]


@lru_cache(maxsize=1)
def load_dl(path):
    """Base model with a 2-class head + LoRA adapter, in eval mode (dl/train_lora.py:255-277, 589)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    from transformers import logging as hf_logging

    # hide the base model's "MISSING classifier" load report: PEFT restores the head (dl/train_lora.py:428)
    hf_logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1},
        cache_dir=HF_CACHE,
    )
    model = PeftModel.from_pretrained(base, path).to(device)
    model.eval()
    return model, device


def make_dl_predictor(path):
    import torch

    model, device = load_dl(path)
    tokenizer = get_tokenizer()

    def predict(text):
        # dl/train_lora.py:188 tokenises text_clean without special tokens and without truncation
        ids = tokenizer(clean_review(text), add_special_tokens=False, truncation=False)["input_ids"]
        seq = build_inputs(tokenizer, ids, MAX_LENGTH, TRUNCATION, HEAD_FRAC)
        # dl/train_lora.py:215-223 pads to the longest review of the batch; one review needs no padding
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        # dl/train_lora.py:241-251: no_grad, fp16 autocast on CUDA only
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        prob = float(torch.softmax(logits.float(), dim=-1)[0, 1])  # dl/train_lora.py:226-230
        return int(prob > 0.5), prob

    return predict


# ---------------------------------------------------------------- all models
def get_predictors():
    """Ordered dict name -> (predict function or None, status message). Never raises."""
    predictors = OrderedDict()
    for name in ML_MODELS:
        path = ML_DIR / f"{name}.joblib"
        if not path.is_file():
            predictors[name] = (None, f"{path} not found: run python eval/make_ml_preds.py")
            continue
        try:
            predictors[name] = (make_ml_predictor(name), f"loaded from {path}")
        except Exception as e:  # e.g. a joblib file written with other library versions
            predictors[name] = (None, f"could not load {path}: {type(e).__name__}: {e}")

    path = adapter_dir()
    if not path.is_dir():
        predictors["distilbert_lora"] = (None, f"DL adapter not found at {path}: set DL_ADAPTER_DIR to the "
                                               "adapter folder or restore dl/checkpoints/distilbert_lora/")
    else:
        try:
            predictors["distilbert_lora"] = (make_dl_predictor(str(path)), f"loaded adapter from {path}")
        except Exception as e:  # missing torch/peft, broken adapter, no internet for the base model, ...
            predictors["distilbert_lora"] = (None, f"DL adapter at {path} could not be loaded: "
                                                   f"{type(e).__name__}: {e}")
    return predictors
