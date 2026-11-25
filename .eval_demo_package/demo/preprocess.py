"""Preprocessing for the demo, copied from Person 1's notebook imdb_data_preprocessing.ipynb.

preprocess(text) -> (text_clean, tokens_classical, text_classical)
    text_clean        minimal cleanup (notebook cell "clean_review"): input of the DL model
    tokens_classical  spaCy tokens after filtering and lemmatization: input of the Word2Vec models
    text_classical    " ".join(tokens_classical): input of the TF-IDF models

The ML models were trained on these columns, so this file must match the notebook exactly.
It is self-contained: if Person 1 ships an official module later, only the import in
demo/predictors.py has to change.

Versions used by the notebook (imdb_outputs/metadata.json): spacy 3.8.16, en_core_web_sm 3.8.0.
"""
import html
import re
from functools import lru_cache

import spacy

# ---- Minimal cleanup (notebook: clean_review) ----
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_review(text):
    """Remove common HTML formatting and normalize whitespace."""
    # Remove actual tags before decoding entities, so escaped
    # literal text such as &lt;movie&gt; is not treated as a tag.
    without_tags = HTML_TAG_PATTERN.sub(" ", text)
    decoded = html.unescape(without_tags)
    return WHITESPACE_PATTERN.sub(" ", decoded).strip()


# ---- Classical pipeline (notebook: PROTECTED_WORDS, token_filter_reason, classical_token_form) ----
# Explicit exceptions to stop-word removal.
PROTECTED_WORDS = {
    "not", "no", "never", "nor", "neither",
    "n't", "n’t", "n‘t",
    "but", "however", "although", "though", "yet",
}
NEGATION_CONTRACTIONS = {"n't", "n’t", "n‘t"}


def token_filter_reason(token):
    """Explain why a token is kept or removed."""
    if token.is_space:
        return "remove: whitespace"
    if token.is_punct:
        return "remove: punctuation"
    if token.text.lower() in PROTECTED_WORDS:
        return "keep: protected word"
    if token.is_stop:
        return "remove: stop word"
    return "keep: content token"


def keep_classical_token(token):
    """Select tokens for the classical preprocessing pipeline."""
    return token_filter_reason(token).startswith("keep:")


def classical_token_form(token):
    """Return the normalized form of a retained spaCy token."""
    surface = token.text.lower()
    # Give straight/curly contracted negation one consistent form.
    if surface in NEGATION_CONTRACTIONS:
        return "not"
    # Preserve our explicit sentiment-related exceptions.
    if surface in PROTECTED_WORDS:
        return surface
    lemma = token.lemma_.strip().lower()
    # Fall back to the surface form if a lemma is empty.
    return lemma if lemma else surface


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy once. Parser and NER are not needed (same as the notebook)."""
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text):
    """Raw review -> (text_clean, tokens_classical, text_classical)."""
    text_clean = clean_review(text)
    doc = get_nlp()(text_clean)
    tokens = [classical_token_form(token) for token in doc if keep_classical_token(token)]
    return text_clean, tokens, " ".join(tokens)
