"""Streamlit demo: type a movie review and see what each model predicts.

Run from the repository root:
    streamlit run demo/app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from predictors import get_predictors, n_wordpieces

ROOT = Path(__file__).resolve().parents[1]
LABELS = {0: "Negative", 1: "Positive"}
NO_EXAMPLE = "(type your own review)"
HARD_EXAMPLE = "Load a hard test example (wrong for all 5 models)"
EXAMPLES = {
    "Positive example": "One of the best films I have seen this year. The acting is superb, the story is moving "
                        "and the soundtrack is beautiful. I would happily watch it again.",
    "Negative example": "A complete waste of two hours. The plot makes no sense, the dialogue is painful and the "
                        "actors look bored the whole time. Avoid this one.",
    "Tricky example": "Not bad at all, but I would never watch it again.",
}


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return get_predictors()


@st.cache_data
def load_metrics():
    df = pd.read_csv(ROOT / "results/eval/metrics_comparison.csv", encoding="utf-8")
    return df[["model", "accuracy", "f1"]]


@st.cache_data
def load_hard_cases():
    return pd.read_csv(ROOT / "results/eval/hard_cases.csv", encoding="utf-8-sig")


def on_example_change():
    """Put the chosen example into the text area and forget old results."""
    choice = st.session_state["example"]
    st.session_state["hard_case"] = None
    st.session_state["reveal"] = False
    st.session_state["results"] = None
    if choice == HARD_EXAMPLE:
        row = load_hard_cases().sample(n=1).iloc[0]  # a new random row every time
        st.session_state["hard_case"] = {"id": row["id"], "label": int(row["label"])}
        st.session_state["review"] = row["text"]
    elif choice in EXAMPLES:
        st.session_state["review"] = EXAMPLES[choice]


def predict_all(predictors, text):
    """One table row per model; unavailable models get 'not available' and the reason."""
    rows = []
    for name, (predict, status) in predictors.items():
        if predict is None:
            rows.append({"model": name, "label": "not available", "confidence": "", "P(positive)": None,
                         "note": status})
            continue
        label, prob = predict(text)
        rows.append({"model": name, "label": LABELS[label], "confidence": f"{max(prob, 1 - prob):.2%}",
                     "P(positive)": prob, "note": ""})
    return pd.DataFrame(rows)


st.set_page_config(page_title="IMDB sentiment demo")
st.title("IMDB sentiment demo")
st.write("Type a movie review and compare the predictions of 4 classical ML models and DistilBERT + LoRA.")
for key in ["hard_case", "results"]:
    st.session_state.setdefault(key, None)
st.session_state.setdefault("reveal", False)

predictors = load_models()

# ---- Sidebar: model status and test-set scores
st.sidebar.header("Models")
for name, (predict, status) in predictors.items():
    if predict is None:
        st.sidebar.error(f"**{name}**: not available. {status}")
    else:
        st.sidebar.success(f"**{name}**: loaded")
st.sidebar.header("Test-set scores")
st.sidebar.dataframe(load_metrics(), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["accuracy", "f1"]})
st.sidebar.caption("From results/eval/metrics_comparison.csv (25,000 test reviews).")

# ---- Input
st.selectbox("Load an example", [NO_EXAMPLE, *EXAMPLES, HARD_EXAMPLE], key="example",
             on_change=on_example_change)
text = st.text_area("Movie review", key="review", height=200)

hard_case = st.session_state["hard_case"]
if hard_case:
    st.caption(f"Hard test example {hard_case['id']}: every model got it wrong.")
    if st.button("Reveal true label"):
        st.session_state["reveal"] = True
    if st.session_state["reveal"]:
        st.info(f"True label: **{LABELS[hard_case['label']]}**")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.session_state["results"] = None
        st.info("Please type a review or load an example first.")
    else:
        try:
            pieces = n_wordpieces(text)
        except Exception as e:  # tokenizer not downloaded and no internet
            pieces = None
            st.caption(f"Could not count DistilBERT wordpieces: {e}")
        with st.spinner("Predicting..."):
            st.session_state["results"] = (predict_all(predictors, text), pieces)

# ---- Results (kept in session_state so they survive the "Reveal true label" click)
if st.session_state["results"] is not None:
    results, pieces = st.session_state["results"]
    if pieces is not None and pieces > 512:
        st.warning(f"This review has {pieces} wordpieces, more than the 512 DistilBERT can read. "
                   "The DL model only sees the beginning and the end of the review.")
    st.dataframe(results, hide_index=True, placeholder="",  # placeholder: show missing values as empty cells
                 column_config={"P(positive)": st.column_config.NumberColumn(format="%.4f")})
    st.caption("SVM scores are the sigmoid of the margin and are NOT calibrated probabilities, "
               "so compare confidence within a model only.")

    available = results.dropna(subset=["P(positive)"])
    if len(available):
        bars = alt.Chart(available).mark_bar().encode(
            x=alt.X("P(positive):Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort=None, title=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=["Negative", "Positive"],
                                                       range=["#d62728", "#2ca02c"])),
        )
        # same field name as the bars, so the shared x-axis title stays "P(positive)"
        threshold = alt.Chart(pd.DataFrame({"P(positive)": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(
            x="P(positive):Q")
        st.altair_chart(bars + threshold, width="stretch")
