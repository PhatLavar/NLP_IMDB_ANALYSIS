# Demo screenshots

Screenshots of the Streamlit demo (`demo/app.py`) running on localhost.
They were captured on 2026-11-18 with headless Microsoft Edge driven by Playwright: viewport 1400x900, 100% zoom, light theme.
Each caption describes what is visible in the image. P(positive) values are listed in the model order
tfidf_logreg, tfidf_svm, w2v_logreg, w2v_svm, distilbert_lora.

| File | Input | What is visible |
|---|---|---|
| `demo_01_overview.png` | none | Sidebar "Models": all five models (tfidf_logreg, tfidf_svm, w2v_logreg, w2v_svm, distilbert_lora) show "loaded". "Test-set scores" table, accuracy / F1: distilbert_lora 0.9256 / 0.9267, tfidf_logreg 0.8804 / 0.8809, tfidf_svm 0.8868 / 0.8865, w2v_logreg 0.8563 / 0.8554, w2v_svm 0.8572 / 0.8562. Empty review box and the Predict button. |
| `demo_02_clear_positive.png` | "An absolute delight from start to finish. Superb acting and a moving story." | All five models predict Positive. P(positive): 0.8611, 0.7769, 0.9980, 0.9145, 0.9993. |
| `demo_03_negation.png` | "It's not too bad." | The four ML models predict Negative (P(positive) 0.0013, 0.0831, 0.0000, 0.0046). distilbert_lora predicts Positive (0.9347). |
| `demo_04_sarcasm.png` | "hilarious antics ensue (I wish)" | All five models predict Positive: 0.8037, 0.7046, 0.9996, 0.9598, 0.9971. No model reads "(I wish)" as sarcasm. |
| `demo_05_mixed.png` | "It is really stupid and lame for the first half, then it actually starts to get good for the last half." | The four ML models predict Negative (0.1227, 0.2156, 0.0000, 0.0048). distilbert_lora predicts Positive (0.7638). |
| `demo_06_truncation_warning.png` | Full text of test review `test_14218` (from `results/eval/annotation/annotations_final.csv`) | Yellow warning: "This review has 1083 wordpieces, more than the 512 DistilBERT can read. The DL model only sees the beginning and the end of the review." Predictions: tfidf_logreg Negative (0.3337), tfidf_svm Negative (0.3971), w2v_logreg Positive (0.5848), w2v_svm Positive (0.5503), distilbert_lora Negative (0.0007). The review box shows the end of the text. |
| `demo_07_probability_chart.png` | Same as `demo_06` (`test_14218`) | Close-up of the results table and the P(positive) bar chart, with the dashed 0.5 threshold and the note that SVM scores are not calibrated probabilities. This input was chosen because it splits the models most evenly (3 Negative, 2 Positive); the other inputs above are 5-0 or 4-1. |

Note on the truncation warning in `demo_06`: the "512" includes the two special tokens [CLS] and [SEP], so the DL
model reads at most 510 wordpieces of review text. The "1083 wordpieces" count also includes [CLS] and [SEP].
