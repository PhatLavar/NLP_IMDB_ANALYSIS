# Test-set metrics (25,000 reviews, positive class = 1)

Written by `eval/error_analysis.py` from `results/predictions/preds_*.csv`. ROC-AUC uses only the ranking of `prob`, so it is valid for the SVM scores too. Log loss, Brier score and calibration are NOT reported: for the two LinearSVC models `prob` = expit(decision_function) is an uncalibrated score, not a probability.

| model | file | accuracy | precision | recall | f1 | roc_auc | tn | fp | fn | tp | n_errors | prob_column |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | preds_tfidf_logreg.csv | 0.8804 | 0.8767 | 0.8853 | 0.8809 | 0.9513 | 10943 | 1557 | 1434 | 11066 | 2991 | probability (LogisticRegression.predict_proba) |
| TF-IDF + LinearSVC | preds_tfidf_svm.csv | 0.8868 | 0.8894 | 0.8836 | 0.8865 | 0.9551 | 11126 | 1374 | 1455 | 11045 | 2829 | uncalibrated score: expit(LinearSVC.decision_function) |
| Word2Vec + LogReg | preds_w2v_logreg.csv | 0.8563 | 0.8607 | 0.8502 | 0.8554 | 0.9328 | 10780 | 1720 | 1873 | 10627 | 3593 | probability (LogisticRegression.predict_proba) |
| Word2Vec + LinearSVC | preds_w2v_svm.csv | 0.8572 | 0.8621 | 0.8505 | 0.8562 | 0.9335 | 10799 | 1701 | 1869 | 10631 | 3570 | uncalibrated score: expit(LinearSVC.decision_function) |
| DistilBERT + LoRA | preds_distilbert_lora.csv | 0.9256 | 0.9132 | 0.9407 | 0.9267 | 0.9773 | 11382 | 1118 | 741 | 11759 | 1859 | softmax probability of the fine-tuned model |
