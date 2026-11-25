# Error-analysis tables

Written by `eval/error_analysis.py`. Every number is recomputed from `results/predictions/preds_*.csv` and the annotation files.

## Step 1b: metrics from preds vs teammates' results files

File: `results/eval/error_analysis/preflight_metric_crosscheck.csv`. Flag if |diff| > 0.0005.

| model | metric | from_preds | in_results_file | diff | flag |
|---|---|---|---|---|---|
| TF-IDF + LogReg | test_accuracy | 0.8804 | 0.8804 | 0.0000 | ok |
| TF-IDF + LogReg | test_precision | 0.8767 | 0.8767 | 0.0000 | ok |
| TF-IDF + LogReg | test_recall | 0.8853 | 0.8853 | 0.0000 | ok |
| TF-IDF + LogReg | test_f1 | 0.8809 | 0.8809 | 0.0000 | ok |
| TF-IDF + LinearSVC | test_accuracy | 0.8868 | 0.8868 | 0.0000 | ok |
| TF-IDF + LinearSVC | test_precision | 0.8894 | 0.8894 | 0.0000 | ok |
| TF-IDF + LinearSVC | test_recall | 0.8836 | 0.8836 | 0.0000 | ok |
| TF-IDF + LinearSVC | test_f1 | 0.8865 | 0.8865 | 0.0000 | ok |
| Word2Vec + LogReg | test_accuracy | 0.8563 | 0.8563 | 0.0000 | ok |
| Word2Vec + LogReg | test_precision | 0.8607 | 0.8607 | 0.0000 | ok |
| Word2Vec + LogReg | test_recall | 0.8502 | 0.8502 | 0.0000 | ok |
| Word2Vec + LogReg | test_f1 | 0.8554 | 0.8554 | 0.0000 | ok |
| Word2Vec + LinearSVC | test_accuracy | 0.8572 | 0.8572 | 0.0000 | ok |
| Word2Vec + LinearSVC | test_precision | 0.8621 | 0.8621 | 0.0000 | ok |
| Word2Vec + LinearSVC | test_recall | 0.8505 | 0.8505 | 0.0000 | ok |
| Word2Vec + LinearSVC | test_f1 | 0.8562 | 0.8562 | 0.0000 | ok |
| DistilBERT + LoRA | test_accuracy | 0.9256 | 0.9256 | 0.0000 | ok |
| DistilBERT + LoRA | test_precision | 0.9132 | 0.9132 | 0.0000 | ok |
| DistilBERT + LoRA | test_recall | 0.9407 | 0.9407 | 0.0000 | ok |
| DistilBERT + LoRA | test_f1 | 0.9267 | 0.9267 | 0.0000 | ok |

## Step 1c: annotation files

File: `results/eval/error_analysis/preflight_annotation_completeness.csv`. 

| file | rows | empty_category | invalid_category | category_with_whitespace | unsure_yes | problems |
|---|---|---|---|---|---|---|
| annotator_A_kiet.csv | 50 | 0 | 0 | 0 | 8 | none |
| annotator_B_Phat.csv | 50 | 0 | 0 | 0 | 1 | none |

## Sampling design (recomputed from preds)

File: `results/eval/error_analysis/sample_design.csv`. Frame = DL errors with duplicated texts removed (keep first): 1,840 of 1,859 DL errors. All 100 sampled reviews are DL errors.

| stratum | population_N | sampled_n | sampling_fraction | weight_N_over_n |
|---|---|---|---|---|
| all5 (wrong for all 5 models) | 702 | 40 | 0.0570 | 17.5500 |
| other (DL wrong, not all 5 wrong) | 1138 | 60 | 0.0527 | 18.9667 |

## Step 4a: category counts in the sample (SAMPLE COUNTS)

File: `results/eval/error_analysis/category_counts.csv`. label_kind main = first value; any = every value (multi-label, shares can add up to more than 1).

| label_kind | slice | n_reviews | category | n | share |
|---|---|---|---|---|---|
| main | all | 100 | sarcasm | 9 | 0.0900 |
| main | all | 100 | negation | 5 | 0.0500 |
| main | all | 100 | mixed | 47 | 0.4700 |
| main | all | 100 | long_truncated | 1 | 0.0100 |
| main | all | 100 | label_noise | 17 | 0.1700 |
| main | all | 100 | other | 21 | 0.2100 |
| main | group=all5 | 40 | sarcasm | 1 | 0.0250 |
| main | group=all5 | 40 | negation | 3 | 0.0750 |
| main | group=all5 | 40 | mixed | 17 | 0.4250 |
| main | group=all5 | 40 | long_truncated | 0 | 0.0000 |
| main | group=all5 | 40 | label_noise | 10 | 0.2500 |
| main | group=all5 | 40 | other | 9 | 0.2250 |
| main | group=other | 60 | sarcasm | 8 | 0.1333 |
| main | group=other | 60 | negation | 2 | 0.0333 |
| main | group=other | 60 | mixed | 30 | 0.5000 |
| main | group=other | 60 | long_truncated | 1 | 0.0167 |
| main | group=other | 60 | label_noise | 7 | 0.1167 |
| main | group=other | 60 | other | 12 | 0.2000 |
| main | file=A | 50 | sarcasm | 5 | 0.1000 |
| main | file=A | 50 | negation | 2 | 0.0400 |
| main | file=A | 50 | mixed | 28 | 0.5600 |
| main | file=A | 50 | long_truncated | 0 | 0.0000 |
| main | file=A | 50 | label_noise | 9 | 0.1800 |
| main | file=A | 50 | other | 6 | 0.1200 |
| main | file=B | 50 | sarcasm | 4 | 0.0800 |
| main | file=B | 50 | negation | 3 | 0.0600 |
| main | file=B | 50 | mixed | 19 | 0.3800 |
| main | file=B | 50 | long_truncated | 1 | 0.0200 |
| main | file=B | 50 | label_noise | 8 | 0.1600 |
| main | file=B | 50 | other | 15 | 0.3000 |
| any | all | 100 | sarcasm | 18 | 0.1800 |
| any | all | 100 | negation | 24 | 0.2400 |
| any | all | 100 | mixed | 72 | 0.7200 |
| any | all | 100 | long_truncated | 2 | 0.0200 |
| any | all | 100 | label_noise | 20 | 0.2000 |
| any | all | 100 | other | 28 | 0.2800 |
| any | group=all5 | 40 | sarcasm | 6 | 0.1500 |
| any | group=all5 | 40 | negation | 9 | 0.2250 |
| any | group=all5 | 40 | mixed | 29 | 0.7250 |
| any | group=all5 | 40 | long_truncated | 1 | 0.0250 |
| any | group=all5 | 40 | label_noise | 12 | 0.3000 |
| any | group=all5 | 40 | other | 11 | 0.2750 |
| any | group=other | 60 | sarcasm | 12 | 0.2000 |
| any | group=other | 60 | negation | 15 | 0.2500 |
| any | group=other | 60 | mixed | 43 | 0.7167 |
| any | group=other | 60 | long_truncated | 1 | 0.0167 |
| any | group=other | 60 | label_noise | 8 | 0.1333 |
| any | group=other | 60 | other | 17 | 0.2833 |
| any | file=A | 50 | sarcasm | 11 | 0.2200 |
| any | file=A | 50 | negation | 10 | 0.2000 |
| any | file=A | 50 | mixed | 38 | 0.7600 |
| any | file=A | 50 | long_truncated | 1 | 0.0200 |
| any | file=A | 50 | label_noise | 11 | 0.2200 |
| any | file=A | 50 | other | 10 | 0.2000 |
| any | file=B | 50 | sarcasm | 7 | 0.1400 |
| any | file=B | 50 | negation | 14 | 0.2800 |
| any | file=B | 50 | mixed | 34 | 0.6800 |
| any | file=B | 50 | long_truncated | 1 | 0.0200 |
| any | file=B | 50 | label_noise | 9 | 0.1800 |
| any | file=B | 50 | other | 18 | 0.3600 |

## Step 4a: estimated share among all DL errors (WEIGHTED)

File: `results/eval/error_analysis/category_weighted.csv`. Stratified estimate with weights N/n; 95% normal CI with finite-population correction; rough at n = 100.

| label_kind | category | unweighted_share | weighted_share | ci95_low | ci95_high |
|---|---|---|---|---|---|
| main | sarcasm | 0.0900 | 0.0920 | 0.0367 | 0.1473 |
| main | negation | 0.0500 | 0.0492 | 0.0080 | 0.0904 |
| main | mixed | 0.4700 | 0.4714 | 0.3755 | 0.5673 |
| main | long_truncated | 0.0100 | 0.0103 | 0.0000 | 0.0300 |
| main | label_noise | 0.1700 | 0.1675 | 0.0971 | 0.2380 |
| main | other | 0.2100 | 0.2095 | 0.1312 | 0.2878 |
| any | sarcasm | 0.1800 | 0.1809 | 0.1068 | 0.2551 |
| any | negation | 0.2400 | 0.2405 | 0.1581 | 0.3228 |
| any | mixed | 0.7200 | 0.7198 | 0.6333 | 0.8064 |
| any | long_truncated | 0.0200 | 0.0198 | 0.0000 | 0.0466 |
| any | label_noise | 0.2000 | 0.1969 | 0.1223 | 0.2715 |
| any | other | 0.2800 | 0.2802 | 0.1936 | 0.3667 |

## Step 4a: multi-label summary

File: `results/eval/error_analysis/multilabel_summary.csv`. 

| reviews | with_2plus_categories | mean_categories_per_review | unsure_yes |
|---|---|---|---|
| 100 | 62 | 1.6400 | 9 |

## Step 4b: main category among each model's sampled errors

File: `results/eval/error_analysis/per_model_categories.csv`. SAMPLE COUNTS, not rates: the sample only contains DL errors, so for an ML model it covers only the errors it shares with DL.

| model | errors_on_test | errors_shared_with_dl | share_of_errors_in_frame | sampled_reviews_wrong | main_sarcasm | main_negation | main_mixed | main_long_truncated | main_label_noise | main_other |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 2991 | 995 | 0.3327 | 55 | 3 | 4 | 27 | 1 | 11 | 9 |
| TF-IDF + LinearSVC | 2829 | 960 | 0.3393 | 52 | 2 | 4 | 23 | 1 | 12 | 10 |
| Word2Vec + LogReg | 3593 | 1012 | 0.2817 | 50 | 2 | 3 | 21 | 0 | 12 | 12 |
| Word2Vec + LinearSVC | 3570 | 1000 | 0.2801 | 49 | 3 | 3 | 20 | 0 | 11 | 12 |
| DistilBERT + LoRA | 1859 | 1859 | 1.0000 | 100 | 9 | 5 | 47 | 1 | 17 | 21 |

## Step 4c: DL wordpieces per labelled review

File: `results/eval/error_analysis/dl_wordpieces.csv`. Rule from results/dl_config.csv: budget 510 = head 128 + tail 382 (plus [CLS]/[SEP]).

| id | group | main_category | final_category | n_wordpieces | truncated | n_unseen | unseen_chars | long_truncated_label |
|---|---|---|---|---|---|---|---|---|
| test_00383 | other | negation | negation | 249 | 0 | 0 | 0 | 0 |
| test_00756 | other | sarcasm | sarcasm;mixed | 279 | 0 | 0 | 0 | 0 |
| test_00982 | other | other | other | 420 | 0 | 0 | 0 | 0 |
| test_01183 | other | sarcasm | sarcasm | 198 | 0 | 0 | 0 | 0 |
| test_01417 | other | sarcasm | sarcasm;other | 187 | 0 | 0 | 0 | 0 |
| test_01469 | other | mixed | mixed;negation | 214 | 0 | 0 | 0 | 0 |
| test_01620 | other | mixed | mixed | 341 | 0 | 0 | 0 | 0 |
| test_01694 | other | sarcasm | sarcasm;negation | 87 | 0 | 0 | 0 | 0 |
| test_01784 | other | sarcasm | sarcasm;mixed | 358 | 0 | 0 | 0 | 0 |
| test_01785 | all5 | label_noise | label_noise | 208 | 0 | 0 | 0 | 0 |
| test_01861 | other | mixed | mixed | 537 | 1 | 27 | 116 | 0 |
| test_02549 | all5 | mixed | mixed;sarcasm | 66 | 0 | 0 | 0 | 0 |
| test_02554 | other | sarcasm | sarcasm;mixed | 746 | 1 | 236 | 1007 | 0 |
| test_02576 | other | mixed | mixed | 285 | 0 | 0 | 0 | 0 |
| test_02686 | all5 | mixed | mixed | 146 | 0 | 0 | 0 | 0 |
| test_02840 | all5 | label_noise | label_noise | 322 | 0 | 0 | 0 | 0 |
| test_02857 | other | sarcasm | sarcasm;other | 146 | 0 | 0 | 0 | 0 |
| test_03113 | other | negation | negation;sarcasm | 149 | 0 | 0 | 0 | 0 |
| test_03287 | all5 | label_noise | label_noise;mixed | 248 | 0 | 0 | 0 | 0 |
| test_03671 | other | mixed | mixed | 299 | 0 | 0 | 0 | 0 |
| test_04261 | other | mixed | mixed;negation | 180 | 0 | 0 | 0 | 0 |
| test_04366 | other | label_noise | label_noise;mixed | 1097 | 1 | 587 | 2592 | 0 |
| test_04492 | all5 | mixed | mixed | 456 | 0 | 0 | 0 | 0 |
| test_04592 | other | mixed | mixed;negation | 345 | 0 | 0 | 0 | 0 |
| test_04635 | other | other | other;negation | 217 | 0 | 0 | 0 | 0 |
| test_04737 | other | mixed | mixed;sarcasm | 420 | 0 | 0 | 0 | 0 |
| test_04758 | other | mixed | mixed | 314 | 0 | 0 | 0 | 0 |
| test_04811 | all5 | label_noise | label_noise | 409 | 0 | 0 | 0 | 0 |
| test_04871 | all5 | mixed | mixed;negation | 226 | 0 | 0 | 0 | 0 |
| test_05009 | all5 | label_noise | label_noise;mixed | 152 | 0 | 0 | 0 | 0 |
| test_05282 | other | other | other;mixed | 335 | 0 | 0 | 0 | 0 |
| test_05541 | other | mixed | mixed;negation | 355 | 0 | 0 | 0 | 0 |
| test_06014 | all5 | mixed | mixed;label_noise | 221 | 0 | 0 | 0 | 0 |
| test_06593 | all5 | label_noise | label_noise | 176 | 0 | 0 | 0 | 0 |
| test_07158 | other | mixed | mixed;negation | 606 | 1 | 96 | 397 | 0 |
| test_07227 | other | label_noise | label_noise;negation | 174 | 0 | 0 | 0 | 0 |
| test_07555 | other | sarcasm | sarcasm | 479 | 0 | 0 | 0 | 0 |
| test_07899 | other | mixed | mixed | 221 | 0 | 0 | 0 | 0 |
| test_08341 | other | mixed | mixed | 235 | 0 | 0 | 0 | 0 |
| test_08392 | other | other | other;mixed | 192 | 0 | 0 | 0 | 0 |
| test_08975 | other | label_noise | label_noise;other | 246 | 0 | 0 | 0 | 0 |
| test_08983 | other | mixed | mixed | 360 | 0 | 0 | 0 | 0 |
| test_08990 | all5 | negation | negation;mixed | 238 | 0 | 0 | 0 | 0 |
| test_09076 | other | mixed | mixed | 114 | 0 | 0 | 0 | 0 |
| test_09254 | other | label_noise | label_noise;mixed | 227 | 0 | 0 | 0 | 0 |
| test_09368 | all5 | label_noise | label_noise | 198 | 0 | 0 | 0 | 0 |
| test_09943 | all5 | other | other;mixed | 646 | 1 | 136 | 592 | 0 |
| test_10257 | other | other | other | 164 | 0 | 0 | 0 | 0 |
| test_10284 | all5 | mixed | mixed;sarcasm | 186 | 0 | 0 | 0 | 0 |
| test_10315 | other | other | other;mixed | 1195 | 1 | 685 | 3214 | 0 |
| test_10491 | all5 | mixed | mixed;negation | 242 | 0 | 0 | 0 | 0 |
| test_10874 | other | mixed | mixed | 121 | 0 | 0 | 0 | 0 |
| test_11053 | all5 | mixed | mixed | 895 | 1 | 385 | 1781 | 0 |
| test_12353 | other | other | other | 592 | 1 | 82 | 400 | 0 |
| test_12498 | other | mixed | mixed | 564 | 1 | 54 | 247 | 0 |
| test_12846 | all5 | other | other | 140 | 0 | 0 | 0 | 0 |
| test_12981 | all5 | label_noise | label_noise | 197 | 0 | 0 | 0 | 0 |
| test_12989 | other | label_noise | label_noise | 314 | 0 | 0 | 0 | 0 |
| test_13076 | all5 | mixed | mixed | 207 | 0 | 0 | 0 | 0 |
| test_13403 | all5 | mixed | mixed;other;label_noise | 216 | 0 | 0 | 0 | 0 |
| test_13474 | all5 | other | other;mixed | 253 | 0 | 0 | 0 | 0 |
| test_13792 | other | mixed | mixed;negation | 106 | 0 | 0 | 0 | 0 |
| test_13929 | all5 | mixed | mixed;other | 771 | 1 | 261 | 1008 | 0 |
| test_14218 | other | long_truncated | long_truncated;mixed | 1081 | 1 | 571 | 2806 | 1 |
| test_14247 | all5 | negation | negation;mixed | 568 | 1 | 58 | 262 | 0 |
| test_14735 | other | mixed | mixed | 742 | 1 | 232 | 996 | 0 |
| test_14796 | other | other | other;mixed | 546 | 1 | 36 | 158 | 0 |
| test_15223 | all5 | label_noise | label_noise;mixed | 314 | 0 | 0 | 0 | 0 |
| test_15608 | other | other | other;mixed | 354 | 0 | 0 | 0 | 0 |
| test_15942 | all5 | other | other | 170 | 0 | 0 | 0 | 0 |
| test_16153 | other | label_noise | label_noise;mixed | 349 | 0 | 0 | 0 | 0 |
| test_16239 | all5 | other | other;negation | 309 | 0 | 0 | 0 | 0 |
| test_16368 | other | mixed | mixed;sarcasm | 414 | 0 | 0 | 0 | 0 |
| test_16556 | other | mixed | mixed;other | 458 | 0 | 0 | 0 | 0 |
| test_16793 | all5 | other | other;sarcasm | 239 | 0 | 0 | 0 | 0 |
| test_17582 | all5 | mixed | mixed;sarcasm | 147 | 0 | 0 | 0 | 0 |
| test_17946 | other | mixed | mixed;negation | 162 | 0 | 0 | 0 | 0 |
| test_18219 | all5 | other | other;mixed | 557 | 1 | 47 | 206 | 0 |
| test_18387 | other | mixed | mixed;sarcasm;negation | 114 | 0 | 0 | 0 | 0 |
| test_18829 | other | mixed | mixed;negation | 257 | 0 | 0 | 0 | 0 |
| test_18988 | other | mixed | mixed;negation | 474 | 0 | 0 | 0 | 0 |
| test_19371 | other | mixed | mixed | 180 | 0 | 0 | 0 | 0 |
| test_19947 | all5 | other | other;mixed | 290 | 0 | 0 | 0 | 0 |
| test_20451 | other | mixed | mixed;label_noise | 273 | 0 | 0 | 0 | 0 |
| test_21193 | other | mixed | mixed | 143 | 0 | 0 | 0 | 0 |
| test_21324 | all5 | mixed | mixed;negation | 347 | 0 | 0 | 0 | 0 |
| test_21540 | all5 | label_noise | label_noise | 200 | 0 | 0 | 0 | 0 |
| test_22132 | other | other | other;mixed | 248 | 0 | 0 | 0 | 0 |
| test_22240 | other | label_noise | label_noise;other | 147 | 0 | 0 | 0 | 0 |
| test_22259 | other | mixed | mixed | 1044 | 1 | 534 | 2376 | 0 |
| test_22317 | other | mixed | mixed | 201 | 0 | 0 | 0 | 0 |
| test_22412 | all5 | mixed | mixed;negation | 307 | 0 | 0 | 0 | 0 |
| test_22624 | all5 | other | other;mixed | 209 | 0 | 0 | 0 | 0 |
| test_22750 | all5 | mixed | mixed;long_truncated | 1023 | 1 | 513 | 2232 | 1 |
| test_22827 | all5 | mixed | mixed;negation | 151 | 0 | 0 | 0 | 0 |
| test_23077 | all5 | mixed | mixed;sarcasm | 117 | 0 | 0 | 0 | 0 |
| test_23257 | all5 | negation | negation;mixed | 189 | 0 | 0 | 0 | 0 |
| test_23967 | all5 | sarcasm | sarcasm;mixed | 238 | 0 | 0 | 0 | 0 |
| test_24383 | other | other | other | 77 | 0 | 0 | 0 | 0 |
| test_24629 | other | other | other | 174 | 0 | 0 | 0 | 0 |

## Step 4d: label_noise cases

File: `results/eval/error_analysis/label_noise_cases.csv`. The IMDb rating is not in the dataset (stanfordnlp/imdb has text and label only); score_written_in_text is a score the author typed in the review, found by regex.

| id | group | label | pred_dl | prob_dl | main_category | final_category | score_written_in_text | note |
|---|---|---|---|---|---|---|---|---|
| test_01785 | all5 | 0 | 1 | 0.9635 | label_noise | label_noise | none found |  |
| test_02840 | all5 | 0 | 1 | 0.9979 | label_noise | label_noise | none found |  |
| test_03287 | all5 | 0 | 1 | 0.9894 | label_noise | label_noise;mixed | none found |  |
| test_04366 | other | 0 | 1 | 0.7747 | label_noise | label_noise;mixed | 2 out of 5 | tự chấm 3.5/5 trong bài, nhãn 0 |
| test_04811 | all5 | 0 | 1 | 0.9989 | label_noise | label_noise | none found |  |
| test_05009 | all5 | 0 | 1 | 0.9511 | label_noise | label_noise;mixed | none found |  |
| test_06014 | all5 | 0 | 1 | 0.9669 | mixed | mixed;label_noise | none found |  |
| test_06593 | all5 | 0 | 1 | 0.9865 | label_noise | label_noise | 8/10 |  |
| test_07227 | other | 0 | 1 | 0.6869 | label_noise | label_noise;negation | none found |  |
| test_08975 | other | 0 | 1 | 0.8806 | label_noise | label_noise;other | none found | Mostly plot summary; final line ambiguous, no clear verdict. |
| test_09254 | other | 0 | 1 | 0.9079 | label_noise | label_noise;mixed | none found |  |
| test_09368 | all5 | 0 | 1 | 0.9980 | label_noise | label_noise | none found |  |
| test_12981 | all5 | 1 | 0 | 0.0661 | label_noise | label_noise | none found | chắc chắn là noise |
| test_12989 | other | 1 | 0 | 0.0061 | label_noise | label_noise | none found |  |
| test_13403 | all5 | 1 | 0 | 0.0163 | mixed | mixed;other;label_noise | none found | chê băng đĩa chứ không chê phim |
| test_15223 | all5 | 1 | 0 | 0.0049 | label_noise | label_noise;mixed | none found |  |
| test_16153 | other | 1 | 0 | 0.0048 | label_noise | label_noise;mixed | none found | Reviewer says mostly negative, “wasted opportunity”; praises mecha, recommends watching |
| test_20451 | other | 1 | 0 | 0.0819 | mixed | mixed;label_noise | none found | sẽ tham vấn nhóm |
| test_21540 | all5 | 1 | 0 | 0.4896 | label_noise | label_noise | none found |  |
| test_22240 | other | 1 | 0 | 0.4574 | label_noise | label_noise;other | none found |  |

## Step 5: heuristic tags vs hand labels (100 DL errors)

File: `results/eval/error_analysis/heuristic_vs_hand.csv`. Gold = hand label. lift = precision / hand_prevalence. `truncated` was NOT blind: the labelling tool greyed out the unseen middle.

| heuristic | hand_category | hand_label_kind | tp | fp | fn | tn | heuristic_fires | hand_prevalence | precision | recall | f1 | lift | kappa | kappa_band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truncated | long_truncated | any | 2 | 15 | 0 | 83 | 17 | 0.0200 | 0.1176 | 1.0000 | 0.2105 | 5.8824 | 0.1812 | slight |
| truncated | long_truncated | main | 1 | 16 | 0 | 83 | 17 | 0.0100 | 0.0588 | 1.0000 | 0.1111 | 5.8824 | 0.0940 | slight |
| neg_cue_strict | negation | any | 24 | 69 | 0 | 7 | 93 | 0.2400 | 0.2581 | 1.0000 | 0.4103 | 1.0753 | 0.0464 | slight |
| neg_cue_strict | negation | main | 5 | 88 | 0 | 7 | 93 | 0.0500 | 0.0538 | 1.0000 | 0.1020 | 1.0753 | 0.0079 | slight |
| neg_of_negative | negation | any | 5 | 5 | 19 | 71 | 10 | 0.2400 | 0.5000 | 0.2083 | 0.2941 | 2.0833 | 0.1781 | slight |
| neg_of_negative | negation | main | 3 | 7 | 2 | 88 | 10 | 0.0500 | 0.3000 | 0.6000 | 0.4000 | 6.0000 | 0.3571 | fair |
| neg_of_positive | negation | any | 4 | 7 | 20 | 69 | 11 | 0.2400 | 0.3636 | 0.1667 | 0.2286 | 1.5152 | 0.0915 | slight |
| neg_of_positive | negation | main | 2 | 9 | 3 | 86 | 11 | 0.0500 | 0.1818 | 0.4000 | 0.2500 | 3.6364 | 0.1946 | slight |
| negated_sentiment_any | negation | any | 8 | 11 | 16 | 65 | 19 | 0.2400 | 0.4211 | 0.3333 | 0.3721 | 1.7544 | 0.2031 | fair |
| negated_sentiment_any | negation | main | 4 | 15 | 1 | 80 | 19 | 0.0500 | 0.2105 | 0.8000 | 0.3333 | 4.2105 | 0.2760 | fair |
| has_contrast_cue | mixed | any | 67 | 18 | 5 | 10 | 85 | 0.7200 | 0.7882 | 0.9306 | 0.8535 | 1.0948 | 0.3353 | fair |
| has_contrast_cue | mixed | main | 43 | 42 | 4 | 11 | 85 | 0.4700 | 0.5059 | 0.9149 | 0.6515 | 1.0763 | 0.1171 | slight |
| mixed_candidate | mixed | any | 42 | 6 | 30 | 22 | 48 | 0.7200 | 0.8750 | 0.5833 | 0.7000 | 1.2153 | 0.2925 | fair |
| mixed_candidate | mixed | main | 27 | 21 | 20 | 32 | 48 | 0.4700 | 0.5625 | 0.5745 | 0.5684 | 1.1968 | 0.1780 | slight |

## Step 6: decisions (last log row per id)

File: `results/eval/error_analysis/ai_decisions.csv`. no_log_row = labelled before the AI suggestions were used (present in the 2026-11-12 05:19 backup). revised_after_ai = 1 in ai_vs_final.csv marks such a label changed later with the suggestion seen (saved outside the tool UI, so it has no log row).

| decision | n |
|---|---|
| accepted | 69 |
| edited | 28 |
| no_log_row | 3 |

## Step 6: AI suggestion vs final label

File: `results/eval/error_analysis/ai_agreement.csv`. This is NOT inter-annotator agreement.

| comparison | subset | n | n_same | raw_agreement | cohen_kappa |
|---|---|---|---|---|---|
| AI suggestion main vs final main (NOT inter-annotator agreement) | all 100 reviews | 100 | 97 | 0.9700 | 0.9566 |
| AI suggestion main vs final main (NOT inter-annotator agreement) | 98 reviews whose final label was set with the AI suggestion seen | 98 | 96 | 0.9796 | 0.9706 |

## Step 6: edit rate per AI-suggested main category

File: `results/eval/error_analysis/ai_edit_rate_by_category.csv`. 

| ai_main | n_suggested | main_kept | main_changed | main_edit_rate | any_edit | final_still_has_it |
|---|---|---|---|---|---|---|
| sarcasm | 9 | 9 | 0 | 0.0000 | 4 | 9 |
| negation | 4 | 4 | 0 | 0.0000 | 0 | 4 |
| mixed | 48 | 46 | 2 | 0.0417 | 15 | 48 |
| long_truncated | 1 | 1 | 0 | 0.0000 | 1 | 1 |
| label_noise | 16 | 16 | 0 | 0.0000 | 8 | 16 |
| other | 22 | 21 | 1 | 0.0455 | 2 | 21 |

## Step 6: AI main (rows) vs final main (columns)

File: `results/eval/error_analysis/ai_confusion.csv`. 

| ai_main \ final_main | sarcasm | negation | mixed | long_truncated | label_noise | other |
|---|---|---|---|---|---|---|
| sarcasm | 9 | 0 | 0 | 0 | 0 | 0 |
| negation | 0 | 4 | 0 | 0 | 0 | 0 |
| mixed | 0 | 1 | 46 | 0 | 1 | 0 |
| long_truncated | 0 | 0 | 0 | 1 | 0 | 0 |
| label_noise | 0 | 0 | 0 | 0 | 16 | 0 |
| other | 0 | 0 | 1 | 0 | 0 | 21 |

## Step 6: edited cases

File: `results/eval/error_analysis/ai_edited_cases.csv`. 

| id | file | ai_main | ai_also | ai_unsure | final_main | final_also | final_unsure | main_changed | also_changed | unsure_changed | log_decision | labelled_before_ai | revised_after_ai |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_01784 | B | sarcasm | mixed | yes | sarcasm | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_02549 | A | mixed | sarcasm | yes | mixed | sarcasm |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_02686 | B | mixed |  | yes | mixed |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_02857 | B | sarcasm | other | yes | sarcasm | other |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_03287 | B | label_noise | mixed | yes | label_noise | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_04366 | A | label_noise | mixed | yes | label_noise | mixed |  | 0 | 0 | 1 | no_log_row | 1 | 1 |
| test_04758 | B | mixed |  | yes | mixed |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_05009 | A | label_noise |  |  | label_noise | mixed |  | 0 | 1 | 0 | edited | 0 | 0 |
| test_06014 | B | mixed |  | yes | mixed | label_noise | yes | 0 | 1 | 0 | edited | 0 | 0 |
| test_07555 | A | sarcasm |  | yes | sarcasm |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_08975 | A | label_noise |  | yes | label_noise | other |  | 0 | 1 | 1 | edited | 0 | 0 |
| test_08990 | A | mixed | negation |  | negation | mixed | yes | 1 | 1 | 1 | edited | 0 | 0 |
| test_09254 | B | label_noise | mixed | yes | label_noise | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_09368 | A | label_noise |  | yes | label_noise |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_10315 | A | other | mixed | yes | other | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_12498 | B | mixed |  | yes | mixed |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_13403 | A | mixed | other | yes | mixed | other;label_noise |  | 0 | 1 | 1 | edited | 0 | 0 |
| test_14218 | B | long_truncated | mixed | yes | long_truncated | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_15223 | B | label_noise | mixed | yes | label_noise | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_16153 | A | mixed |  | yes | label_noise | mixed |  | 1 | 1 | 1 | edited | 0 | 0 |
| test_17582 | A | mixed |  |  | mixed | sarcasm |  | 0 | 1 | 0 | edited | 0 | 0 |
| test_18387 | A | other |  | yes | mixed | sarcasm;negation |  | 1 | 1 | 1 | no_log_row | 1 | 0 |
| test_19371 | A | mixed |  | yes | mixed |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_20451 | A | mixed |  | yes | mixed | label_noise | yes | 0 | 1 | 0 | edited | 0 | 0 |
| test_21324 | A | mixed |  | yes | mixed | negation |  | 0 | 1 | 1 | edited | 0 | 0 |
| test_22240 | A | label_noise | other | yes | label_noise | other |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_22259 | A | mixed |  | yes | mixed |  |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_22412 | A | mixed | negation | yes | mixed | negation |  | 0 | 0 | 1 | edited | 0 | 0 |
| test_22750 | A | mixed |  | yes | mixed | long_truncated |  | 0 | 1 | 1 | edited | 0 | 0 |
| test_23967 | A | sarcasm | mixed | yes | sarcasm | mixed |  | 0 | 0 | 1 | edited | 0 | 0 |
