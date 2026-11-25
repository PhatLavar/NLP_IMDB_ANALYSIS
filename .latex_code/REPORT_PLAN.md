# Final report plan and evidence map

Report language: English. Output: `.report/NLP_CC01_Group04_Report.pdf`.
The course brief prescribes eight sections, in the order below. The existing
HCMUT cover, lecturer, course, group and academic-year information are retained.
Names and student IDs are confirmed by the report editor: Person 1 2352889,
Person 2 2560175, Person 3 2352295 and Person 4 2352649. Each member receives
a contribution percentage of 100% for completing their assigned responsibilities;
this percentage multiplies the group score to determine the individual score.

| Required section | Content and primary evidence |
|---|---|
| 1 Introduction | Task, motivation, objectives and scope; Person 1 |
| 2 Related Work (brief) | Linear text classifiers, n-grams, embeddings, BERT/DistilBERT and LoRA; Persons 2--3 and original papers |
| 3 Dataset Description | Source, fixed splits, class counts, lengths, sample excerpts, duplicate policy; Person 1, notebook, `imdb_outputs.zip` |
| 4 Methodology | Three motivated preprocessing steps; TF-IDF and mean Word2Vec; Logistic Regression and LinearSVC; DistilBERT + LoRA; Persons 1--3 and implementation |
| 5 Experiments | Settings, eight-run validation search, checkpoint selection, metrics, reproducibility and Streamlit demonstration; Persons 2--4, recorded CSVs, `eval-demo` branch |
| 6 Results | Five-model comparison, confusion matrices, ranking scores, timing caveats; root CSVs and Person 4 evaluation package |
| 7 Error Analysis | Stratified 100-review sample, multilabel categories, examples, truncation, annotation assistance and limitations; Person 4 package |
| 8 Conclusion | Answer the experimental questions, quantified improvement, practical interpretation and future work |
| Appendix A | Detailed task allocation and individual contribution percentages (100% each); commit evidence for each member |
| Appendix B | Requirement coverage, artifact locations and reproduction commands |

## Editorial decisions

- Consolidate repeated data/metric descriptions instead of concatenating the four drafts.
- Use the recorded numbers, not newly trained models. Check reported metrics against
  root CSVs and supplied confusion counts; independently recompute DL metrics from
  all 25,000 saved test predictions.
- Keep an explicit distinction between validation tuning results (best epoch 6)
  and the final retraining run (best epoch 8).
- Correct claims in the drafts where code is more precise: WordPiece can still
  emit unknown tokens; head-tail input is 128 + 382 review wordpieces plus two
  special tokens; the parameter percentage uses the script's base-model denominator.
- Disclose classical test exposure, unequal tuning, single-seed results, internal
  test duplicates, uncertain labels, uncalibrated SVM scores and AI-assisted annotation.
- Distinguish supplied validation evidence from checks rerun for this report.
  The four classical prediction files are absent from the supplied evaluation ZIP;
  ROC-AUC and shared-error counts come from the recorded evaluation outputs.
- Keep evaluation and demo files in `.eval_demo_package/` unchanged. Document its
  real paths and in-place setup; preserve historical branch/commit evidence separately.
- Preserve every file under `contents/` byte for byte. Images are extracted to
  `assets/report/`; generated numerical tables go in `tables/`.

## Figure mapping (all embedded figures are retained)

| Extracted asset | Placement |
|---|---|
| person1_image1.png | Dataset: fixed-split label counts |
| person1_image2.png | Dataset: training length histogram |
| person2_image1.png | Methodology: classical workflow |
| person2_image2.png | Methodology: representation comparison |
| person2_image3.png | Results: classical accuracy/F1 |
| person3_image1.png | Methodology: DistilBERT + LoRA workflow |
| person3_image2.png | Experiments: eight-run validation F1 |
| person4_rId23.png | Results: confusion matrices (rearranged into two readable panels) |
| person4_rId27.png | Error analysis: main and multilabel categories |
| person4_rId34.png | Experiments/demo: initial interface |
| person4_rId35.png | Experiments/demo: negation example |
| person4_rId36.png | Experiments/demo: long-review warning |

## Git evidence reviewed on 25 November 2026

`master` at c6ce9ef: data 2fe2e97 (PhatLavar), ML 3e6508f and 11893c7
(Pathania Sohil), DL b9b531e and 64d08c7 (Ngo Dang Hao), report materials
c6ce9ef (PhatLavar). `eval-demo` at 8cf52a4: demo 2737ca5, screenshots
895c203, setup fixes 8cf52a4 (Lương Thế Kiệt). Commit file changes and
demo implementation were inspected. Line counts are evidence of artifacts,
not a basis for estimating individual contribution percentages.

## Completion checks

1. Generate assets/tables and validate numerical consistency.
2. Compile LaTeX with cross-references and bibliography resolved.
3. Check logs for missing glyphs, unresolved references and overflowing boxes.
4. Render every PDF page, inspect contact sheets and full-size critical pages.
5. Recheck source hashes, output filename and final working-tree scope.
