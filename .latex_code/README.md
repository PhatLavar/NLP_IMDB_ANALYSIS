# Group 04 final report

The report follows the eight sections required by the course assignment. It is
written in English using our HCMUT cover and project documentation.
Team: Nguyễn Tấn Phát (2352889), Pathania Sohil (2560175),
Ngô Đăng Hào (2352295) and Lương Thế Kiệt (2352649).
Each member's contribution percentage is 100% for completing their assigned
responsibilities, so their individual score equals the group score under the
assignment's grading formula.

- Main source: `main.tex` (XeLaTeX/Tectonic, Times New Roman; Vietnamese names supported).
- Sections: `sections/`; numbered section labels are controlled by LaTeX, not filenames.
- Editorial plan and source mapping: `REPORT_PLAN.md`.
- Original material: `contents/` (never edited by report preparation).
- Extracted original Word images: `assets/report/` (12 figures, all used).
- Generated numerical table bodies: `tables/`.
- Source hashes, figure provenance, selected CSVs and validation outcome: `evidence/`.
- Final output: `../.report/NLP_CC01_Group04_Report.pdf`.

## Build on Windows

From this directory:

```powershell
./build.ps1
# Alternatively, with a fully provisioned MiKTeX/TeX Live installation:
./build.ps1 -Engine xelatex
```

The preparation script needs Python's standard library only. Tectonic retrieves
LaTeX packages if they are not already cached. The alternative engine requires
fontspec, geometry, amsmath, amssymb, graphicx, booktabs, tabularx, longtable,
caption, float, placeins, fancyhdr, lastpage, enumitem, xcolor, tikz, hyperref and their
transitive dependencies (including stringenc). Windows fonts Times New Roman,
Arial and Consolas are used. This report was built with Tectonic because the
local MiKTeX installation lacks stringenc.

For Overleaf, upload this directory excluding `build/`, select XeLaTeX and
replace Windows fonts with installed Unicode fonts if necessary. Keep `contents/`
if you want to rerun extraction outside Overleaf; compilation itself uses the
already extracted images and tables.

## Evidence boundaries

No models were retrained for this revision. The 25,000 saved DL test predictions
reproduce its confusion matrix and ROC-AUC. All five confusion-count rows
reproduce the root result CSV metrics. The final 100 annotation records reproduce
category counts. Prepared-data SHA-256 values are verified against metadata.
Classical per-review predictions are not included; their ROC-AUC and shared-error
figures are recorded in `.eval_demo_package/results/eval/`. The 12/12 demo checks
are recorded experiment evidence, not new checks performed for this revision.

Evaluation and demo sources stay in `../.eval_demo_package/eval/` and
`../.eval_demo_package/demo/`. The package is unchanged. Appendix B and the root
README explain how to run its scripts in place and locate its recorded outputs.
