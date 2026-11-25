# Final artifact review

- Final report: `NLP_CC01_Group04_Report.pdf`, 30 pages, English.
- Source build: `build.ps1` completed successfully with Tectonic.
- Rendered all 30 final pages with Poppler and inspected all contact sheets.
- Inspected the cover, demo screenshots, five confusion matrices, qualitative
  error table and contribution table at full page size.
- No clipped content, overlapping tables, missing figures or incorrect
  Vietnamese name glyphs were observed.
- Final TeX log contains no overfull boxes, missing glyphs or unresolved
  citations/references. Contents, lists of figures/tables and bookmarks resolve.
- Windows font lookup uses installed Times New Roman/Consolas; these fonts are
  embedded in the PDF. Tectonic emits its standard external-font portability
  notices and a fontconfig notice, but its native Windows font lookup succeeds.
- Student IDs are confirmed: Person 1 2352889, Person 2 2560175, Person 3 2352295 and Person 4 2352649.
- Every member receives an individual contribution percentage of 100% under the assignment grading rule; there is no total row.
- The report documents the supplied evidence limits; no model retraining or
  fresh demonstration tests are claimed.

See `final_report_validation.json` for the final PDF checksum and size, and
`validation_summary.txt` for numerical checks and evidence limitations.

## Coverage revision, 25 November 2026

- Appendix B.1 now confirms all three required code components and gives project-relative file/function locations.
- B.4 Rebuilding the report was removed, including its contents/bookmark entry.
- The rebuilt PDF remains 30 pages. Changed PDF pages 3, 4, 29 and 30 were rendered with Poppler and visually inspected.
- Artifact locations start together with their table on the next page; experiment commands remain together. No overflow, missing glyph or unresolved reference warnings were found.
- Cover text and all other PDF page text are unchanged. Student IDs remain ID1--ID4.

## Reference revision, 25 November 2026

- Removed project references [10] and [11] and their citations in Section 5.5 and Appendix A.1. The remaining nine references retain their numbering.
- Rebuilt using the cached Tectonic bundle. Changed PDF pages 16, 26 and 27 were rendered with Poppler and visually inspected; no layout defects were observed.
- The report remains 30 pages; the cover and all other page text are unchanged. No overfull boxes, missing glyphs or unresolved citations/references were found.

## Student ID and contribution revision, 25 November 2026

- Updated the cover and task-allocation table with the four supplied student IDs.
- Each member receives 100% as an individual score multiplier. Removed the summed total and replaced the former shared-contribution explanation.
- Updated the report README, editorial plan and validation metadata to match. Earlier revision notes describe the report at those earlier stages.
- Rebuilt with cached Tectonic; PDF pages 1, 4, 27 and 28 rendered with Poppler and visually inspected. The report remains 30 pages; no overflow, missing glyph or unresolved citation/reference warnings were found.

## Team narrative and submission layout revision, 25 November 2026

- Corrected Pathania Sohil, changed editorial language to team voice, and corrected four classical plus one deep pipeline.
- Kept all 76 package files unchanged inside .eval_demo_package; removed only temporary root copies. Appendix B documents the actual paths and in-place setup.
- PDF is self-contained, with no references to files excluded from the submission. The final file is .report/NLP_CC01_Group04_Report.pdf, 31 pages.
- All 31 pages rendered and inspected on contact sheets. Cover, introduction, workflow, experiments, contribution table and Appendix B inspected at full size. Final two pages re-rendered after pagination adjustment. No clipping, overfull boxes, missing glyphs or unresolved references/citations.
- Saved numerical evidence and original documents passed validation. Model training and live demo checks were not rerun for this revision.
