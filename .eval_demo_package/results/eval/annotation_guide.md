# Annotation guide: why did the model get this review wrong?

Each review below was misclassified by the DistilBERT model. The 100-review sample is split into two files of 50 reviews, one per annotator, with no shared reviews:

- `results/eval/annotation/annotator_A_*.csv` (exactly one file matching this pattern, for example `annotator_A_kiet.csv`)
- `results/eval/annotation/annotator_B_*.csv` (exactly one file matching this pattern, for example `annotator_B_Phat.csv`)

Label only the reviews in your own file. Fill in three columns and leave every other column as it is.

- `category`: one or more of `sarcasm`, `negation`, `mixed`, `long_truncated`, `label_noise`, `other`. Separate several values with `;` and put the main reason first, for example `negation;mixed`.
- `unsure`: write `yes` if you cannot decide between categories or you doubt your choice. Otherwise leave it empty. Fill in `category` with your best guess either way.
- `note`: optional and short, for example the key phrase.

## How to label

1. Read the true `label` first (0 = negative, 1 = positive), then read the full `text`.
2. Decide whether the text clearly supports the model's prediction (`pred_dl`) instead of the label. If it does, the label is probably wrong: use `label_noise`.
3. Otherwise, choose the reason or reasons the text was hard to classify. Label from the text only.

When both files are finished, all `unsure` reviews are discussed together, and the agreed category is written into the discussion file.

## Categories

| Value | Definition | Made-up example |
|---|---|---|
| `sarcasm` | The literal words say the opposite of what the writer means. | "Oh great, another two hours of my life I will never get back. Truly a masterpiece." (negative) |
| `negation` | A negation flips a sentiment word, and the model probably missed it. | "The acting was not bad, and the ending was not boring at all." (positive) |
| `mixed` | The review praises some parts and criticises others, and the overall verdict is subtle or comes late. | "Beautiful camera work and a great score, but the story is so weak that I cannot recommend it." (negative) |
| `long_truncated` | The review is longer than the model's 512-token limit, and the decisive opinion sits in the part that was cut. The DL model keeps the beginning (about 25%) and the end of a long review, and drops the middle (`truncation = head_tail` in `results/dl_config.csv`). | A long plot summary, with "In the end I loved it" placed only in the middle of the text. (positive) |
| `label_noise` | The text clearly agrees with the model's prediction, not with the true label. | "Dull, slow, and badly acted. Avoid." with label = 1 (positive) |
| `other` | None of the above. Explain the reason in `note`. | A review that only describes the plot and states no opinion. |
