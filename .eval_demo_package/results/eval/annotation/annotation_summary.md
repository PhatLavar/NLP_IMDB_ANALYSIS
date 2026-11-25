# Annotation summary

**These percentages come from a sample of 100 misclassified reviews, not from all errors.** The sample has 40 reviews wrong for all 5 models (group all5) and 60 reviews wrong for DistilBERT but not for all 5 models (group other).

- Reviews: 100 (annotator A: 50, annotator B: 50)
- Unsure reviews: 9
- discussion.csv not found: final_category = the annotator's category
- Reviews whose category changed after the discussion: 0

## Main category (the first value of final_category)

| category | all (n=100) | all5 (n=40) | other (n=60) |
|---|---:|---:|---:|
| sarcasm | 9 (9%) | 1 (2%) | 8 (13%) |
| negation | 5 (5%) | 3 (8%) | 2 (3%) |
| mixed | 47 (47%) | 17 (42%) | 30 (50%) |
| long_truncated | 1 (1%) | 0 (0%) | 1 (2%) |
| label_noise | 17 (17%) | 10 (25%) | 7 (12%) |
| other | 21 (21%) | 9 (22%) | 12 (20%) |

## Every category value

A review with several values is counted once for each value, so a column can add up to more than 100%.

| category | all (n=100) | all5 (n=40) | other (n=60) |
|---|---:|---:|---:|
| sarcasm | 18 (18%) | 6 (15%) | 12 (20%) |
| negation | 24 (24%) | 9 (22%) | 15 (25%) |
| mixed | 72 (72%) | 29 (72%) | 43 (72%) |
| long_truncated | 2 (2%) | 1 (2%) | 1 (2%) |
| label_noise | 20 (20%) | 12 (30%) | 8 (13%) |
| other | 28 (28%) | 11 (28%) | 17 (28%) |
