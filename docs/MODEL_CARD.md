# Model card (LR v1)

## Intended use

Assist creators prioritizing replies and human review of potentially harmful comments. **Not** for automated removal.

## Training data

- [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (Hugging Face)
- Optional: Jigsaw Toxic Comments (Kaggle, user-supplied CSV)

Harmful label: any toxicity dimension ≥ 0.5 (Civil) or any Jigsaw toxic column = 1.

## Features

TF-IDF word n-grams (1–2) + char wb n-grams (3–5), max 200k features each; LogisticRegression / LinearSVC / SGD compared on validation F1.

## Metrics

Stored in `models/lr/v1/metrics.json` after training. Test set evaluated once when running full `make train` (not `train-smoke`).

## Limitations

- English-centric training corpora
- Sarcasm and context-poor short comments remain hard
- LinkedIn / YouTube DOM changes break extension parsers until selectors updated