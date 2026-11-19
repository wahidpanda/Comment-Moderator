# Evaluation report

Regenerated after: per-scenario **benign FPR** (`FP / benign_count`), threat heads + rules, UCI spam layer (gated), Jigsaw identity audit, `docs/eval_metrics.json`.

## Civil Comments (stratified **3,000** test, 300k train)

| Scenario | Harmful F1 | 95% CI | FP | FN | Benign n | **Benign FPR** (95% Wilson) | → Transformer | p50 / p95 ms |
|----------|------------|--------|----|----|----------|----------------------------|---------------|--------------|
| **A** rules+LR+spam | 0.503 | 0.452 – 0.552 | 121 | 122 | 2755 | **0.044** (0.037 – 0.052) | 0% | 19 / 34 |
| **B** Detoxify only | 0.728 | 0.684 – 0.770 | 80 | 59 | 2755 | **0.029** (0.023 – 0.036) | 100% | 360 / 558 |
| **C** hybrid | **0.596** | 0.547 – 0.640 | 151 | 77 | 2755 | **0.055** (0.047 – 0.057) | 10.7% | 24 / 303 |

Benign FPR is computed **per scenario** on the same 3k subsample (`benign_count` = gold non-harmful rows).

### Threat recall

| Scenario | Civil `threat ≥ 0.5` (n=10 positives) | Creator eval threat (n=6) |
|----------|--------------------------------------|---------------------------|
| A | **0.20** (2/10) | **0.67** (4/6) |
| C | **0.30** (3/10) | **0.67** (4/6) |

Indirect patterns (`watch your back`, `I will find you`, `hope your house burns/floods`) + LR threat head + Detoxify threat ≥ **0.35** → `needs_human=true`; label THREAT ≥ **0.5**.

### Spam classifier (UCI YouTube Spam Collection)

| Metric | Value |
|--------|--------|
| Test F1 | **0.967** |
| Precision | **0.989** |
| Recall | **0.947** |
| Threshold | **0.60** (pipeline gates low scores unless URL/promo cues) |
| Pipeline spam calls (3k A) | 78 |

### Identity FPR — **Jigsaw Unintended Bias** (`james-burton/jigsaw_unintended_bias100K`, identity annotation ≥ 0.5, benign = `target < 0.5`, n≈3766 benign in 4k sample)

| Scenario | Overall benign FPR | Wilson 95% |
|----------|-------------------|------------|
| A | ~0.21 | see JSON |
| C | ~0.29 | see JSON |

Per-group tables (e.g. `muslim`, `black`, `white`, …) with **benign_rows**, **FPR**, and Wilson CIs: `identity_audit_jigsaw` in `docs/eval_metrics.json` — no longer text mention masks.

### Creator eval (170 rows)

- **critique** / **sarcasm**: **0%** label accuracy → marked **experimental** in UI (`docs/LABELS.md`).
- **threat** harmful accuracy **0.67** (A/C) after rule + score changes.

### Hand-labeled channel comments (held out)

```bash
python -m research.import_video_comments path/to/youtube_export.csv
# fill label_expected in research/data/creator_handlabeled.csv
python -m research.eval_handlabeled
```

Never used for rules or LR training.

## Limits

1. Spam model is YouTube-specific; gating limits Civil FPR blow-up but domain shift remains.
2. Jigsaw identity audit sample is 4k rows — small per-slice Wilson bands.
3. Civil threat positives in 3k subsample = **10** — threat recall noisy.
4. Cached scenario **B** Detoxify latency from prior run (`make eval --force` to refresh B).

## Reproduce

```bash
make train-spam
make eval    # research.eval_bundle
```