# Decisions

## Threshold objective (LR bands)

We sweep `t_low` / `t_high` on the validation split with these constraints (see `research/train.py`):

- Max **30%** of comments in the ambiguous band (reserved for transformer).
- Maximize **binary harmful F1** on validation using LR decisions on safe/harmful bands and 0.5 cutover for ambiguous proxy during tuning.

**Latest 300k Civil retrain (seed=42):**

| Field | Value |
|-------|--------|
| Objective | Maximize val harmful F1 subject to ambiguous_fraction ≤ 0.30 |
| Chosen `t_low` | **0.45** |
| Chosen `t_high` | **0.85** |
| Val harmful F1 (proxy) | **0.594** |
| Ambiguous fraction on val | **8.2%** |
| Test harmful F1 (LR only, held-out) | **0.602** |

Persisted in `models/lr/v1/thresholds.json`.

## Identity mitigation (2026-11-16)

1. **Pipeline:** identity term + LR `p < t_high` → escalate to Detoxify (skip LR-only safe label).
2. **Training:** 20 synthetic benign identity sentences (`research/identity_augment.py`) + sample weights (×2 benign civil identity mentions, ×3 augment rows).
3. **Eval:** Wilson 95% intervals on per-term benign FPR (`research/stats.py`).

## Classical vs neural

- Primary production path: **rules + LR + Detoxify hybrid** (free, local CPU).
- Optional DistilBERT fine-tune on Colab/Kaggle if LR validation F1 is weak — see `research/notebooks/README.md`.

## Identity policy

Identity term mentions **alone** must not trigger rules. Identity-related false positives are audited in `make eval` / `eval_bundle` and documented in `docs/EVALUATION_REPORT.md`.