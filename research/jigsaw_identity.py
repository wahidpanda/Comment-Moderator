"""Identity FPR audit on Jigsaw Unintended Bias (identity annotation columns)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.eval import pred_harmful_from_results
from research.labels import BINARY_THRESHOLD, normalize_text_column
from research.stats import wilson_ci

JIGSAW_IDENTITY_COLS = [
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
]

CACHE_PARQUET = Path("data/raw/jigsaw_unintended_bias.parquet")
IDENTITY_SLICE_MIN = 0.5


def identity_mention_row_mask(frame: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    for col in JIGSAW_IDENTITY_COLS:
        if col not in frame.columns:
            continue
        scores = pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy()
        mask |= scores >= IDENTITY_SLICE_MIN
    return mask


def load_jigsaw_identity_frame(
    *,
    sample: int | None = 8000,
    seed: int = 42,
    cache_path: Path = CACHE_PARQUET,
    exclude_hashes: set[str] | None = None,
) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from datasets import load_dataset

        ds = load_dataset("james-burton/jigsaw_unintended_bias100K", split="train")
        df = ds.to_pandas()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    df = df.rename(columns={"comment_text": "text"})
    df["text"] = normalize_text_column(df, "text")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if exclude_hashes:
        from research.civil_overlap import _text_hash

        keep = [_text_hash(t) not in exclude_hashes for t in df["text"]]
        df = df.loc[keep].reset_index(drop=True)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    return df


def identity_audit_jigsaw(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    toxic_col: str = "target",
) -> dict[str, Any]:
    toxic = frame[toxic_col].astype(float).to_numpy()
    benign = toxic < BINARY_THRESHOLD
    fp_mask = (y_pred == 1) & benign
    overall_fpr = float(fp_mask.sum() / max(int(benign.sum()), 1))
    o_lo, o_hi = wilson_ci(int(fp_mask.sum()), int(benign.sum()))

    groups: dict[str, Any] = {}
    for col in JIGSAW_IDENTITY_COLS:
        if col not in frame.columns:
            continue
        scores = pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy()
        slice_mask = benign & (scores >= IDENTITY_SLICE_MIN)
        n_slice = int(slice_mask.sum())
        if n_slice == 0:
            groups[col] = {"benign_rows": 0, "false_positives": 0, "fpr": None}
            continue
        fps = int(((y_pred == 1) & slice_mask).sum())
        lo, hi = wilson_ci(fps, n_slice)
        groups[col] = {
            "benign_rows": n_slice,
            "false_positives": fps,
            "fpr": float(fps / n_slice),
            "fpr_ci95_low": lo,
            "fpr_ci95_high": hi,
        }

    id_mask = identity_mention_row_mask(frame)
    non_id = benign & (~id_mask)
    id_benign = benign & id_mask
    id_fps = int(((y_pred == 1) & id_benign).sum())
    non_fps = int(((y_pred == 1) & non_id).sum())
    id_lo, id_hi = wilson_ci(id_fps, int(id_benign.sum()))
    non_lo, non_hi = wilson_ci(non_fps, int(non_id.sum()))

    return {
        "source": "jigsaw_unintended_bias (identity annotation columns)",
        "methodology": {
            "benign_definition": f"target toxicity score < {BINARY_THRESHOLD} (non-toxic)",
            "identity_row": f"any identity annotation column >= {IDENTITY_SLICE_MIN}",
            "sample": "random rows from james-burton/jigsaw_unintended_bias100K after optional Civil-train text dedup",
            "fp_definition": "predicted harmful on benign row (spam flags count as harmful for this audit)",
        },
        "overall_benign_rows": int(benign.sum()),
        "overall_false_positives": int(fp_mask.sum()),
        "overall_fpr": overall_fpr,
        "overall_fpr_ci95_low": o_lo,
        "overall_fpr_ci95_high": o_hi,
        "benign_identity_rows": int(id_benign.sum()),
        "benign_non_identity_rows": int(non_id.sum()),
        "fpr_benign_identity_rows": id_fps / max(int(id_benign.sum()), 1),
        "fpr_benign_identity_ci95_low": id_lo,
        "fpr_benign_identity_ci95_high": id_hi,
        "fpr_benign_non_identity_rows": non_fps / max(int(non_id.sum()), 1),
        "fpr_benign_non_identity_ci95_low": non_lo,
        "fpr_benign_non_identity_ci95_high": non_hi,
        "per_identity_group": groups,
        "identity_slice_threshold": IDENTITY_SLICE_MIN,
        "benign_toxic_threshold": BINARY_THRESHOLD,
    }


def run_jigsaw_identity_audit(
    pipeline: Any,
    *,
    sample: int = 8000,
    seed: int = 42,
    batch_size: int = 64,
    exclude_hashes: set[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    from engine.schemas import CommentInput

    frame = load_jigsaw_identity_frame(sample=sample, seed=seed, exclude_hashes=exclude_hashes)
    comments = [CommentInput(id=str(i), text=t) for i, t in enumerate(frame["text"].tolist())]
    resp = pipeline.analyze_batch(comments, transformer_batch_size=batch_size)
    y_pred = pred_harmful_from_results(resp.results)
    return identity_audit_jigsaw(frame, y_pred), y_pred, frame