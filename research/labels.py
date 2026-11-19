"""Map Civil Comments / Jigsaw columns to moderation labels and training targets."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Civil Comments toxicity columns (multi-label floats 0–1)
CIVIL_TOXIC_COLS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]

# Subgroup toxicity scores in Civil Comments (attacks *toward* each identity).
CIVIL_IDENTITY_GROUP_COLS = [
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "black",
    "white",
    "asian",
    "physical_disability",
    "psychiatric_or_mental_illness",
]

LR_HEAD_COLS = ["toxicity", "insult", "threat", "identity_attack"]

MIN_TRAIN_ROWS = 10_000

# Jigsaw train.csv columns (0/1)
JIGSAW_TOXIC_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Output schema labels (engine Phase 2+)
class ModerationLabel(str, Enum):
    PRAISE = "praise"
    QUESTION = "question"
    NEUTRAL = "neutral"
    CRITIQUE = "critique"
    SPAM = "spam"
    TOXIC = "toxic"
    THREAT = "threat"
    IDENTITY_ATTACK = "identity_attack"


# Civil threshold for binarizing float scores (matches common literature ~0.5)
BINARY_THRESHOLD = 0.5


def _row_max_bin(series: pd.DataFrame, cols: list[str], threshold: float) -> np.ndarray:
    present = [c for c in cols if c in series.columns]
    if not present:
        return np.zeros(len(series), dtype=np.int8)
    return (series[present].astype(float).max(axis=1) >= threshold).astype(np.int8)


def civil_harmful_binary(df: pd.DataFrame, threshold: float = BINARY_THRESHOLD) -> np.ndarray:
    """1 if any toxicity dimension >= threshold."""
    return _row_max_bin(df, CIVIL_TOXIC_COLS, threshold)


def jigsaw_harmful_binary(df: pd.DataFrame) -> np.ndarray:
    """1 if any Jigsaw toxic column is 1."""
    return _row_max_bin(df, JIGSAW_TOXIC_COLS, threshold=0.5)


def primary_toxic_dimension_civil(row: pd.Series, threshold: float = BINARY_THRESHOLD) -> str:
    """Pick dominant dimension for per-class analysis (not used in Phase 1 LR binary)."""
    priority = [
        ("threat", ModerationLabel.THREAT),
        ("identity_attack", ModerationLabel.IDENTITY_ATTACK),
        ("severe_toxicity", ModerationLabel.TOXIC),
        ("toxicity", ModerationLabel.TOXIC),
        ("insult", ModerationLabel.TOXIC),
        ("obscene", ModerationLabel.TOXIC),
        ("sexual_explicit", ModerationLabel.TOXIC),
    ]
    for col, label in priority:
        if col in row.index and float(row[col]) >= threshold:
            return label.value
    return ModerationLabel.NEUTRAL.value


def identity_mention_term_masks(df: pd.DataFrame, text_col: str = "text") -> dict[str, np.ndarray]:
    """Benign-audit slices: comment text contains an identity term (rules config)."""
    import re

    from engine.rules import load_rules

    terms = load_rules().get("identity_terms", [])
    texts = normalize_text_column(df, text_col)
    masks: dict[str, np.ndarray] = {}
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
        masks[term] = texts.str.contains(pattern, regex=True).to_numpy()
    return masks


def identity_mention_mask_civil(df: pd.DataFrame) -> np.ndarray:
    """Slice for identity audit: rows with identity_attack score >= threshold."""
    if "identity_attack" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["identity_attack"].astype(float).values >= BINARY_THRESHOLD


def normalize_text_column(df: pd.DataFrame, text_col: str) -> pd.Series:
    return df[text_col].fillna("").astype(str).str.strip()


def build_unified_frame(
    civil: pd.DataFrame | None,
    jigsaw: pd.DataFrame | None,
    *,
    civil_text_col: str = "text",
    jigsaw_text_col: str = "comment_text",
) -> pd.DataFrame:
    """Merge Civil + Jigsaw into one frame: text, harmful (0/1), source."""
    frames: list[pd.DataFrame] = []
    if civil is not None and len(civil):
        c = pd.DataFrame(
            {
                "text": normalize_text_column(civil, civil_text_col),
                "harmful": civil_harmful_binary(civil),
                "source": "civil_comments",
            }
        )
        frames.append(c)
    if jigsaw is not None and len(jigsaw):
        j = pd.DataFrame(
            {
                "text": normalize_text_column(jigsaw, jigsaw_text_col),
                "harmful": jigsaw_harmful_binary(jigsaw),
                "source": "jigsaw",
            }
        )
        frames.append(j)
    if not frames:
        raise ValueError("No datasets provided")
    unified = pd.concat(frames, ignore_index=True)
    unified = unified[unified["text"].str.len() > 0]
    return unified.reset_index(drop=True)


def stratified_split_indices(
    y: np.ndarray,
    *,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / val / test index arrays, stratified on binary harmful."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    idx_train, idx_rest, _, y_rest = train_test_split(
        idx, y, test_size=val_frac + test_frac, random_state=seed, stratify=y
    )
    relative_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=relative_test, random_state=seed, stratify=y_rest
    )
    return idx_train, idx_val, idx_test