"""Detect text overlap between Civil train split and external eval frames."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from research.labels import civil_harmful_binary, normalize_text_column, stratified_split_indices


def _text_hash(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def civil_train_text_hashes(civil: pd.DataFrame, *, seed: int = 42) -> set[str]:
    df = civil.copy()
    df["text"] = normalize_text_column(df, "text")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    y = civil_harmful_binary(df)
    idx_train, _, _ = stratified_split_indices(y, seed=seed)
    hashes: set[str] = set()
    for i in idx_train:
        hashes.add(_text_hash(df["text"].iloc[i]))
    return hashes


def filter_non_civil_train(frame: pd.DataFrame, train_hashes: set[str], text_col: str = "text") -> pd.DataFrame:
    texts = frame[text_col].astype(str)
    keep = [_text_hash(t) not in train_hashes for t in texts]
    return frame.loc[keep].reset_index(drop=True)