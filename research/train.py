"""Train logistic regression (and compare SVM/SGD) on Civil + Jigsaw unified corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from engine.identity import mentions_identity_term
from research.features import attach_classifier, make_tfidf_pipeline
from research.identity_augment import IDENTITY_SAFE_AUGMENTS, append_identity_augment
from research.labels import (
    BINARY_THRESHOLD,
    LR_HEAD_COLS,
    MIN_TRAIN_ROWS,
    build_unified_frame,
    civil_harmful_binary,
    normalize_text_column,
    stratified_split_indices,
)


def load_raw(data_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    civil_path = data_dir / "civil_comments.parquet"
    jigsaw_path = data_dir / "jigsaw_toxic_comments.parquet"
    civil = pd.read_parquet(civil_path) if civil_path.exists() else None
    jigsaw = pd.read_parquet(jigsaw_path) if jigsaw_path.exists() else None
    if civil is None and jigsaw is None:
        raise FileNotFoundError(
            f"No data in {data_dir}. Run: make download (or provide parquet files)."
        )
    return civil, jigsaw


def build_sample_weights(
    texts: list[str],
    y: np.ndarray,
    sources: list[str],
) -> np.ndarray:
    from research.identity_augment import AUGMENT_ROW_WEIGHT, AUGMENT_SOURCE, IDENTITY_BENIGN_WEIGHT

    w = np.ones(len(texts), dtype=float)
    for i, (text, src) in enumerate(zip(texts, sources)):
        if src == AUGMENT_SOURCE:
            w[i] = AUGMENT_ROW_WEIGHT
        elif int(y[i]) == 0 and mentions_identity_term(text):
            w[i] = IDENTITY_BENIGN_WEIGHT
    return w


def tune_on_validation(
    x_train: list[str],
    y_train: np.ndarray,
    x_val: list[str],
    y_val: np.ndarray,
    *,
    fast: bool = False,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    best_score = -1.0
    best_artifact: Any = None
    best_params: dict[str, Any] = {}

    if fast or len(x_train) > 10_000:
        cs = [1.0]
        class_weights: list[dict | None] = ["balanced"]
        word_ranges = [(1, 2)]
    else:
        cs = [0.25, 1.0, 4.0]
        class_weights = [None, "balanced"]
        word_ranges = [(1, 1), (1, 2)]

    for c in cs:
        for cw in class_weights:
            for w_range in word_ranges:
                vec = make_tfidf_pipeline(word_ngram_range=w_range)
                candidates = [
                    (
                        "lr",
                        LogisticRegression(
                            C=c,
                            class_weight=cw,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                    (
                        "linearsvc",
                        LinearSVC(C=c, class_weight=cw, max_iter=4000, random_state=42),
                    ),
                    (
                        "sgd",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1.0 / (c * max(len(x_train), 1)),
                            class_weight=cw,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
                for name, clf in candidates:
                    pipe = attach_classifier(vec, clf)
                    if name == "linearsvc":
                        calibrated = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
                        calibrated.fit(x_train, y_train)
                        proba = calibrated.predict_proba(x_val)[:, 1]
                        model = calibrated
                    else:
                        fit_kw = (
                            {"clf__sample_weight": sample_weight}
                            if sample_weight is not None
                            else {}
                        )
                        pipe.fit(x_train, y_train, **fit_kw)
                        proba = pipe.predict_proba(x_val)[:, 1]
                        model = pipe
                    pred = (proba >= 0.5).astype(int)
                    f1 = f1_score(y_val, pred, zero_division=0)
                    if f1 > best_score:
                        best_score = f1
                        best_artifact = model
                        best_params = {
                            "estimator": name,
                            "C": c,
                            "class_weight": cw,
                            "word_ngram_range": list(w_range),
                            "val_f1": f1,
                        }
    return best_artifact, best_params


def train_civil_heads(
    civil: pd.DataFrame,
    *,
    seed: int = 42,
    max_train_rows: int = 20_000,
) -> dict[str, Any]:
    """Per-dimension LR heads on Civil Comments only."""
    texts = normalize_text_column(civil, "text").tolist()
    y_harm = civil_harmful_binary(civil)
    idx_train, _, _ = stratified_split_indices(y_harm, seed=seed)
    if len(idx_train) > max_train_rows:
        rng = np.random.default_rng(seed)
        idx_train = np.sort(rng.choice(idx_train, size=max_train_rows, replace=False))
    heads: dict[str, Any] = {}
    for col in LR_HEAD_COLS:
        if col not in civil.columns:
            continue
        y = (civil[col].astype(float) >= BINARY_THRESHOLD).astype(np.int8).to_numpy()
        if y[idx_train].sum() == 0:
            continue
        x_tr = [texts[i] for i in idx_train]
        y_tr = y[idx_train]
        vec = make_tfidf_pipeline(word_ngram_range=(1, 2))
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        pipe = attach_classifier(vec, clf)
        pipe.fit(x_tr, y_tr)
        heads[col] = pipe
    return heads


def evaluate(model: Any, texts: list[str], y: np.ndarray) -> dict[str, Any]:
    proba = model.predict_proba(texts)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "report": classification_report(y, pred, zero_division=0),
    }


def sweep_thresholds(
    model: Any,
    x_val: list[str],
    y_val: np.ndarray,
) -> dict[str, float]:
    proba = model.predict_proba(x_val)[:, 1]
    best: dict[str, float] = {"t_low": 0.35, "t_high": 0.65, "val_macro_f1_proxy": 0.0}
    for t_low in np.linspace(0.15, 0.45, 7):
        for t_high in np.linspace(0.55, 0.85, 7):
            if t_high <= t_low + 0.05:
                continue
            pred = np.zeros_like(y_val)
            pred[proba >= t_high] = 1
            pred[proba <= t_low] = 0
            mid = (proba > t_low) & (proba < t_high)
            mid_frac = float(mid.mean())
            if mid_frac > 0.30:
                continue
            pred[mid] = (proba[mid] >= 0.5).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 >= best["val_macro_f1_proxy"]:
                best = {
                    "t_low": float(t_low),
                    "t_high": float(t_high),
                    "val_macro_f1_proxy": float(f1),
                    "ambiguous_fraction": mid_frac,
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("models/lr/v1"))
    parser.add_argument("--sample", type=int, default=None, help="Random subsample of unified frame")
    parser.add_argument("--skip-test", action="store_true", help="Do not touch test set")
    parser.add_argument("--smoke", action="store_true", help="Allow datasets smaller than MIN_TRAIN_ROWS")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    civil, jigsaw = load_raw(args.data_dir)
    frame = build_unified_frame(civil, jigsaw)
    frame = append_identity_augment(frame)
    if args.sample and len(frame) > args.sample:
        frame = frame.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    n_rows = len(frame)
    if n_rows < MIN_TRAIN_ROWS and not args.smoke:
        print(
            f"ERROR: unified training frame has {n_rows} rows (< {MIN_TRAIN_ROWS}). "
            "Run `make download` (≥50k Civil rows) or pass --smoke for local smoke training.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    y = frame["harmful"].to_numpy(dtype=np.int8)
    texts = frame["text"].tolist()
    sources = frame["source"].tolist()
    idx_train, idx_val, idx_test = stratified_split_indices(y, seed=args.seed)

    x_train = [texts[i] for i in idx_train]
    y_train = y[idx_train]
    src_train = [sources[i] for i in idx_train]
    x_val = [texts[i] for i in idx_val]
    y_val = y[idx_val]
    sw_train = build_sample_weights(x_train, y_train, src_train)

    harmful_model, tune_params = tune_on_validation(
        x_train, y_train, x_val, y_val, fast=n_rows >= 10_000, sample_weight=sw_train
    )
    thresholds = sweep_thresholds(harmful_model, x_val, y_val)
    val_metrics = evaluate(harmful_model, x_val, y_val)

    heads: dict[str, Any] = {}
    if civil is not None and len(civil):
        civil_clean = civil.copy()
        civil_clean["text"] = normalize_text_column(civil_clean, "text")
        civil_clean = civil_clean[civil_clean["text"].str.len() > 0].reset_index(drop=True)
        heads = train_civil_heads(civil_clean, seed=args.seed)

    bundle = {"version": 2, "harmful": harmful_model, "heads": heads}

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out / "model.joblib")

    metrics: dict[str, Any] = {
        "tune": tune_params,
        "thresholds": thresholds,
        "validation": val_metrics,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_rows": n_rows,
        "harmful_rate_train": float(y_train.mean()),
        "heads": list(heads.keys()),
        "identity_augment_rows": len(IDENTITY_SAFE_AUGMENTS),
        "train_seconds": time.perf_counter() - t0,
    }

    if not args.skip_test:
        x_test = [texts[i] for i in idx_test]
        y_test = y[idx_test]
        metrics["test"] = evaluate(harmful_model, x_test, y_test)
        metrics["n_test"] = len(x_test)

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("tune", "thresholds", "validation", "heads")}, indent=2))
    if "test" in metrics:
        print("Test F1:", metrics["test"]["f1"])


if __name__ == "__main__":
    main()