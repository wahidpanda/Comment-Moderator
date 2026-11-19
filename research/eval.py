"""Evaluate rules+LR, transformer-only, and hybrid on held-out Civil Comments test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel
from engine.transformer_layer import TransformerLayer
from research.labels import (
    BINARY_THRESHOLD,
    CIVIL_IDENTITY_GROUP_COLS,
    MIN_TRAIN_ROWS,
    civil_harmful_binary,
    identity_mention_term_masks,
    normalize_text_column,
    stratified_split_indices,
)
from research.train import load_raw

HARMFUL_LABELS = {
    ModerationLabel.TOXIC.value,
    ModerationLabel.THREAT.value,
    ModerationLabel.IDENTITY_ATTACK.value,
    ModerationLabel.SPAM.value,
}


from research.stats import bootstrap_f1_ci, wilson_ci


def pred_harmful_from_results(results: list[Any]) -> np.ndarray:
    out = []
    for r in results:
        out.append(1 if r.label.value in HARMFUL_LABELS else 0)
    return np.array(out, dtype=np.int8)


def run_scenario(
    name: str,
    pipeline: ModerationPipeline,
    texts: list[str],
    ids: list[str],
    y_true: np.ndarray,
) -> dict[str, Any]:
    comments = [CommentInput(id=i, text=t) for i, t in zip(ids, texts)]
    t0 = time.perf_counter()
    resp = pipeline.analyze_batch(comments)
    elapsed = time.perf_counter() - t0
    y_pred_arr = pred_harmful_from_results(resp.results)
    tp = int(((y_pred_arr == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true == 1)).sum())
    f1_stats = bootstrap_f1_ci(y_true, y_pred_arr)
    benign = y_true == 0
    overall_fpr = float(fp / max(int(benign.sum()), 1))

    return {
        "scenario": name,
        "f1_macro_proxy": f1_stats["f1"],
        "f1_ci95_low": f1_stats["f1_ci95_low"],
        "f1_ci95_high": f1_stats["f1_ci95_high"],
        "precision": float(precision_score(y_true, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_arr, zero_division=0)),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "overall_fpr_benign": overall_fpr,
        "latency_ms_total": round(elapsed * 1000, 2),
        "latency_ms_p50": round((elapsed / max(len(texts), 1)) * 1000, 3),
        "transformer_calls": pipeline.stats.get("transformer", 0),
        "layer_counts": pipeline.stats,
        "cpu_per_10k_comments_sec": round((elapsed / max(len(texts), 1)) * 10000, 2),
    }


def identity_audit_hybrid(
    civil_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """FPR on benign rows, overall vs per Civil identity subgroup column."""
    benign = y_true == 0
    fp_mask = (y_pred == 1) & benign
    overall_fpr = float(fp_mask.sum() / max(int(benign.sum()), 1))

    groups: dict[str, Any] = {}
    subgroup_cols = [c for c in CIVIL_IDENTITY_GROUP_COLS if c in civil_test.columns]
    if subgroup_cols:
        for col in subgroup_cols:
            scores = civil_test[col].astype(float).to_numpy()
            slice_mask = benign & (scores > 0)
            n_slice = int(slice_mask.sum())
            if n_slice == 0:
                groups[col] = {"benign_rows": 0, "false_positives": 0, "fpr": None}
                continue
            fps = int(((y_pred == 1) & slice_mask).sum())
            groups[col] = {
                "benign_rows": n_slice,
                "false_positives": fps,
                "fpr": float(fps / n_slice),
            }
    else:
        term_masks = identity_mention_term_masks(civil_test)
        for term, mask in term_masks.items():
            slice_mask = benign & mask
            n_slice = int(slice_mask.sum())
            if n_slice == 0:
                groups[term] = {"benign_rows": 0, "false_positives": 0, "fpr": None}
                continue
            fps = int(((y_pred == 1) & slice_mask).sum())
            lo, hi = wilson_ci(fps, n_slice)
            groups[term] = {
                "benign_rows": n_slice,
                "false_positives": fps,
                "fpr": float(fps / n_slice),
                "fpr_ci95_low": lo,
                "fpr_ci95_high": hi,
            }

    id_attack = None
    if "identity_attack" in civil_test.columns:
        id_scores = civil_test["identity_attack"].astype(float).to_numpy()
        slice_mask = benign & (id_scores > 0)
        n_slice = int(slice_mask.sum())
        if n_slice:
            fps = int(((y_pred == 1) & slice_mask).sum())
            id_attack = {
                "benign_rows": n_slice,
                "false_positives": fps,
                "fpr": float(fps / n_slice),
                "threshold_note": f"identity_attack>{0} among benign",
            }

    return {
        "overall_benign_rows": int(benign.sum()),
        "overall_false_positives": int(fp_mask.sum()),
        "overall_fpr": overall_fpr,
        "identity_attack_slice": id_attack,
        "per_identity_group": groups,
        "subgroup_columns_present": bool(subgroup_cols),
        "annotation_threshold": BINARY_THRESHOLD,
    }


def prepare_civil_test_frame(
    civil: pd.DataFrame,
    *,
    sample: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    df = civil.copy()
    df["text"] = normalize_text_column(df, "text")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    y = civil_harmful_binary(df)
    texts = df["text"].tolist()
    _, _, idx_test = stratified_split_indices(y, seed=seed)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    y_test = y[idx_test]
    x_test = [texts[i] for i in idx_test]
    id_test = [str(i) for i in idx_test]
    return test_df, y_test, x_test, id_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/eval_metrics.json"))
    parser.add_argument("--sample", type=int, default=None, help="Optional random subsample of Civil")
    parser.add_argument("--smoke", action="store_true", help="Allow Civil frame < MIN_TRAIN_ROWS")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    civil, _jigsaw = load_raw(args.data_dir)
    if civil is None:
        print("ERROR: Civil Comments parquet required for eval identity audit.", file=sys.stderr)
        raise SystemExit(2)

    civil_work = civil.copy()
    civil_work["text"] = normalize_text_column(civil_work, "text")
    civil_work = civil_work[civil_work["text"].str.len() > 0]
    n_rows = len(civil_work)
    if n_rows < MIN_TRAIN_ROWS and not args.smoke:
        print(
            f"ERROR: Civil frame has {n_rows} rows (< {MIN_TRAIN_ROWS}). "
            "Run `make download` or pass --smoke.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    test_df, y_test, x_test, id_test = prepare_civil_test_frame(civil, sample=args.sample, seed=args.seed)
    y_test = np.asarray(y_test, dtype=np.int8)

    hybrid = ModerationPipeline(use_transformer=True)
    rules_lr = ModerationPipeline(use_transformer=False)
    transformer_only = ModerationPipeline(use_transformer=True)
    transformer_only.rules.evaluate = lambda *a, **k: None  # type: ignore[method-assign]
    transformer_only.lr.classify_band = lambda t: ("ambiguous", 0.5)  # type: ignore[method-assign]

    results = [
        run_scenario("A_rules_lr", rules_lr, x_test, id_test, y_test),
        run_scenario("C_hybrid", hybrid, x_test, id_test, y_test),
    ]
    try:
        TransformerLayer(enabled=True).predict("eval probe")
        results.append(run_scenario("B_transformer_only", transformer_only, x_test, id_test, y_test))
    except RuntimeError:
        results.append({"scenario": "B_transformer_only", "skipped": "detoxify not installed"})

    hybrid_resp = hybrid.analyze_batch([CommentInput(id=i, text=t) for i, t in zip(id_test, x_test)])
    y_pred_hybrid = pred_harmful_from_results(hybrid_resp.results)
    identity_audit = identity_audit_hybrid(test_df, y_test, y_pred_hybrid)

    payload = {
        "results": results,
        "identity_audit": identity_audit,
        "n_test": len(x_test),
        "n_civil_rows_used": n_rows if args.sample is None else args.sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()