"""Full eval: Civil 3k B/C cache, identity Wilson CI, creator_eval per label."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import os
from datetime import UTC, datetime

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel
from engine.transformer_layer import TransformerLayer
from research.eval import HARMFUL_LABELS, prepare_civil_test_frame, pred_harmful_from_results
from research.jigsaw_identity import run_jigsaw_identity_audit
from research.threat_metrics import civil_threat_mask, threat_recall
from research.labels import MIN_TRAIN_ROWS, civil_harmful_binary, normalize_text_column, stratified_split_indices
from research.stats import bootstrap_f1_ci, latency_percentiles, wilson_ci
from research.train import load_raw

CACHE_DIR = Path("docs/cache")
CREATOR_EVAL = Path("research/data/creator_eval.csv")

HARMFUL_EXPECTED = {"toxic", "threat", "spam", "identity_attack"}


def scenario_benign_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    benign = y_true == 0
    n_benign = int(benign.sum())
    fp = int(((y_pred == 1) & benign).sum())
    fpr = float(fp / max(n_benign, 1))
    lo, hi = wilson_ci(fp, n_benign)
    return {
        "benign_count": n_benign,
        "false_positives": fp,
        "benign_fpr": fpr,
        "benign_fpr_ci95_low": lo,
        "benign_fpr_ci95_high": hi,
    }


def subsample_test_stratified(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    x_test: list[str],
    id_test: list[str],
    *,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    if len(y_test) <= n:
        return test_df, y_test, x_test, id_test
    idx = np.arange(len(y_test))
    rng = np.random.default_rng(seed)
    # stratified subsample
    chosen: list[int] = []
    for label in (0, 1):
        pool = idx[y_test == label]
        k = int(round(n * len(pool) / len(y_test)))
        k = min(k, len(pool))
        if k > 0:
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    while len(chosen) < n and len(chosen) < len(y_test):
        extra = int(rng.choice(idx))
        if extra not in chosen:
            chosen.append(extra)
    chosen = chosen[:n]
    sub = np.sort(np.array(chosen))
    return (
        test_df.iloc[sub].reset_index(drop=True),
        np.asarray(y_test[sub]),
        [x_test[i] for i in sub],
        [id_test[i] for i in sub],
    )


def run_transformer_only_batched(
    texts: list[str],
    ids: list[str],
    y_true: np.ndarray,
    *,
    scenario: str = "D_detoxify_only",
    n_boot: int = 500,
) -> dict[str, Any]:
    """Scenario D: every comment through Detoxify (batch inference)."""
    layer = TransformerLayer(enabled=True)
    latencies: list[float] = []
    results = []
    t0_total = time.perf_counter()
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        t0 = time.perf_counter()
        preds = layer.predict_batch(chunk, batch_size=batch_size)
        dt = (time.perf_counter() - t0) * 1000
        per = dt / max(len(chunk), 1)
        for j, tr in enumerate(preds):
            results.append(
                type(
                    "R",
                    (),
                    {
                        "label": tr["label"],
                        "decided_by": "transformer",
                        "needs_human": tr["label"] in (ModerationLabel.THREAT, ModerationLabel.IDENTITY_ATTACK),
                        "scores": tr.get("scores"),
                        "reasons": tr.get("reasons") or [],
                    },
                )()
            )
            latencies.append(per)
    y_pred_arr = pred_harmful_from_results(results)
    f1_stats = bootstrap_f1_ci(y_true, y_pred_arr, n_boot=n_boot)
    tp = int(((y_pred_arr == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true == 1)).sum())
    lat = latency_percentiles(latencies)
    elapsed = time.perf_counter() - t0_total
    fpr = scenario_benign_fpr(y_true, y_pred_arr)
    return {
        "scenario": scenario,
        "n": len(texts),
        "f1_macro_proxy": f1_stats["f1"],
        "f1_ci95_low": f1_stats["f1_ci95_low"],
        "f1_ci95_high": f1_stats["f1_ci95_high"],
        "precision": float(precision_score(y_true, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_arr, zero_division=0)),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "transformer_pct": 100.0,
        "transformer_calls": len(texts),
        "latency_ms_p50": lat["p50"],
        "latency_ms_p95": lat["p95"],
        "latency_ms_total": round(elapsed * 1000, 2),
        "results": results,
        **fpr,
    }


def run_pipeline_batch(
    name: str,
    pipeline: ModerationPipeline,
    texts: list[str],
    ids: list[str],
    y_true: np.ndarray,
    *,
    n_boot: int = 500,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = pipeline.analyze_batch(
        [CommentInput(id=cid, text=text) for cid, text in zip(ids, texts)]
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results = resp.results
    latencies = list(resp.summary.get("latency_ms_per_comment") or [])
    y_pred_arr = pred_harmful_from_results(results)
    f1_stats = bootstrap_f1_ci(y_true, y_pred_arr, n_boot=n_boot)
    tp = int(((y_pred_arr == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true == 1)).sum())
    n = len(texts)
    transformer_pct = 100.0 * pipeline.stats.get("transformer", 0) / max(n, 1)
    lat = latency_percentiles(latencies)
    fpr = scenario_benign_fpr(y_true, y_pred_arr)
    return {
        "scenario": name,
        "n": n,
        "f1_macro_proxy": f1_stats["f1"],
        "f1_ci95_low": f1_stats["f1_ci95_low"],
        "f1_ci95_high": f1_stats["f1_ci95_high"],
        "precision": float(precision_score(y_true, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_arr, zero_division=0)),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "transformer_pct": round(transformer_pct, 2),
        "transformer_calls": pipeline.stats.get("transformer", 0),
        "latency_ms_p50": lat["p50"],
        "latency_ms_p95": lat["p95"],
        "latency_ms_total": round(elapsed_ms, 2),
        "layer_counts": dict(pipeline.stats),
        "results": results,
        **fpr,
    }


def scenario_threat_slices(
    *,
    civil_test: pd.DataFrame,
    y_civil_threat: np.ndarray,
    civil_results: list[Any],
    creator_rows: list[dict[str, str]],
    creator_results: list[Any],
    scenario: str,
) -> dict[str, Any]:
    civil_rec = threat_recall([(y_civil_threat, civil_results)])
    threat_idx = [i for i, r in enumerate(creator_rows) if r.get("label_expected") == "threat"]
    if threat_idx and creator_results:
        y_cr = np.ones(len(threat_idx), dtype=np.int8)
        cr_results = [creator_results[i] for i in threat_idx]
        creator_rec = threat_recall([(y_cr, cr_results)])
    else:
        creator_rec = {"n": 0, "recall": None, "true_threats": 0, "detected": 0}
    return {
        "scenario": scenario,
        "civil_threat_ge_0.5": civil_rec,
        "creator_eval_threat": creator_rec,
    }


def eval_creator_row(expected: str, predicted: str) -> dict[str, bool]:
    exp_h = expected in HARMFUL_EXPECTED
    pred_h = predicted in HARMFUL_LABELS
    return {
        "label_match": expected == predicted,
        "harmful_match": exp_h == pred_h,
    }


def run_creator_eval(pipeline: ModerationPipeline, scenario: str) -> dict[str, Any]:
    if not CREATOR_EVAL.exists():
        return {"scenario": scenario, "skipped": "no creator_eval.csv"}
    rows = list(csv.DictReader(CREATOR_EVAL.open(encoding="utf-8")))
    out = pipeline.analyze_batch(
        [CommentInput(id=row["id"], text=row["text"]) for row in rows]
    )
    per_label: dict[str, dict[str, int]] = {}
    for row, r in zip(rows, out.results):
        pred = r.label.value
        exp = row["label_expected"]
        bucket = per_label.setdefault(exp, {"n": 0, "label_match": 0, "harmful_match": 0})
        bucket["n"] += 1
        m = eval_creator_row(exp, pred)
        bucket["label_match"] += int(m["label_match"])
        bucket["harmful_match"] += int(m["harmful_match"])
    summary = {
        k: {
            "n": v["n"],
            "label_accuracy": v["label_match"] / max(v["n"], 1),
            "harmful_accuracy": v["harmful_match"] / max(v["n"], 1),
        }
        for k, v in per_label.items()
    }
    return {"scenario": scenario, "per_label": summary, "total": len(rows)}


def run_creator_transformer_only(scenario: str) -> dict[str, Any]:
    if not CREATOR_EVAL.exists():
        return {"scenario": scenario, "skipped": "no creator_eval.csv"}
    rows = list(csv.DictReader(CREATOR_EVAL.open(encoding="utf-8")))
    layer = TransformerLayer(enabled=True)
    texts = [row["text"] for row in rows]
    preds = layer.predict_batch(texts, batch_size=32)
    per_label: dict[str, dict[str, int]] = {}
    for row, tr in zip(rows, preds):
        pred = tr["label"].value
        exp = row["label_expected"]
        bucket = per_label.setdefault(exp, {"n": 0, "label_match": 0, "harmful_match": 0})
        bucket["n"] += 1
        m = eval_creator_row(exp, pred)
        bucket["label_match"] += int(m["label_match"])
        bucket["harmful_match"] += int(m["harmful_match"])
    summary = {
        k: {
            "n": v["n"],
            "label_accuracy": v["label_match"] / max(v["n"], 1),
            "harmful_accuracy": v["harmful_match"] / max(v["n"], 1),
        }
        for k, v in per_label.items()
    }
    return {"scenario": scenario, "per_label": summary, "total": len(rows)}


def cache_key(model_path: Path, n: int, seed: int) -> str:
    h = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    return f"transformer_{n}_{seed}_{h}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/lr/v1"))
    parser.add_argument("--test-subset", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("docs/eval_metrics.json"))
    parser.add_argument("--before", type=Path, default=Path("docs/eval_metrics_before.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    civil, _ = load_raw(args.data_dir)
    if civil is None:
        raise SystemExit("Civil parquet required")
    n_rows = len(civil)
    if n_rows < MIN_TRAIN_ROWS:
        print(f"WARN: only {n_rows} civil rows", file=sys.stderr)

    test_df, y_test, x_test, id_test = prepare_civil_test_frame(civil, sample=None, seed=args.seed)
    y_test = np.asarray(y_test, dtype=np.int8)
    test_df, y_test, x_test, id_test = subsample_test_stratified(
        test_df, y_test, x_test, id_test, n=args.test_subset, seed=args.seed
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "model.joblib"
    ck = cache_key(model_path, args.test_subset, args.seed)
    cache_path = CACHE_DIR / f"{ck}.json"

    detoxify_ok = True
    try:
        TransformerLayer(enabled=True).predict("probe")
    except RuntimeError:
        detoxify_ok = False

    def _strip_results(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k != "results"}

    scenarios_b_c: list[dict[str, Any]] = []
    scenario_c_hybrid: dict[str, Any] | None = None
    if cache_path.exists() and not args.force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        scenarios_b_c = cached.get("scenarios_b_c", [])
    elif detoxify_ok:
        hybrid = ModerationPipeline(use_transformer=True)
        scenario_c_hybrid = run_pipeline_batch("C_hybrid", hybrid, x_test, id_test, y_test)
        scenarios_b_c = [
            run_transformer_only_batched(x_test, id_test, y_test),
            _strip_results(scenario_c_hybrid),
        ]
        cache_path.write_text(
            json.dumps({"scenarios_b_c": scenarios_b_c, "n_test_subset": len(x_test)}, indent=2),
            encoding="utf-8",
        )
    else:
        scenarios_b_c = [{"scenario": "B/C", "skipped": "detoxify not installed"}]

    rules_lr = ModerationPipeline(use_transformer=False)
    if scenario_c_hybrid is None and detoxify_ok:
        scenario_c_hybrid = run_pipeline_batch(
            "C_hybrid", ModerationPipeline(use_transformer=True), x_test, id_test, y_test
        )

    creator_rows = (
        list(csv.DictReader(CREATOR_EVAL.open(encoding="utf-8"))) if CREATOR_EVAL.exists() else []
    )
    y_civil_threat = civil_threat_mask(test_df)
    scenario_a_full = run_pipeline_batch("A_rules_lr", rules_lr, x_test, id_test, y_test)
    scenario_a = _strip_results(scenario_a_full)

    identity_audit: dict[str, Any] = {}
    threat_metrics: list[dict[str, Any]] = []
    use_llm_audit = os.getenv("USE_LLM", "false").lower() == "true"
    audit_specs: list[tuple[str, ModerationPipeline, dict | None]] = [
        ("A_rules_lr", rules_lr, scenario_a_full),
        ("C_hybrid", ModerationPipeline(use_transformer=detoxify_ok, use_llm=False), scenario_c_hybrid),
    ]
    if use_llm_audit and detoxify_ok:
        audit_specs.append(
            (
                "C_hybrid_with_llm",
                ModerationPipeline(use_transformer=detoxify_ok, use_llm=True),
                scenario_c_hybrid,
            )
        )
    for label, pipe, full in audit_specs:
        if full is None or "results" not in full:
            continue
        audit, _, _ = run_jigsaw_identity_audit(pipe, sample=4000, seed=args.seed)
        identity_audit[label] = audit
        threat_metrics.append(
            scenario_threat_slices(
                civil_test=test_df,
                y_civil_threat=y_civil_threat,
                civil_results=full["results"],
                creator_rows=creator_rows,
                creator_results=(
                    pipe.analyze_batch(
                        [CommentInput(id=r["id"], text=r["text"]) for r in creator_rows]
                    ).results
                    if creator_rows
                    else []
                ),
                scenario=label,
            )
        )

    creator = {
        "A_rules_lr": run_creator_eval(ModerationPipeline(use_transformer=False), "A_rules_lr"),
    }
    if detoxify_ok:
        creator["C_hybrid"] = run_creator_eval(
            ModerationPipeline(use_transformer=True), "C_hybrid"
        )
        creator["B_transformer_only"] = run_creator_transformer_only("B_transformer_only")

    before_snap: dict[str, Any] = {}
    if args.before.exists():
        before_snap = json.loads(args.before.read_text(encoding="utf-8"))

    thresholds = {}
    th_path = args.model_dir / "thresholds.json"
    if th_path.exists():
        thresholds = json.loads(th_path.read_text(encoding="utf-8"))
    metrics_train = {}
    mt = args.model_dir / "metrics.json"
    if mt.exists():
        metrics_train = json.loads(mt.read_text(encoding="utf-8"))
    spam_metrics = {}
    sm = Path("models/spam/v1/metrics.json")
    if sm.exists():
        spam_metrics = json.loads(sm.read_text(encoding="utf-8"))

    benign_n = int((y_test == 0).sum())
    for row in scenarios_b_c:
        if "false_positives" in row:
            fp = int(row["false_positives"])
            lo, hi = wilson_ci(fp, benign_n)
            row["benign_count"] = benign_n
            row["benign_fpr"] = fp / max(benign_n, 1)
            row["benign_fpr_ci95_low"] = lo
            row["benign_fpr_ci95_high"] = hi

    from engine.llm.prompt import BUCKET_ONE_LINERS

    payload = {
        "eval_generated_at": datetime.now(UTC).isoformat(),
        "n_civil_rows_training": n_rows,
        "n_test_subset": len(x_test),
        "test_subset_note": f"stratified {args.test_subset} from Civil test split",
        "results": [scenario_a, *scenarios_b_c],
        "identity_audit_jigsaw": identity_audit,
        "bucket_definitions": BUCKET_ONE_LINERS,
        "threat_recall": threat_metrics,
        "spam_classifier": spam_metrics,
        "creator_eval": creator,
        "cache_file": str(cache_path),
        "before_snapshot_keys": list(before_snap.keys()) if before_snap else [],
        "model_metrics": metrics_train,
        "thresholds": thresholds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()