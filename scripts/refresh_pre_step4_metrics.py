"""Refresh A, D (Detoxify-only), and C hybrid (no LLM) on Civil 3k; update eval_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.pipeline import ModerationPipeline
from engine.schemas import ModerationLabel
from research.civil_overlap import civil_train_text_hashes
from research.eval import prepare_civil_test_frame
from research.eval_bundle import run_pipeline_batch, run_transformer_only_batched, subsample_test_stratified
from research.jigsaw_identity import run_jigsaw_identity_audit
from research.needs_human_sweep import scenario_needs_human_rate, sweep_needs_human
from research.spam_decisions import export_spam_layer_csv
from research.task_metrics import civil_spam_proxy, task_metrics
from research.threat_eval import run_threat_evals
from research.threat_metrics import civil_threat_mask
from research.train import load_raw


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "results"}


def _report_task(name: str, dec: dict, nh: dict, row: dict) -> None:
    tox = dec["toxicity"]
    print(f"\n=== {name} ===")
    print(
        f"  toxicity F1={tox['f1']:.4f} FP(benign)={tox['false_positives_benign']} "
        f"macro_FN={row['false_negatives']}"
    )
    if "spam" in dec:
        print(f"  spam F1={dec['spam']['f1']:.4f} spam_decisions={dec['spam'].get('n_spam_decisions')}")
    if "threat" in dec:
        thr = dec["threat"]
        pos = int(thr.get("positives", 0))
        det = int(thr.get("detected", 0))
        rec = det / max(pos, 1)
        print(f"  threat recall={rec:.4f} detected={det}/{pos}")
    print(f"  needs_human rate={nh['needs_human_rate']:.4f} macro benign_fpr={row['benign_fpr']:.4f}")


def _d_needs_human_rate(d_results: list) -> dict:
    nh = sum(1 for r in d_results if getattr(r, "needs_human", False))
    n = len(d_results)
    return {"needs_human_count": nh, "needs_human_rate": nh / max(n, 1), "n": n}


def main() -> None:
    import sys

    skip_jigsaw = "--no-jigsaw" in sys.argv
    skip_threat = "--no-threat" in sys.argv
    out_path = Path("docs/eval_metrics.json")
    prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    civil, _ = load_raw(Path("data/raw"))
    train_hashes = civil_train_text_hashes(civil, seed=42)
    test_df, y_test, x_test, id_test = prepare_civil_test_frame(civil, sample=None, seed=42)
    y_test = np.asarray(y_test, dtype=np.int8)
    test_df, y_test, x_test, id_test = subsample_test_stratified(
        test_df, y_test, x_test, id_test, n=3000, seed=42
    )
    y_threat = civil_threat_mask(test_df)
    y_spam = civil_spam_proxy(test_df, x_test)

    a_full = run_pipeline_batch("A_rules_lr", ModerationPipeline(use_transformer=False), x_test, id_test, y_test)
    d_full = run_transformer_only_batched(x_test, id_test, y_test, scenario="D_detoxify_only")
    c_full = run_pipeline_batch(
        "C_hybrid",
        ModerationPipeline(use_transformer=True, use_llm=False),
        x_test,
        id_test,
        y_test,
    )

    csv_path = Path("docs/spam_layer_decisions_civil3k.csv")
    spam_stats = export_spam_layer_csv(a_full["results"], x_test, id_test, test_df, y_test, csv_path)
    spam_stats["csv"] = str(csv_path)

    dec = {
        "A_rules_lr": task_metrics(y_test, y_spam, y_threat, a_full["results"], task_name="A_rules_lr"),
        "D_detoxify_only": task_metrics(y_test, y_spam, y_threat, d_full["results"], task_name="D_detoxify_only"),
        "C_hybrid": task_metrics(y_test, y_spam, y_threat, c_full["results"], task_name="C_hybrid"),
    }

    nh = {
        "A_rules_lr": scenario_needs_human_rate(x_test, id_test, use_transformer=False),
        "D_detoxify_only": _d_needs_human_rate(d_full["results"]),
        "C_hybrid": scenario_needs_human_rate(x_test, id_test, use_transformer=True),
    }

    jigsaw: dict = dict(prev.get("identity_audit_jigsaw", {}))
    if skip_jigsaw:
        jigsaw["_note"] = "unchanged (--no-jigsaw)"
    else:
        for label, pipe in (
            ("A_rules_lr", ModerationPipeline(use_transformer=False)),
            ("C_hybrid", ModerationPipeline(use_transformer=True, use_llm=False)),
        ):
            audit, _, _ = run_jigsaw_identity_audit(pipe, sample=2000, seed=42, exclude_hashes=train_hashes)
            audit["sample_n"] = 2000
            audit["civil_train_hashes_excluded"] = len(train_hashes)
            jigsaw[label] = audit
        jigsaw["D_detoxify_only"] = {"skipped": "Jigsaw audit uses full pipeline; D is Detoxify-only batch"}

    sweep = {
        "A_rules_lr": sweep_needs_human(x_test[:400], id_test[:400], thresholds=[0.25, 0.35, 0.45, 0.55]),
        "C_hybrid": sweep_needs_human(
            x_test[:400], id_test[:400], thresholds=[0.25, 0.35, 0.45, 0.55], use_transformer=True
        ),
    }

    payload = {
        **prev,
        "n_test_subset": 3000,
        "test_subset_note": "stratified 3000 from Civil test split; refreshed pre-step4",
        "results": [_strip(a_full), _strip(d_full), _strip(c_full)],
        "task_metrics_decomposed": dec,
        "spam_layer_review": spam_stats,
        "needs_human_rates": nh,
        "needs_human_threshold_sweep": sweep,
        "identity_audit_jigsaw": jigsaw,
        "threat_eval_no_rules": prev.get("threat_eval_no_rules") if skip_threat else run_threat_evals(),
        "scenario_naming": {
            "A": "rules_lr",
            "B": "llm_only",
            "C": "hybrid_llm_detoxify",
            "D": "detoxify_only",
        },
        "stale_BC_replaced_pre_step4": True,
    }
    payload.pop("scenario_B_contamination", None)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, row in zip(("A_rules_lr", "D_detoxify_only", "C_hybrid"), payload["results"]):
        _report_task(name, dec[name], nh[name], row)
    print("\nWrote", out_path)


if __name__ == "__main__":
    main()