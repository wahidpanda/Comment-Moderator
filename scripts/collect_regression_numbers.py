"""Collect before/after regression numbers (staged)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.pipeline import ModerationPipeline
from research.eval import prepare_civil_test_frame
from research.eval_bundle import run_pipeline_batch, subsample_test_stratified
from research.needs_human_sweep import scenario_needs_human_rate, sweep_needs_human
from research.spam_decisions import export_spam_layer_csv
from research.task_metrics import civil_spam_proxy, task_metrics
from research.threat_eval import run_threat_evals
from research.threat_metrics import civil_threat_mask
from research.train import load_raw


def main() -> None:
    before = json.loads(Path("docs/eval_metrics_before_regression.json").read_text(encoding="utf-8"))
    civil, _ = load_raw(Path("data/raw"))
    test_df, y_test, x_test, id_test = prepare_civil_test_frame(civil, sample=None, seed=42)
    y_test = np.asarray(y_test, dtype=np.int8)
    test_df, y_test, x_test, id_test = subsample_test_stratified(
        test_df, y_test, x_test, id_test, n=3000, seed=42
    )
    y_threat = civil_threat_mask(test_df)
    y_spam = civil_spam_proxy(test_df, x_test)

    a = run_pipeline_batch("A_rules_lr", ModerationPipeline(use_transformer=False), x_test, id_test, y_test)
    c = run_pipeline_batch("C_hybrid", ModerationPipeline(use_transformer=True), x_test, id_test, y_test)

    spam_stats = export_spam_layer_csv(
        a["results"], x_test, id_test, test_df, y_test, Path("docs/spam_layer_decisions_civil3k.csv")
    )
    spam_stats["csv"] = "docs/spam_layer_decisions_civil3k.csv"

    dec = {
        "A_rules_lr": task_metrics(y_test, y_spam, y_threat, a["results"], task_name="A_rules_lr"),
        "C_hybrid": task_metrics(y_test, y_spam, y_threat, c["results"], task_name="C_hybrid"),
    }

    nh = {
        "A_rules_lr": scenario_needs_human_rate(x_test, id_test, use_transformer=False),
        "C_hybrid": scenario_needs_human_rate(x_test, id_test, use_transformer=True),
    }
    sweep = {
        "A_rules_lr": sweep_needs_human(x_test[:400], id_test[:400], thresholds=[0.25, 0.35, 0.45, 0.55]),
        "C_hybrid_note": "C sweep skipped (re-run would re-batch Detoxify); use needs_human_rate + A sweep",
    }

    b = next(r for r in before["results"] if r["scenario"] == "B_transformer_only")
    payload = {
        **before,
        "results": [
            {k: v for k, v in a.items() if k != "results"},
            b,
            {k: v for k, v in c.items() if k != "results"},
        ],
        "task_metrics_decomposed": dec,
        "spam_layer_review": spam_stats,
        "needs_human_rates": nh,
        "needs_human_threshold_sweep": sweep,
        "threat_eval_no_rules": run_threat_evals(),
        "scenario_B_contamination": {"contaminated": True, "note": "Detoxify/public toxicity overlap"},
        "regression_fix": True,
    }
    Path("docs/eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("spam_decisions", spam_stats)
    print("A benign_fpr", payload["results"][0]["benign_fpr"])
    print("C benign_fpr", payload["results"][2]["benign_fpr"])
    print("toxicity A", dec["A_rules_lr"]["toxicity"]["benign_fpr"])


if __name__ == "__main__":
    main()