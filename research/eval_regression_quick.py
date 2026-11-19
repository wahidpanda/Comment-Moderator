"""Fast regression metrics without long Jigsaw/sweep loops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.pipeline import ModerationPipeline
from research.civil_overlap import civil_train_text_hashes
from research.eval import prepare_civil_test_frame
from research.eval_bundle import run_pipeline_batch, subsample_test_stratified
from research.jigsaw_identity import run_jigsaw_identity_audit
from research.needs_human_sweep import scenario_needs_human_rate, sweep_needs_human
from research.spam_decisions import export_spam_layer_csv
from research.task_metrics import civil_spam_proxy, task_metrics
from research.threat_eval import run_threat_evals
from research.threat_metrics import civil_threat_mask
from research.train import load_raw


def main() -> None:
    before_path = Path("docs/eval_metrics_before_regression.json")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    civil, _ = load_raw(Path("data/raw"))
    train_hashes = civil_train_text_hashes(civil, seed=42)
    test_df, y_test, x_test, id_test = prepare_civil_test_frame(civil, sample=None, seed=42)
    y_test = np.asarray(y_test, dtype=np.int8)
    test_df, y_test, x_test, id_test = subsample_test_stratified(
        test_df, y_test, x_test, id_test, n=3000, seed=42
    )
    y_threat = civil_threat_mask(test_df)
    y_spam = civil_spam_proxy(test_df, x_test)

    pipes = {
        "A_rules_lr": ModerationPipeline(use_transformer=False),
        "C_hybrid": ModerationPipeline(use_transformer=True),
    }
    after: dict = {}
    decomposed: dict = {}
    for name, pipe in pipes.items():
        row = run_pipeline_batch(name, pipe, x_test, id_test, y_test)
        after[name] = {k: v for k, v in row.items() if k != "results"}
        decomposed[name] = task_metrics(y_test, y_spam, y_threat, row["results"], task_name=name)
        if name == "A_rules_lr":
            csv_path = Path("docs/spam_layer_decisions_civil3k.csv")
            stats = export_spam_layer_csv(row["results"], x_test, id_test, test_df, y_test, csv_path)
            stats["csv"] = str(csv_path)

    sweep_ids, sweep_texts = id_test[:500], x_test[:500]
    nh_sweep = {
        "A_rules_lr": sweep_needs_human(sweep_texts, sweep_ids, thresholds=[0.25, 0.35, 0.45, 0.55]),
        "C_hybrid": sweep_needs_human(
            sweep_texts, sweep_ids, thresholds=[0.25, 0.35, 0.45, 0.55], use_transformer=True
        ),
    }
    nh = {
        "A_rules_lr": scenario_needs_human_rate(x_test, id_test, use_transformer=False),
        "C_hybrid": scenario_needs_human_rate(x_test, id_test, use_transformer=True),
    }
    jigsaw_c, _, _ = run_jigsaw_identity_audit(
        pipes["C_hybrid"], sample=2000, seed=42, exclude_hashes=train_hashes
    )
    jigsaw_c["sample_n"] = 2000
    jigsaw_c["civil_train_hashes_excluded"] = len(train_hashes)

    b = next(r for r in before["results"] if r["scenario"] == "B_transformer_only")
    payload = {
        **before,
        "results": [after["A_rules_lr"], b, after["C_hybrid"]],
        "task_metrics_decomposed": decomposed,
        "spam_layer_review": stats,
        "needs_human_rates": nh,
        "needs_human_threshold_sweep": nh_sweep,
        "identity_audit_jigsaw": {"C_hybrid": jigsaw_c},
        "threat_eval_no_rules": run_threat_evals(),
        "scenario_B_contamination": {
            "contaminated": True,
            "note": "Detoxify unbiased likely saw overlapping public toxicity text; do not treat B as unbiased.",
        },
        "regression_fix": True,
    }
    out = Path("docs/eval_metrics.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k in (
        "results","spam_layer_review","needs_human_rates","task_metrics_decomposed"
    )}, indent=2))


if __name__ == "__main__":
    main()