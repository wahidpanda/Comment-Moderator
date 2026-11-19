"""Regression eval: decomposed tasks, spam CSV, needs_human sweep, threat holdout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput
from research.civil_overlap import civil_train_text_hashes
from research.eval import prepare_civil_test_frame
from research.eval_bundle import (
    CACHE_DIR,
    run_pipeline_batch,
    scenario_benign_fpr,
    subsample_test_stratified,
)
from research.jigsaw_identity import run_jigsaw_identity_audit
from research.needs_human_sweep import scenario_needs_human_rate, sweep_needs_human
from research.spam_decisions import export_spam_layer_csv
from research.task_metrics import civil_spam_proxy, task_metrics
from research.threat_eval import run_threat_evals
from research.threat_metrics import civil_threat_mask
from research.train import load_raw


def main() -> None:
    out_json = Path("docs/eval_metrics.json")
    before = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    before_path = Path("docs/eval_metrics_before_regression.json")
    if not before_path.exists():
        shutil.copy(out_json, before_path)

    civil, _ = load_raw(Path("data/raw"))
    assert civil is not None
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
    after_results: dict[str, dict] = {}
    decomposed: dict[str, dict] = {}
    spam_csv_stats = {}
    for name, pipe in pipes.items():
        row = run_pipeline_batch(name, pipe, x_test, id_test, y_test)
        after_results[name] = {k: v for k, v in row.items() if k != "results"}
        decomposed[name] = task_metrics(y_test, y_spam, y_threat, row["results"], task_name=name)
        if name == "A_rules_lr":
            csv_path = Path("docs/spam_layer_decisions_civil3k.csv")
            spam_csv_stats = export_spam_layer_csv(
                row["results"], x_test, id_test, test_df, y_test, csv_path
            )
            spam_csv_stats["csv"] = str(csv_path)

    nh_rates = {
        k: scenario_needs_human_rate(x_test, id_test, use_transformer=(k == "C"))
        for k in ("A_rules_lr", "C_hybrid")
    }
    sweep_texts = x_test[:800]
    sweep_ids = id_test[:800]
    nh_sweep = {
        "A_rules_lr": sweep_needs_human(sweep_texts, sweep_ids, use_transformer=False),
        "C_hybrid": sweep_needs_human(sweep_texts, sweep_ids, use_transformer=True),
        "sweep_note": "first 800 Civil 3k subsample for threshold sweep",
    }

    jigsaw = {}
    for name, pipe in pipes.items():
        audit, _, frame = run_jigsaw_identity_audit(
            pipe, sample=2500, seed=42, exclude_hashes=train_hashes
        )
        audit["sample_n"] = len(frame)
        audit["civil_train_rows_excluded"] = len(train_hashes)
        jigsaw[name] = audit

    threat_eval = run_threat_evals()

    # Refresh B from cache with Wilson fix only
    b_row = next((r for r in before.get("results", []) if r.get("scenario") == "B_transformer_only"), None)
    if b_row and "false_positives" in b_row:
        fp = int(b_row["false_positives"])
        bn = int((y_test == 0).sum())
        from research.stats import wilson_ci

        lo, hi = wilson_ci(fp, bn)
        b_row = {
            **b_row,
            "benign_count": bn,
            "benign_fpr": fp / max(bn, 1),
            "benign_fpr_ci95_low": lo,
            "benign_fpr_ci95_high": hi,
        }

    result_list = [after_results["A_rules_lr"], after_results["C_hybrid"]]
    if b_row:
        result_list.insert(1, b_row)

    payload = {
        **{k: v for k, v in before.items() if k not in ("results", "task_metrics_decomposed")},
        "regression_fix_date": "2026-11-17",
        "before_regression_snapshot": before_path.name,
        "results": result_list,
        "task_metrics_decomposed": decomposed,
        "spam_layer_review": spam_csv_stats,
        "needs_human_rates": nh_rates,
        "needs_human_threshold_sweep": nh_sweep,
        "identity_audit_jigsaw": jigsaw,
        "threat_eval_no_rules": threat_eval,
        "scenario_B_contamination": {
            "contaminated": True,
            "note": "Detoxify unbiased was trained on large public toxicity corpora that overlap Jigsaw/Civil Comments text; "
            "scenario B F1 is not a clean generalization estimate.",
        },
        "wilson_ci_note": "benign_fpr_ci computed as wilson_ci(false_positives, benign_count) per scenario",
    }
    # Drop empty B if missing
    payload["results"] = [r for r in payload["results"] if r and r.get("scenario")]
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()