"""Compare Jigsaw identity audit C_hybrid (cached metrics) vs live C_hybrid+LLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.pipeline import ModerationPipeline
from research.jigsaw_identity import run_jigsaw_identity_audit

METRICS = ROOT / "docs" / "eval_metrics.json"
SAMPLE = int(__import__("os").getenv("JIGSAW_AUDIT_SAMPLE", "2000"))


def _pick(audit: dict) -> dict:
    return {
        "overall_fpr": audit.get("overall_fpr"),
        "overall_fpr_ci95_low": audit.get("overall_fpr_ci95_low"),
        "overall_fpr_ci95_high": audit.get("overall_fpr_ci95_high"),
        "fpr_benign_identity_rows": audit.get("fpr_benign_identity_rows"),
        "fpr_benign_non_identity_rows": audit.get("fpr_benign_non_identity_rows"),
    }


def main() -> None:
    before = {}
    if METRICS.exists():
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        before = data.get("identity_audit_jigsaw", {}).get("C_hybrid", {})
    pipe = ModerationPipeline(use_transformer=True, use_llm=True)
    after, _, _ = run_jigsaw_identity_audit(pipe, sample=SAMPLE, seed=42, batch_size=32)
    print(
        json.dumps(
            {
                "sample_n": SAMPLE,
                "before_C_hybrid": _pick(before),
                "after_C_hybrid_with_llm": _pick(after),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()