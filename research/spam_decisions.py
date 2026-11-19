"""Export spam-layer decisions for manual review."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput
from engine.spam_cues import spam_cue_reasons
from research.labels import civil_harmful_binary


def export_spam_layer_csv(
    results: list[Any],
    texts: list[str],
    ids: list[str],
    test_df: Any,
    y_harmful: np.ndarray,
    out_path: Path,
) -> dict[str, int]:
    rows: list[dict[str, str]] = []
    y_spam_proxy = None
    from research.task_metrics import civil_spam_proxy

    y_spam_proxy = civil_spam_proxy(test_df, texts)
    for i, r in enumerate(results):
        if r.decided_by != "spam":
            continue
        cues = "|".join(spam_cue_reasons(texts[i]))
        rows.append(
            {
                "id": ids[i],
                "text": texts[i],
                "spam_prob": f"{r.scores.spam:.3f}",
                "civil_harmful_gold": str(int(y_harmful[i])),
                "spam_cue_proxy_gold": str(int(y_spam_proxy[i])),
                "reasons": "; ".join(r.reasons),
                "cues_detected": cues,
                "manual_real_spam": "",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    proxy_pos = sum(int(r["spam_cue_proxy_gold"]) for r in rows)
    benign_fp = sum(1 for r in rows if r["civil_harmful_gold"] == "0")
    return {
        "spam_layer_decisions": len(rows),
        "proxy_real_spam_among_decisions": proxy_pos,
        "benign_civil_among_decisions": benign_fp,
    }