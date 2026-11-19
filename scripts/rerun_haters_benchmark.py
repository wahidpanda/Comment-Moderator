"""Re-run moderation on cached 137-comment haters job from SQLite."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.creator_buckets import apply_creator_buckets
from engine.pipeline import pipeline_from_env
from engine.schemas import CommentInput

JOB_ID = "5e176bfa-c99d-429f-9d3e-f78abf928672"
BASELINE_NEEDS_HUMAN = 7


def main() -> None:
    db = ROOT / "data" / "moderator.db"
    row = sqlite3.connect(db).execute(
        "SELECT result_json FROM jobs WHERE id=?", (JOB_ID,)
    ).fetchone()
    if not row:
        raise SystemExit(f"job {JOB_ID} not found")
    cached = json.loads(row[0])
    comments = [
        CommentInput(id=r["id"], text=r["text"]) for r in cached["results"]
    ]
    pipe = pipeline_from_env()
    out = pipe.analyze_batch(comments)
    out = apply_creator_buckets(out, comments)
    nh = int(out.summary.get("needs_human", 0))
    rag = next((r for r in out.results if "only able to use AI" in r.text), None)
    print(
        json.dumps(
            {
                "job_id": JOB_ID,
                "total": len(out.results),
                "needs_human_before": BASELINE_NEEDS_HUMAN,
                "needs_human_after": nh,
                "moderation_llm_requests": out.summary.get("moderation_llm_requests"),
                "creator_bucket_llm_requests": out.summary.get("creator_bucket_llm_requests"),
                "layer_counts": out.summary.get("layer_counts"),
                "rag_comment": rag.model_dump() if rag else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()