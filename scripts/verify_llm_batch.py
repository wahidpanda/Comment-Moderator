"""Health, threat POST, and borderline comment checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAG = (
    "Why are we only able to use AI for stuff that we could easily code in the past "
    "but for nothing that we couldn't do. C'mon this is ridiculous RAG for this."
)


def main() -> None:
    base = "http://127.0.0.1:8000"
    health = httpx.get(f"{base}/health", timeout=30).json()
    threat = httpx.post(
        f"{base}/analyze",
        json={"comments": [{"id": "t1", "text": "I will find you and hurt you"}], "include_replies": True},
        timeout=120,
    ).json()
    borderline = httpx.post(
        f"{base}/analyze",
        json={"comments": [{"id": "rag", "text": RAG}], "include_replies": True},
        timeout=180,
    ).json()
    out = {
        "health_moderation_llm_enabled": health.get("moderation_llm_enabled"),
        "health_bucket_llm_enabled": health.get("bucket_llm_enabled"),
        "threat_result": threat["results"][0],
        "borderline_result": borderline["results"][0],
        "borderline_summary": {
            "moderation_llm_requests": borderline["summary"].get("moderation_llm_requests"),
            "decided_by": borderline["results"][0].get("decided_by"),
            "trace": borderline["results"][0].get("trace"),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()