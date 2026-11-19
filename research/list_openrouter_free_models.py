"""List current :free chat models from OpenRouter."""

from __future__ import annotations

import argparse
import json
import os

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    headers = {}
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base}/models?min_price=0&endpoints=/api/v1/chat/completions"
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    models = []
    for row in payload.get("data", []):
        mid = row.get("id", "")
        if not mid.endswith(":free"):
            continue
        pricing = row.get("pricing") or {}
        models.append(
            {
                "id": mid,
                "name": row.get("name"),
                "context_length": row.get("context_length"),
                "prompt_per_m": pricing.get("prompt"),
                "completion_per_m": pricing.get("completion"),
            }
        )
    print(json.dumps({"count": len(models), "models": models}, indent=2))


if __name__ == "__main__":
    main()