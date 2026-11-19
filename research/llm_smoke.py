"""Live OpenRouter smoke test (one batch). Not for CI."""

from __future__ import annotations

import json
import sys

from engine.llm.client import LLMClient, LLMError, LLMNoEndpointsError
from engine.llm.config import load_openrouter_config
from engine.schemas import CommentInput

# Fixed benign samples — not drawn from Civil eval holdout.
SMOKE_COMMENTS: list[tuple[str, str]] = [
    ("s01", "Thanks for the clear walkthrough on proxy metrics."),
    ("s02", "Could you link the paper you mentioned at minute twelve?"),
    ("s03", "The audio is a bit quiet on my phone but content is great."),
    ("s04", "Subscribed after seeing the spam layer demo."),
    ("s05", "Honest critique: the intro runs long."),
    ("s06", "Neutral note: release v2 touched the LR thresholds."),
    ("s07", "Praise for keeping identity terms out of rules."),
    ("s08", "Question: does hybrid mode call Detoxify on every comment?"),
    ("s09", "Appreciate the Wilson CI explanation."),
    ("s10", "This helped me tune needs_human on my channel."),
    ("s11", "Fine summary of threat vs toxicity heads."),
    ("s12", "Would love a follow-up on creator_eval rows."),
    ("s13", "Benign shout-out to the research reproducibility notes."),
    ("s14", "Logistics: when is phase 3 API shipping?"),
    ("s15", "Solid moderation philosophy in the README."),
]


def main() -> int:
    cfg = load_openrouter_config()
    if not cfg.api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1
    if not cfg.models:
        print("Set OPENROUTER_MODEL or OPENROUTER_MODELS", file=sys.stderr)
        return 1

    client = LLMClient(cfg)
    comments = [CommentInput(id=cid, text=t) for cid, t in SMOKE_COMMENTS]
    print(
        f"Smoke batch: n={len(comments)} configured_batch_size={cfg.batch_size} "
        f"data_collection={cfg.data_collection} models={cfg.models}"
    )
    endpoint_ok = False
    try:
        rows, metas = client.moderate_batch(comments)
        endpoint_ok = True
    except LLMNoEndpointsError as exc:
        print("NO_ENDPOINT under data_collection=deny:", exc)
        return 2
    except LLMError as exc:
        print("LLM error:", exc, file=sys.stderr)
        return 1

    for meta in metas:
        if meta.from_cache:
            print("meta: from_cache", meta.model_used)
            continue
        raw = json.loads(meta.raw_response) if meta.raw_response else {}
        print("response.model:", raw.get("model", meta.model_used))
        print("raw_response (truncated):", meta.raw_response[:2000])
    print("parsed_items:", json.dumps([{**r, "label": r["label"].value} for r in rows], indent=2))
    print("data_collection=deny endpoint_available:", endpoint_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())