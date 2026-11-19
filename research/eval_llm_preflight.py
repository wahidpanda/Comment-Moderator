"""Print OpenRouter eval request budget (step 4); does not call the API."""

from __future__ import annotations

import math
import os


def llm_requests_for_n(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size) if n else 0


def main() -> None:
    batch = int(os.getenv("LLM_BATCH_SIZE", "15"))
    manual_cap = int(os.getenv("OPENROUTER_MAX_REQUESTS_PER_DAY", "40"))
    buffer = int(os.getenv("OPENROUTER_DAILY_REQUEST_BUFFER", "5"))

    scenario_b_n = 1000
    sweep_n = 300
    sweep_sizes = (5, 10, 15, 25)

    b_req = llm_requests_for_n(scenario_b_n, batch)
    sweep_req = sum(llm_requests_for_n(sweep_n, bs) for bs in sweep_sizes)
    total = b_req + sweep_req

    usable = max(manual_cap - buffer, 1)
    days = math.ceil(total / usable)

    print("OpenRouter step-4 preflight (no API calls)")
    print(f"  LLM_BATCH_SIZE={batch} OPENROUTER_MAX_REQUESTS_PER_DAY={manual_cap} buffer={buffer}")
    print(f"  Scenario B (LLM-only): n={scenario_b_n} -> {b_req} requests")
    print(f"  Scenario C (hybrid): LLM calls only on escalations (not in this total)")
    print(f"  Batch sweep on n={sweep_n} sizes {sweep_sizes} -> {sweep_req} requests (sweep_tag isolated)")
    print(f"  TOTAL expected LLM requests: {total}")
    print(f"  Usable requests/day (manual cap - buffer): {usable} (re-run after smoke: min with GET /key remaining)")
    print(f"  Estimated calendar days at usable/day: {days}")
    if int(os.getenv("OPENROUTER_MAX_REQUESTS_PER_DAY", "40")) > manual_cap:
        print("  Note: if you raise OPENROUTER_MAX_REQUESTS_PER_DAY, days decrease accordingly.")


if __name__ == "__main__":
    main()