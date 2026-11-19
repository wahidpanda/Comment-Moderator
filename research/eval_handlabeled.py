"""Evaluate hand-labeled creator CSV (excluded from training/rules tuning)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput
from research.eval import pred_harmful_from_results
from research.stats import bootstrap_f1_ci

DEFAULT_PATH = Path("research/data/creator_handlabeled.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--hybrid", action="store_true", default=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    labeled = [r for r in rows if r.get("label_expected", "").strip()]
    if not labeled:
        raise SystemExit(f"No labels in {args.csv} — fill label_expected after import")
    pipe = ModerationPipeline(use_transformer=args.hybrid)
    resp = pipe.analyze_batch(
        [CommentInput(id=r["id"], text=r["text"]) for r in labeled]
    )
    harmful_expected = {"toxic", "threat", "spam", "identity_attack"}
    y_true = [1 if r["label_expected"] in harmful_expected else 0 for r in labeled]
    import numpy as np

    y_pred = pred_harmful_from_results(resp.results)
    stats = bootstrap_f1_ci(np.asarray(y_true), y_pred)
    out = {"n": len(labeled), "f1_harmful": stats, "path": str(args.csv)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()