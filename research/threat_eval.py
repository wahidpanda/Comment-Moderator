"""Threat eval: creator / holdout with threat rules disabled."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel
from engine.rules import load_rules
from research.threat_metrics import predict_threat

CREATOR = Path("research/data/creator_eval.csv")
HOLDOUT = Path("research/data/threat_holdout.csv")


def _threat_rule_patterns() -> list[str]:
    return list(load_rules().get("threat", {}).get("patterns", []))


def matches_threat_rule(text: str) -> bool:
    for pattern in _threat_rule_patterns():
        if re.search(pattern, text):
            return True
    return False


def pipeline_without_threat_rules(*, use_transformer: bool) -> ModerationPipeline:
    pipe = ModerationPipeline(use_transformer=use_transformer)
    orig = pipe.rules.evaluate

    def _wrapped(text: str, comment_id: str) -> dict[str, Any] | None:
        hit = orig(text, comment_id)
        if hit is not None and hit.get("label") == ModerationLabel.THREAT:
            return None
        return hit

    pipe.rules.evaluate = _wrapped  # type: ignore[method-assign]
    return pipe


def eval_threat_rows(rows: list[dict[str, str]], pipe: ModerationPipeline) -> dict[str, Any]:
    results = pipe.analyze_batch([CommentInput(id=r["id"], text=r["text"]) for r in rows])
    detected = sum(1 for r in results.results if predict_threat(r))
    return {
        "n": len(rows),
        "detected": detected,
        "recall": detected / max(len(rows), 1),
        "ids": [r["id"] for r in rows],
        "predictions": [r.label.value for r in results.results],
    }


def run_threat_evals() -> dict[str, Any]:
    creator = list(csv.DictReader(CREATOR.open(encoding="utf-8")))
    creator_threats = [r for r in creator if r.get("label_expected") == "threat"]
    holdout = list(csv.DictReader(HOLDOUT.open(encoding="utf-8")))
    rule_covered = [r for r in creator_threats if matches_threat_rule(r["text"])]
    non_rule = [r for r in creator_threats if not matches_threat_rule(r["text"])]
    pipes = {
        "A_no_threat_rules": pipeline_without_threat_rules(use_transformer=False),
        "C_no_threat_rules": pipeline_without_threat_rules(use_transformer=True),
    }
    out: dict[str, Any] = {
        "rule_patterns_count": len(_threat_rule_patterns()),
        "creator_threats_rule_covered": [r["id"] for r in rule_covered],
        "creator_threats_non_rule": [r["id"] for r in non_rule],
    }
    for name, pipe in pipes.items():
        out[name] = {
            "creator_all_threats": eval_threat_rows(creator_threats, pipe),
            "creator_non_rule_threats": eval_threat_rows(non_rule, pipe) if non_rule else {"n": 0},
            "holdout_csv": eval_threat_rows(holdout, pipe),
        }
    return out