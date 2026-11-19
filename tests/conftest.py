"""Test fixtures: LR model bundle under tests/fixtures (does not touch data/raw)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MODEL_DIR = Path(__file__).resolve().parent / "fixtures" / "models" / "lr" / "v1"


def _build_fixture_bundle() -> None:
    FIXTURE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FIXTURE_MODEL_DIR / "model.joblib"
    if model_path.exists():
        model_path.unlink()

    safe_thanks = [
        "thank you great video",
        "thanks for the tutorial",
        "thank you so much",
        "thanks explained perfectly",
    ]
    corpus = [
        ("safe comment", 0),
        ("great job thanks", 0),
        *[(t, 0) for t in safe_thanks],
        ("what is the timestamp", 0),
        ("as a gay developer i love this", 0),
        ("toxic hateful garbage idiot", 1),
        ("kill threat murder", 1),
        ("thanks for nothing idiot", 1),
        ("thanks for nothing idiot", 1),
        ("thanks for nothing you idiot", 1),
        ("what kind of moron makes this", 1),
        ("what kind of moron makes this", 1),
        ("moron question what kind", 1),
        ("buy now click here free money", 1),
    ]
    texts = [t for t, _ in corpus]
    y = [lbl for _, lbl in corpus]
    harmful = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    harmful.fit(texts, y)
    insult = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    insult_y = [1 if "idiot" in t or "moron" in t or "trash" in t else 0 for t in texts]
    insult.fit(texts, insult_y)
    bundle = {
        "version": 2,
        "harmful": harmful,
        "heads": {"toxicity": harmful, "insult": insult},
    }
    joblib.dump(bundle, model_path)
    thresholds = {"t_low": 0.4, "t_high": 0.85}
    (FIXTURE_MODEL_DIR / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def _configure_lr_fixture() -> None:
    _build_fixture_bundle()
    os.environ["LR_MODEL_PATH"] = str(FIXTURE_MODEL_DIR / "model.joblib")
    os.environ["LR_THRESHOLDS_PATH"] = str(FIXTURE_MODEL_DIR / "thresholds.json")
    os.environ["USE_TRANSFORMER"] = "false"
    os.environ["LLM_BACKEND"] = "openrouter"