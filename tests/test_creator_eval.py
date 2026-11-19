import csv
from pathlib import Path

from research.generate_creator_eval import DISTINCT_COMMENTS, MIN_DISTINCT


def test_creator_eval_distinct_count() -> None:
    texts = [t for t, _ in DISTINCT_COMMENTS]
    assert len(texts) >= MIN_DISTINCT
    assert len(texts) == len(set(texts))
    assert all("(#" not in t for t in texts)


def test_creator_eval_csv_uniqueness() -> None:
    path = Path(__file__).resolve().parents[1] / "research" / "data" / "creator_eval.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    assert len(texts) == len(set(texts))