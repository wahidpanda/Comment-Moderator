"""Import YouTube (or other) comment CSV for hand-labeling — kept out of rule tuning."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("research/data/creator_handlabeled.csv")
TEXT_CANDIDATES = ("text", "comment", "comment_text", "content", "body", "message")
ID_CANDIDATES = ("id", "comment_id", "cid")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import video comment export for manual labels")
    parser.add_argument("csv", type=Path, help="Exported comments CSV from your platform")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-id", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8", on_bad_lines="skip")
    text_col = _pick_column(df.columns.tolist(), TEXT_CANDIDATES)
    if text_col is None:
        raise SystemExit(f"No text column found in {list(df.columns)}")
    id_col = _pick_column(df.columns.tolist(), ID_CANDIDATES)
    df = df[df[text_col].astype(str).str.len() > 0].reset_index(drop=True)
    if len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, row in df.iterrows():
        cid = str(row[id_col]) if id_col else f"import-{i}"
        rows.append(
            {
                "id": cid,
                "text": str(row[text_col]).strip(),
                "label_expected": "",
                "video_id": args.video_id,
                "imported_at": stamp,
                "notes": "Hand-label only; excluded from rules/LR training",
            }
        )
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()