"""Download Civil Comments (Hugging Face) and optional Jigsaw Toxic Comments (Kaggle path)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PARQUET_ROWS = 50_000


def _row_count(path: Path) -> int:
    return len(pd.read_parquet(path))


def download_civil_comments(
    out_dir: Path,
    *,
    sample: int | None = None,
    force: bool = False,
    seed: int = 42,
) -> Path:
    """Fetch Civil Comments via `datasets` and write parquet."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "civil_comments.parquet"

    if dest.exists() and not force:
        rows = _row_count(dest)
        if rows < MIN_PARQUET_ROWS:
            print(
                f"ERROR: {dest} has {rows} rows (< {MIN_PARQUET_ROWS}). "
                "Refusing to reuse. Re-run with --force to replace.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if sample is None:
            print(f"Skip existing {dest} ({rows} rows)")
            return dest
        if rows < sample:
            print(
                f"ERROR: existing parquet has {rows} rows but --sample={sample} requested. "
                "Use --force to re-download.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    print("Loading Civil Comments from Hugging Face (google/civil_comments)...")
    ds = load_dataset("google/civil_comments", split="train")
    if sample is not None:
        n = min(sample, len(ds))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=n, replace=False)
        ds = ds.select(sorted(indices.tolist()))
    df = ds.to_pandas()
    df.to_parquet(dest, index=False)
    meta = {"rows": len(df), "source": "huggingface:google/civil_comments", "sample": sample, "seed": seed}
    (out_dir / "civil_comments.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {dest} ({len(df)} rows)")
    return dest


def ingest_jigsaw(src: Path, out_dir: Path, *, force: bool = False) -> Path:
    """Copy or convert Jigsaw train.csv to parquet in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "jigsaw_toxic_comments.parquet"
    if dest.exists() and not force:
        print(f"Skip existing {dest}")
        return dest
    if not src.exists():
        raise FileNotFoundError(
            f"Jigsaw file not found: {src}. Download from Kaggle "
            "(jigsaw-toxic-comment-classification-challenge) and pass --jigsaw-path."
        )
    if src.suffix.lower() == ".parquet":
        shutil.copy2(src, dest)
    else:
        df = pd.read_csv(src)
        df.to_parquet(dest, index=False)
    meta = {"rows": len(pd.read_parquet(dest)), "source": str(src)}
    (out_dir / "jigsaw_toxic_comments.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download moderation training data")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random subsample size (not the first N rows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    parser.add_argument("--force", action="store_true", help="Replace existing parquet files")
    parser.add_argument(
        "--jigsaw-path",
        type=Path,
        default=None,
        help="Local path to Jigsaw train.csv or parquet (optional)",
    )
    parser.add_argument("--skip-civil", action="store_true")
    args = parser.parse_args()

    if not args.skip_civil:
        download_civil_comments(args.out, sample=args.sample, force=args.force, seed=args.seed)
    if args.jigsaw_path:
        ingest_jigsaw(args.jigsaw_path, args.out, force=args.force)
    elif (args.out / "jigsaw_toxic_comments.parquet").exists():
        print("Jigsaw parquet already present")
    else:
        print(
            "No Jigsaw data yet. Optional: download train.csv from Kaggle and run\n"
            "  python -m research.download --jigsaw-path path/to/train.csv"
        )


if __name__ == "__main__":
    main()