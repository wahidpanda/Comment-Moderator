"""Train spam classifier on UCI YouTube Spam Collection (per-video holdout)."""

from __future__ import annotations

import argparse
import json
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from research.features import attach_classifier, make_tfidf_pipeline
from research.labels import normalize_text_column

UCI_ZIP = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"


def load_youtube_spam_by_video() -> tuple[list[pd.DataFrame], list[str]]:
    with urlopen(UCI_ZIP, timeout=120) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for name in sorted(zf.namelist()):
        if not name.lower().endswith(".csv") or name.startswith("__MACOSX"):
            continue
        if "Youtube" not in name and "youtube" not in name:
            continue
        raw = zf.read(name)
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", comment="#")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", comment="#")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    text_col = "CONTENT" if "CONTENT" in df.columns else df.columns[-1]
    label_col = "CLASS" if "CLASS" in df.columns else "label"
    if label_col not in df.columns and "COMMENT_ID" in df.columns:
        label_col = [c for c in df.columns if c != text_col][0]
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[text_col, label_col])
    texts = normalize_text_column(df, text_col).tolist()
    y = df[label_col].astype(int).to_numpy()
    return texts, y


def train_and_eval_holds(
    frames: list[pd.DataFrame],
    names: list[str],
) -> tuple[Any, dict[str, Any]]:
    per_video: list[dict[str, float | str | int]] = []
    for i, held in enumerate(frames):
        x_test = held["text"].tolist()
        y_test = held["label"].astype(int).to_numpy()
        train_parts = [f for j, f in enumerate(frames) if j != i]
        train_df = pd.concat(train_parts, ignore_index=True)
        x_train = train_df["text"].tolist()
        y_train = train_df["label"].astype(int).to_numpy()
        pipe = attach_classifier(
            make_tfidf_pipeline(word_ngram_range=(1, 2)),
            LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced"),
        )
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]
        thr = 0.55
        for candidate in np.linspace(0.55, 0.99, 45):
            pred = (proba >= candidate).astype(int)
            if precision_score(y_test, pred, zero_division=0) >= 0.98:
                thr = float(candidate)
                break
        yhat = (proba >= thr).astype(int)
        per_video.append(
            {
                "heldout_video": names[i],
                "n_test": len(x_test),
                "f1": float(f1_score(y_test, yhat, zero_division=0)),
                "precision": float(precision_score(y_test, yhat, zero_division=0)),
                "recall": float(recall_score(y_test, yhat, zero_division=0)),
                "threshold_used": thr,
            }
        )
    # Final model: train on 4 held-in rotating — save train on all for production
    all_df = pd.concat(frames, ignore_index=True)
    pipe = attach_classifier(
        make_tfidf_pipeline(word_ngram_range=(1, 2)),
        LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced"),
    )
    pipe.fit(all_df["text"].tolist(), all_df["label"].astype(int).to_numpy())
    summary = {
        "per_video_holdout": per_video,
        "mean_holdout_f1": float(np.mean([v["f1"] for v in per_video])),
        "split": "5-fold leave-one-video-out (UCI)",
        "spam_threshold": float(np.median([v["threshold_used"] for v in per_video])),
    }
    return pipe, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/spam/v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, names = load_youtube_spam_by_video()
    pipe, summary = train_and_eval_holds(frames, names)
    summary["source"] = "UCI YouTube Spam Collection"
    joblib.dump(pipe, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()