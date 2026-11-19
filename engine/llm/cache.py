"""SQLite cache for LLM moderation responses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from engine.llm.text import normalize_comment_text

DEFAULT_DB = Path(os.environ.get("LLM_CACHE_DB", "data/llm_cache.sqlite"))


def _resolve_db(db_path: Path | None) -> Path:
    return db_path if db_path is not None else DEFAULT_DB


@dataclass(frozen=True)
class CacheContext:
    prompt_version: str
    batch_size: int
    model_chain: str
    sweep_tag: str | None = None


def cache_key(text: str, ctx: CacheContext) -> str:
    norm = normalize_comment_text(text)
    sweep = ctx.sweep_tag or ""
    payload = f"{norm}\n{ctx.prompt_version}\n{ctx.batch_size}\n{ctx.model_chain}\n{sweep}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def _connect(db_path: Path = DEFAULT_DB) -> Generator[sqlite3.Connection, None, None]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                normalized_text TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                model_chain TEXT NOT NULL,
                response_json TEXT NOT NULL,
                raw_response TEXT,
                model_used TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            CREATE TABLE IF NOT EXISTS llm_budget_day (
                day TEXT PRIMARY KEY,
                request_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_cached(key: str, *, db_path: Path | None = None) -> dict[str, Any] | None:
    with _connect(_resolve_db(db_path)) as conn:
        row = conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["response_json"])


def put_cached(
    key: str,
    text: str,
    ctx: CacheContext,
    response: dict[str, Any],
    *,
    raw_response: str | None = None,
    model_used: str | None = None,
    db_path: Path | None = None,
) -> None:
    with _connect(_resolve_db(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (cache_key, normalized_text, prompt_version, batch_size, model_chain,
             response_json, raw_response, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                normalize_comment_text(text),
                ctx.prompt_version,
                ctx.batch_size,
                ctx.model_chain,
                json.dumps(response),
                raw_response,
                model_used,
            ),
        )


def increment_daily_requests(*, db_path: Path | None = None) -> int:
    db_path = _resolve_db(db_path)
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_budget_day (day, request_count) VALUES (?, 1)
            ON CONFLICT(day) DO UPDATE SET request_count = request_count + 1
            """,
            (day,),
        )
        row = conn.execute(
            "SELECT request_count FROM llm_budget_day WHERE day = ?",
            (day,),
        ).fetchone()
    return int(row["request_count"]) if row else 1


def get_daily_requests(*, db_path: Path | None = None) -> int:
    db_path = _resolve_db(db_path)
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT request_count FROM llm_budget_day WHERE day = ?",
            (day,),
        ).fetchone()
    return int(row["request_count"]) if row else 0