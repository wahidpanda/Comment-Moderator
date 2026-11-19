"""SQLite job store for async YouTube analysis."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Generator

from engine.schemas import AnalyzeResponse, JobResponse, JobStatus

DB_PATH = Path(os.environ.get("MODERATOR_DB", "data/moderator.db"))


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                result_json TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
            """
        )


@contextmanager
def connect() -> Generator[sqlite3.Connection, None, None]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status) VALUES (?, ?)",
            (job_id, JobStatus.PENDING.value),
        )
    return job_id


def update_job(
    job_id: str,
    *,
    status: JobStatus,
    error: str | None = None,
    result: AnalyzeResponse | None = None,
) -> None:
    payload = json.dumps(result.model_dump()) if result else None
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, result_json = ? WHERE id = ?",
            (status.value, error, payload, job_id),
        )


def get_job(job_id: str) -> JobResponse | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    result = None
    if row["result_json"]:
        result = AnalyzeResponse.model_validate(json.loads(row["result_json"]))
    return JobResponse(
        id=row["id"],
        status=JobStatus(row["status"]),
        error=row["error"],
        result=result,
    )