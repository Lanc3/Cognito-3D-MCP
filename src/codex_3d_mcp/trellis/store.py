"""SQLite-backed durable TRELLIS job state."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import JobNotFoundError

TERMINAL_STATES = {"completed", "failed", "failed_quality", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrellisJobStore:
    """Serialize all writes and make stage transitions restart-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()
        self._interrupt_inflight()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, stage),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                );
                """
            )

    def _interrupt_inflight(self) -> None:
        now = utc_now()
        with self._connection:
            self._connection.execute(
                """
                UPDATE jobs SET state='interrupted', updated_at=?
                WHERE state IN ('queued', 'running')
                """,
                (now,),
            )

    def create(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id,prompt,state,stage,progress,params_json,metadata_json,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (job_id, prompt, "queued", "validating_pair", 0, _json(params), "{}", now, now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"Unknown job_id: {job_id}")
        return _row(row)

    def list_by_state(self, *states: str) -> list[dict[str, Any]]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY created_at", states
            ).fetchall()
        return [_row(row) for row in rows]

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        permitted = {
            "state",
            "stage",
            "progress",
            "metadata",
            "error",
            "cancel_requested",
            "started_at",
            "finished_at",
        }
        unknown = set(changes) - permitted
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            column = f"{key}_json" if key in {"metadata", "error"} else key
            if key in {"metadata", "error"}:
                value = None if value is None else _json(value)
            if key == "cancel_requested":
                value = int(bool(value))
            assignments.append(f"{column}=?")
            values.append(value)
        assignments.append("updated_at=?")
        values.extend([utc_now(), job_id])
        with self._lock, self._connection:
            result = self._connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id=?", values
            )
            if result.rowcount != 1:
                raise JobNotFoundError(f"Unknown job_id: {job_id}")
        return self.get(job_id)

    def merge_metadata(self, job_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.get(job_id)
            metadata = record["metadata"]
            metadata.update(values)
            return self.update(job_id, metadata=metadata)

    def checkpoint(self, job_id: str, stage: str, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO checkpoints(job_id,stage,payload_json,created_at)
                VALUES (?,?,?,?)
                ON CONFLICT(job_id,stage) DO UPDATE SET
                    payload_json=excluded.payload_json, created_at=excluded.created_at
                """,
                (job_id, stage, _json(payload), utc_now()),
            )

    def checkpoints(self, job_id: str) -> dict[str, dict[str, Any]]:
        self.get(job_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT stage,payload_json FROM checkpoints WHERE job_id=?", (job_id,)
            ).fetchall()
        return {row["stage"]: json.loads(row["payload_json"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "prompt": row["prompt"],
        "state": row["state"],
        "status": row["state"],
        "stage": row["stage"],
        "progress": row["progress"],
        "params": json.loads(row["params_json"]),
        "metadata": json.loads(row["metadata_json"]),
        "error": json.loads(row["error_json"]) if row["error_json"] else None,
        "cancel_requested": bool(row["cancel_requested"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }
