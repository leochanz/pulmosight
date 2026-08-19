from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "app.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """Create the SQLite tables used by the historical history dashboard."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_jobs (
                job_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                job_id TEXT PRIMARY KEY,
                is_cancer INTEGER NOT NULL DEFAULT 0,
                malignancy_score REAL,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES scan_jobs(job_id)
            )
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_jobs_patient ON scan_jobs(patient_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_results_job ON analysis_results(job_id)"
        )
        conn.commit()
    finally:
        conn.close()


def upsert_patient(patient_id: str, name: str, age: int | str | None, gender: str | None) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO patients (patient_id, name, age, gender)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(patient_id)
            DO UPDATE SET name = excluded.name, age = excluded.age, gender = excluded.gender
            """,
            (patient_id, name, int(age) if str(age).strip() else None, gender),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_scan_job(job_id: str, patient_id: str, scan_date: str, image_path: str | None = None) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO scan_jobs (job_id, patient_id, scan_date, image_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id)
            DO UPDATE SET patient_id = excluded.patient_id,
                         scan_date = excluded.scan_date,
                         image_path = excluded.image_path
            """,
            (job_id, patient_id, scan_date, image_path),
        )
        conn.commit()
    finally:
        conn.close()


def save_analysis_result(
    job_id: str,
    is_cancer: bool,
    malignancy_score: float | None,
    confidence: float | None,
) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO analysis_results (job_id, is_cancer, malignancy_score, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id)
            DO UPDATE SET is_cancer = excluded.is_cancer,
                          malignancy_score = excluded.malignancy_score,
                          confidence = excluded.confidence,
                          created_at = CURRENT_TIMESTAMP
            """,
            (job_id, 1 if is_cancer else 0, malignancy_score, confidence),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_history() -> list[dict[str, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                sj.job_id,
                sj.patient_id,
                p.name AS patient_name,
                p.age,
                p.gender,
                sj.scan_date,
                sj.image_path,
                ar.is_cancer,
                ar.malignancy_score,
                ar.confidence
            FROM scan_jobs sj
            LEFT JOIN patients p ON p.patient_id = sj.patient_id
            LEFT JOIN analysis_results ar ON ar.job_id = sj.job_id
            ORDER BY datetime(sj.scan_date) DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_analysis_detail(job_id: str) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT
                sj.job_id,
                sj.patient_id,
                p.name AS patient_name,
                p.age,
                p.gender,
                sj.scan_date,
                sj.image_path,
                ar.is_cancer,
                ar.malignancy_score,
                ar.confidence
            FROM scan_jobs sj
            LEFT JOIN patients p ON p.patient_id = sj.patient_id
            LEFT JOIN analysis_results ar ON ar.job_id = sj.job_id
            WHERE sj.job_id = ?
            """,
            (job_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_db()
    print(f"SQLite database initialized at: {DB_PATH}")
