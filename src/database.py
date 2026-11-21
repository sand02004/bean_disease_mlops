"""
database.py
-----------
A small SQLite database that records:
  - every image uploaded for retraining (file path, label, date, used or not)
  - every retraining run (date, number of images, metrics, deployed or not)
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "uploads.db")


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """Create the tables if they don't exist yet."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                label TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                used_for_training INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                finished_at TEXT,
                num_new_images INTEGER,
                old_accuracy REAL,
                new_accuracy REAL,
                new_f1 REAL,
                deployed INTEGER,
                model_version TEXT
            )""")


def add_upload(filename, filepath, label, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO uploads (filename, filepath, label, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, filepath, label, datetime.now().isoformat(timespec="seconds")),
        )


def count_new_uploads(db_path=DB_PATH):
    """Number of uploaded images not yet used for retraining."""
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM uploads WHERE used_for_training = 0").fetchone()[0]


def upload_counts_by_label(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n, SUM(used_for_training = 0) AS new "
            "FROM uploads GROUP BY label").fetchall()
    return {r["label"]: {"total": r["n"], "new": r["new"]} for r in rows}


def mark_uploads_used(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute("UPDATE uploads SET used_for_training = 1 WHERE used_for_training = 0")


def log_training_run(run, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO training_runs (started_at, finished_at, num_new_images, old_accuracy, "
            "new_accuracy, new_f1, deployed, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run["started_at"], run["finished_at"], run["num_new_images"],
             run["old_metrics"]["accuracy"], run["new_metrics"]["accuracy"],
             run["new_metrics"]["f1_score"], int(run["deployed"]), run["model_version"]),
        )


def get_training_runs(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]