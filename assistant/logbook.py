"""The command log: every handled command is one row in logs/commands.db (SQLite: built into Python,
one file, no server). train/learn_from_logs.py turns the LLM-handled rows into training data.

Look at it from the terminal:
    sqlite3 logs/commands.db "select time, path, text, error from commands order by rowid desc limit 20"
"""

import json
import sqlite3

from assistant.config import LOGS_DIR

DB = LOGS_DIR / "commands.db"


def _connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB)
    # time/text/path/error as columns for queries; the whole entry (steps, Laya's decision, timings) as JSON
    db.execute("create table if not exists commands (time text, text text, path text, error text, data text)")
    return db


def add(entry: dict) -> None:
    """Store one handled command (needs the keys time, text, path and error; everything is kept as JSON)."""
    db = _connect()
    with db:
        db.execute("insert into commands values (?, ?, ?, ?, ?)",
                   (entry["time"], entry["text"], entry["path"], entry["error"], json.dumps(entry)))
    db.close()


def entries(path: str | None = None) -> list[dict]:
    """All logged commands, oldest first; only one path ("laya", "llm", ...) if given."""
    db = _connect()
    rows = db.execute("select data from commands" + (" where path = ?" if path else "") + " order by rowid",
                      (path,) if path else ()).fetchall()
    db.close()
    return [json.loads(data) for (data,) in rows]
