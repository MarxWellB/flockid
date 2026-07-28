"""SQLite connection helper and schema initialization."""
import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "flockid.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset: bool = False):
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = get_connection()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def dict_from_row(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}
