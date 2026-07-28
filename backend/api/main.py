"""
REST API for the FlockID platform.

Run:
    uvicorn api.main:app --reload --port 8000

Try:
    curl http://localhost:8000/api/v1/batches
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from database.db import get_connection, dict_from_row

app = FastAPI(title="FlockID API", version="0.1.0",
              description="API for the poultry visual intelligence platform.")


@app.get("/api/v1/farms/{farm_id}/houses")
def get_houses(farm_id: str):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM houses WHERE farm_id = ?", (farm_id,)).fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/houses/{house_id}/batches")
def get_batches(house_id: str):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM batches WHERE house_id = ?", (house_id,)).fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/batches")
def list_batches():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM batches").fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/batches/{batch_id}/events")
def get_events(batch_id: str, event_type: Optional[str] = Query(None), limit: int = 50):
    conn = get_connection()
    if event_type:
        rows = conn.execute(
            "SELECT * FROM events WHERE batch_id = ? AND event_type = ? ORDER BY id DESC LIMIT ?",
            (batch_id, event_type, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events WHERE batch_id = ? ORDER BY id DESC LIMIT ?",
            (batch_id, limit)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict_from_row(r)
        d["evidence"] = json.loads(d["evidence"]) if d["evidence"] else {}
        out.append(d)
    return out


@app.get("/api/v1/batches/{batch_id}/risk-score/history")
def get_risk_history(batch_id: str):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM risk_scores WHERE batch_id = ? ORDER BY computed_at", (batch_id,)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="No hay risk_scores para este batch_id")
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/batches/{batch_id}/birds")
def get_bird_profiles(batch_id: str, sort_by_risk: bool = True, limit: int = 30):
    conn = get_connection()
    order = "ORDER BY last_risk_score DESC" if sort_by_risk else ""
    rows = conn.execute(
        f"SELECT * FROM bird_profiles WHERE batch_id = ? {order} LIMIT ?", (batch_id, limit)).fetchall()
    conn.close()
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/tracks/{track_id}/trajectory")
def get_trajectory(track_id: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM track_positions WHERE track_id = ? ORDER BY frame_ts", (track_id,)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="track_id no encontrado")
    return [dict_from_row(r) for r in rows]


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "FlockID API"}
