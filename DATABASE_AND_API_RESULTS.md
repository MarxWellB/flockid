# Database and API -- from paper design to a running service

## What existed before

The architecture document had a full SQL schema and an API design -- but
those were design only, no tables had ever been created and no server had
ever run.

## What exists now

**A real SQLite database** (`backend/database/schema.sql` + `db.py`),
with the same relational schema from the document (adapted from
PostgreSQL to SQLite -- no date partitioning or native UUIDs, honestly
noted as a simplification) -- 10 tables: tenants, farms, houses, cameras,
batches, tracks, track_positions, events, risk_scores, bird_profiles.

**A real ingestion pipeline** (`ingest_pipeline.py`) that runs the full
system (nest simulator -> Detector -> Tracker -> Behavior -> Risk ->
Identity Fusion) and persists the results into the database -- real
relational rows, not the loose JSON files used in earlier demos.

**A real REST API** (`backend/api/main.py`, FastAPI) -- run and tested
with `curl` against real data, not just defined on paper:

```
GET /api/v1/batches                          -> existing batches
GET /api/v1/batches/{id}/events               -> events (filterable by type)
GET /api/v1/batches/{id}/birds                -> bird profiles, sorted by risk
GET /api/v1/batches/{id}/risk-score/history    -> risk score history
GET /api/v1/tracks/{id}/trajectory             -> full trajectory for a track
GET /api/v1/health                             -> healthcheck
```

## Real verification (not just "should work")

```bash
$ curl http://localhost:8000/api/v1/batches/{batch_id}/birds?limit=3
[
  {"track_id": 18, "last_risk_score": 85.3, ...},
  {"track_id": 10, "last_risk_score": 83.2, ...},
  {"track_id": 116, "resolved_tag": "TAG-009", "last_risk_score": 78.1, ...}
]
```

148 tracks, 2,018 trajectory points, 134 events, 148 bird profiles -- all
queryable over HTTP, not sitting in a notebook.

## How to run it

```bash
cd backend/database && python3 ingest_pipeline.py   # populate the database from a real run
cd .. && uvicorn api.main:app --reload --port 8000    # start the API
curl http://localhost:8000/api/v1/batches
```

## What's missing (honestly)

- SQLite, not PostgreSQL -- enough to demonstrate the schema and API
  working, not for real multi-tenant production with concurrency.
- No authentication/authorization -- already flagged as an explicit gap
  in the original architecture document, still pending.
- No WebSocket yet (designed, not implemented here) -- the REST endpoints
  cover historical queries, a live event channel is still missing.
- `ingest_pipeline.py` runs once and exits -- not a continuous real-time
  ingestion service, it's a demonstration that the data -> database -> API
  path works end to end.
