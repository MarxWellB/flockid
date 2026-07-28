-- Esquema real, adaptado de la sección 11 del documento de arquitectura
-- (allí era PostgreSQL de diseño; esto es SQLite REAL que corre en este
-- entorno -- sin PostgreSQL disponible aquí, SQLite es la adaptación
-- honesta para demostrar el esquema funcionando, no un reemplazo de
-- producción. Las particiones por fecha y los tipos UUID nativos de
-- Postgres se simplifican aquí -- documentado, no escondido).

CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS farms (
    id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES tenants(id),
    name TEXT NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS houses (
    id TEXT PRIMARY KEY,
    farm_id TEXT REFERENCES farms(id),
    name TEXT NOT NULL,
    capacity INTEGER
);

CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    house_id TEXT REFERENCES houses(id),
    label TEXT,
    position_x REAL,
    position_y REAL,
    calibration TEXT  -- JSON serializado
);

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    house_id TEXT REFERENCES houses(id),
    start_date TEXT,
    end_date TEXT,
    bird_count INTEGER
);

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT REFERENCES batches(id),
    camera_id TEXT REFERENCES cameras(id),
    global_identity_id INTEGER,
    first_seen TEXT,
    last_seen TEXT
);

CREATE TABLE IF NOT EXISTS track_positions (
    track_id INTEGER REFERENCES tracks(id),
    frame_ts TEXT,
    x REAL, y REAL, vx REAL, vy REAL,
    PRIMARY KEY (track_id, frame_ts)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT REFERENCES batches(id),
    source_engine TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_id INTEGER,
    confidence REAL,
    evidence TEXT,  -- JSON serializado
    clip_url TEXT,
    occurred_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT REFERENCES batches(id),
    computed_at TEXT,
    health_score REAL,
    behavior_score REAL,
    risk_score REAL,
    top_evidence TEXT  -- JSON serializado
);

CREATE TABLE IF NOT EXISTS bird_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT REFERENCES batches(id),
    track_id INTEGER,
    resolved_tag TEXT,
    fusion_confidence REAL,
    egg_count INTEGER DEFAULT 0,
    avg_egg_weight_g REAL,
    last_risk_score REAL,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_batch_time ON events (batch_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_track_positions_track ON track_positions (track_id, frame_ts DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_batch_time ON risk_scores (batch_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_bird_profiles_batch ON bird_profiles (batch_id, last_risk_score DESC);
