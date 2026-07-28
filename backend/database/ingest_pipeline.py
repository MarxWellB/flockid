"""
Runs the full pipeline (nest simulator -> Detector -> Tracker -> Behavior
-> Risk -> Identity Fusion) and persists the results into SQLite as
relational rows, the same way the API would query them.
"""
import sys
import os
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.db import init_db, get_connection

from vision.synth.nest_simulator import NestSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.behavior.engine import BehaviorEngine
from vision.behavior.zones import Zone
from vision.risk.engine import RiskEngineV1
from vision.fusion.identity_fusion import IdentityFusionEngine


def run_and_persist(n_frames: int = 1200, n_birds: int = 15, seed: int = 11):
    init_db(reset=True)
    conn = get_connection()
    cur = conn.cursor()

    tenant_id, farm_id, house_id, camera_id, batch_id = [str(uuid.uuid4()) for _ in range(5)]
    cur.execute("INSERT INTO tenants (id, name) VALUES (?, ?)", (tenant_id, "Cliente Demo"))
    cur.execute("INSERT INTO farms (id, tenant_id, name, location) VALUES (?, ?, ?, ?)",
                (farm_id, tenant_id, "Granja Demo", "N/A"))
    cur.execute("INSERT INTO houses (id, farm_id, name, capacity) VALUES (?, ?, ?, ?)",
                (house_id, farm_id, "House 3", n_birds))
    cur.execute("INSERT INTO cameras (id, house_id, label, position_x, position_y) VALUES (?, ?, ?, ?, ?)",
                (camera_id, house_id, "cam_1", 480, 270))
    cur.execute("INSERT INTO batches (id, house_id, start_date, end_date, bird_count) VALUES (?, ?, ?, ?, ?)",
                (batch_id, house_id, "2026-07-01", None, n_birds))
    conn.commit()

    sim = NestSimulator(n_birds=n_birds, seed=seed)
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    zones = [Zone(n.nest_id, n.x - n.radius, n.y - n.radius, n.x + n.radius, n.y + n.radius) for n in sim.nests]
    behavior = BehaviorEngine(zones=zones, frame_size=(sim.width, sim.height),
                               low_activity_window=30, low_activity_threshold=0.3)
    fusion = IdentityFusionEngine(min_observations_to_resolve=3)
    risk_engine = RiskEngineV1()

    seen_tracks = set()
    for frame_idx in range(n_frames):
        frame, gt, rfid_reads, egg_events = sim.step()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        behavior.process_frame(frame_idx, tracks)
        track_positions = {tid: (t.x, t.y) for tid, t in tracks.items()}
        fusion.observe(frame_idx, track_positions, rfid_reads, egg_events, sim.nests)

        for tid, t in tracks.items():
            if tid not in seen_tracks:
                cur.execute(
                    "INSERT INTO tracks (id, batch_id, camera_id, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (tid, batch_id, camera_id, str(frame_idx), str(frame_idx)))
                seen_tracks.add(tid)
            else:
                cur.execute("UPDATE tracks SET last_seen = ? WHERE id = ?", (str(frame_idx), tid))

            # sample positions every 10 frames -- storing every frame of
            # every track bloats the database without adding much value here
            if frame_idx % 10 == 0:
                cur.execute(
                    "INSERT OR REPLACE INTO track_positions (track_id, frame_ts, x, y, vx, vy) VALUES (?, ?, ?, ?, ?, ?)",
                    (tid, str(frame_idx), t.x, t.y, 0, 0))

    conn.commit()

    behavior_report = behavior.summary()
    for e in behavior_report["events"]:
        cur.execute(
            """INSERT INTO events (batch_id, source_engine, event_type, entity_id, confidence, evidence, occurred_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (batch_id, "behavior", e["event_type"], e["entity_id"], e["confidence"],
             json.dumps(e["evidence"]), str(e["frame_idx"])))
    conn.commit()

    fusion_summary = fusion.summary()
    for tid, stats in behavior_report["tracks"].items():
        tid_int = int(tid)
        events_for_track = [e for e in behavior_report["events"] if e["entity_id"] == tid_int]
        rr = risk_engine.score({"tracks": {tid_int: stats}, "events": events_for_track})
        profile = fusion_summary.get(tid_int, {})

        cur.execute(
            """INSERT INTO bird_profiles (batch_id, track_id, resolved_tag, fusion_confidence,
               egg_count, avg_egg_weight_g, last_risk_score) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (batch_id, tid_int, profile.get("resolved_tag"), profile.get("confidence", 0),
             profile.get("egg_count", 0), profile.get("avg_egg_weight_g", 0), rr.risk_score))

    fleet_health = sum(1 for _ in behavior_report["tracks"]) and \
        round(sum(risk_engine.score({"tracks": {int(t): s}, "events": []}).risk_score
                  for t, s in behavior_report["tracks"].items()) / len(behavior_report["tracks"]), 1)
    cur.execute(
        """INSERT INTO risk_scores (batch_id, computed_at, health_score, behavior_score, risk_score, top_evidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (batch_id, str(n_frames), fleet_health, fleet_health, fleet_health, json.dumps([])))
    conn.commit()
    conn.close()

    print(f"Persistido en {os.path.join(os.path.dirname(__file__), 'flockid.db')}")
    print(f"batch_id = {batch_id}")
    print(f"Tracks: {len(seen_tracks)}  Eventos: {len(behavior_report['events'])}  "
          f"Perfiles de ave: {len(behavior_report['tracks'])}")
    return batch_id


if __name__ == "__main__":
    run_and_persist()
