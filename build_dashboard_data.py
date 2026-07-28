"""
Consolidates REAL data (already computed by the Behavior Engine, Risk
Engine, Identity Fusion Engine, and nest simulator) into a single JSON to
feed the dashboard. No numbers are invented -- scores are recomputed per
bird by reusing the same already-validated RiskEngineV1 logic, just
applied per track instead of at the population level.
"""
import json
import numpy as np
from vision.synth.nest_simulator import NestSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.behavior.engine import BehaviorEngine
from vision.behavior.zones import Zone
from vision.risk.engine import RiskEngineV1
from vision.fusion.identity_fusion import IdentityFusionEngine


def per_bird_risk(track_id, track_stats, events_for_track, engine: RiskEngineV1):
    fake_report = {
        "tracks": {track_id: track_stats},
        "events": events_for_track,
    }
    report = engine.score(fake_report)
    return report


def main():
    n_frames = 1800
    n_birds = 15
    sim = NestSimulator(n_birds=n_birds, seed=7)
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    zones = [Zone(n.nest_id, n.x - n.radius, n.y - n.radius, n.x + n.radius, n.y + n.radius) for n in sim.nests]
    behavior = BehaviorEngine(zones=zones, frame_size=(sim.width, sim.height),
                               low_activity_window=30, low_activity_threshold=0.3)
    fusion = IdentityFusionEngine(min_observations_to_resolve=3)

    last_positions = {}
    for frame_idx in range(n_frames):
        frame, gt, rfid_reads, egg_events = sim.step()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        behavior.process_frame(frame_idx, tracks)
        track_positions = {tid: (t.x, t.y) for tid, t in tracks.items()}
        fusion.observe(frame_idx, track_positions, rfid_reads, egg_events, sim.nests)
        last_positions = track_positions

    behavior_report = behavior.summary()
    fusion_summary = fusion.summary()
    risk_engine = RiskEngineV1()

    birds_out = []
    for tid, stats in behavior_report["tracks"].items():
        tid_int = int(tid) if isinstance(tid, str) else tid
        events_for_track = [e for e in behavior_report["events"] if e["entity_id"] == tid_int]
        rr = per_bird_risk(tid_int, stats, events_for_track, risk_engine)
        profile = fusion_summary.get(tid_int, {})
        hist = behavior.position_history.get(tid_int, [])
        pos = (hist[-1][1], hist[-1][2]) if hist else (None, None)
        birds_out.append({
            "track_id": tid_int,
            "x": round(pos[0], 1) if pos[0] is not None else None,
            "y": round(pos[1], 1) if pos[1] is not None else None,
            "resolved_tag": profile.get("resolved_tag"),
            "fusion_confidence": profile.get("confidence", 0),
            "egg_count": profile.get("egg_count", 0),
            "avg_egg_weight_g": profile.get("avg_egg_weight_g", 0),
            "avg_speed": stats["avg_speed_px_per_frame"],
            "repetitiveness": stats["repetitiveness_score"],
            "frames_tracked": stats["frames_tracked"],
            "risk_score": rr.risk_score,
            "behavior_score": rr.behavior_score,
            "top_evidence": rr.top_evidence,
        })

    birds_out.sort(key=lambda b: -b["risk_score"])
    # filter out tracks with too little history to be a useful profile --
    # the tracker fragments identity under occlusion and erratic motion,
    # and there's no point showing a dashboard "bird" that was only seen
    # for 5 frames. Keep sufficiently long profiles, the way a real
    # quality filter would before showing this to a user.
    birds_out = [b for b in birds_out if b["frames_tracked"] >= 80][:24]

    nests_out = [{"nest_id": n.nest_id, "x": n.x, "y": n.y, "radius": n.radius} for n in sim.nests]

    fleet_risk = float(np.mean([b["risk_score"] for b in birds_out])) if birds_out else 0
    total_eggs = sum(b["egg_count"] for b in birds_out)

    out = {
        "meta": {
            "n_birds": n_birds, "n_frames": n_frames,
            "width": sim.width, "height": sim.height,
            "fleet_avg_risk": round(fleet_risk, 1),
            "total_eggs": total_eggs,
            "n_isolation_events": behavior_report["n_isolation_events"],
            "n_low_activity_events": behavior_report["n_low_activity_events"],
        },
        "birds": birds_out,
        "nests": nests_out,
        "events": sorted(behavior_report["events"], key=lambda e: -e["frame_idx"])[:30],
    }

    with open("demo_out/dashboard_data.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Birds: {len(birds_out)}  |  Fleet average risk: {fleet_risk:.1f}")
    print(f"Top 3 risk: {[(b['track_id'], b['risk_score']) for b in birds_out[:3]]}")


if __name__ == "__main__":
    main()
