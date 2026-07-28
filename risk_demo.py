"""
End-to-end demo: Detector -> Tracker -> Behavior Engine -> Risk Engine.

Minimum validity check (before trusting the score): run a "normal"
scenario and a "degraded" one (more isolation, more stillness) and
confirm risk_score actually rises for the degraded case. If it didn't,
the score would be worthless -- this isn't assumed, it's measured.
"""
import json
import os

from vision.synth.generator import ConveyorSimulator
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.behavior.engine import BehaviorEngine
from vision.behavior.zones import Zone
from vision.risk.engine import RiskEngineV1


def run_pipeline(sim, n_frames: int, zones):
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    engine = BehaviorEngine(zones=zones, frame_size=(sim.width, sim.height))
    for frame_idx in range(n_frames):
        frame, gt = sim.step()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        engine.process_frame(frame_idx, tracks)
    return engine.summary()


if __name__ == "__main__":
    out_dir = "demo_out"
    os.makedirs(out_dir, exist_ok=True)
    n_frames = 400
    zones = [Zone("zone_A_left", 0, 0, 480, 540), Zone("zone_B_right", 480, 0, 960, 540)]
    risk_engine = RiskEngineV1()

    scenarios = {
        "conveyor_normal": ConveyorSimulator(seed=42),
        "random_normal": RandomWalkSimulator(seed=42, n_grains=25),
        # "Degraded" scenario: much stiller (high stop_prob and duration,
        # low speed while moving) and more spread out (fewer close
        # neighbors) -- the most honest proxy that can be built without
        # real bird data to simulate "population with reduced activity
        # and more isolation," exactly what the Behavior Engine measures.
        "random_degraded": RandomWalkSimulator(
            seed=42, n_grains=25, speed=1.2, stop_prob=0.08,
            stop_duration_range=(40, 90)),
    }

    results = {}
    for name, sim in scenarios.items():
        print(f"Running scenario: {name} ...")
        behavior_report = run_pipeline(sim, n_frames, zones)
        risk_report = risk_engine.score(behavior_report)
        results[name] = {
            "risk_report": vars(risk_report),
            "n_isolation_events": behavior_report["n_isolation_events"],
            "n_low_activity_events": behavior_report["n_low_activity_events"],
        }

    print("\n=== Risk score comparison across scenarios ===\n")
    header = f"{'scenario':<22}{'health':>10}{'behavior':>10}{'risk':>10}{'isolation':>10}{'low act.':>12}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        rr = r["risk_report"]
        print(f"{name:<22}{rr['health_score']:>10}{rr['behavior_score']:>10}{rr['risk_score']:>10}"
              f"{r['n_isolation_events']:>10}{r['n_low_activity_events']:>12}")

    print("\nTop evidence per scenario:")
    for name, r in results.items():
        print(f"  {name}: {r['risk_report']['top_evidence']}")

    with open(os.path.join(out_dir, "risk_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Validity check: the degraded scenario should have the highest risk_score.
    scores = {k: v["risk_report"]["risk_score"] for k, v in results.items()}
    worst = max(scores, key=scores.get)
    print(f"\nScenario with highest risk_score: {worst} ({scores[worst]})")
    if worst == "random_degraded":
        print("Score correctly identifies the degraded scenario.")
    else:
        print("Score did NOT identify the degraded scenario as highest risk -- check weights/thresholds.")
