"""
Demo end-to-end: Detector -> Tracker -> Behavior Engine.

Runs on BOTH synthetic scenarios (conveyor and random motion) to show
the Behavior Engine is agnostic to whichever tracker/scene feeds it --
it only needs a TrackState per frame.
"""
import json
import os
import cv2
import numpy as np

from vision.synth.generator import ConveyorSimulator
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.behavior.engine import BehaviorEngine
from vision.behavior.zones import Zone


def run(sim, n_frames: int, zones, out_prefix: str, out_dir: str):
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    engine = BehaviorEngine(zones=zones, frame_size=(sim.width, sim.height))

    for frame_idx in range(n_frames):
        frame, gt = sim.step()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        engine.process_frame(frame_idx, tracks)

    report = engine.summary()
    with open(os.path.join(out_dir, f"{out_prefix}_behavior_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    heatmap = engine.heatmap_image()
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_heatmap.png"), heatmap)

    return report


if __name__ == "__main__":
    out_dir = "demo_out"
    os.makedirs(out_dir, exist_ok=True)
    n_frames = 400

    # Example zones: left/right half of the frame, purely to
    # demonstrate the zone-occupancy mechanism -- these do NOT represent
    # real feeders/waterers (those don't exist in the grain simulator).
    conveyor_zones = [
        Zone("zone_A_left", 0, 0, 480, 540),
        Zone("zone_B_right", 480, 0, 960, 540),
    ]
    random_walk_zones = [
        Zone("zone_A_left", 0, 0, 480, 540),
        Zone("zone_B_right", 480, 0, 960, 540),
    ]

    print("Running Behavior Engine on Phase 1 (conveyor)...")
    report_conveyor = run(ConveyorSimulator(seed=42), n_frames, conveyor_zones, "conveyor", out_dir)

    print("Running Behavior Engine on Phase 2 (random motion)...")
    report_random = run(RandomWalkSimulator(seed=42, n_grains=25), n_frames, random_walk_zones, "random_walk", out_dir)

    print("\n=== Phase 1 summary (conveyor) ===")
    print(f"Tracks observed: {len(report_conveyor['tracks'])}")
    print(f"Isolation events: {report_conveyor['n_isolation_events']}")
    print(f"Low activity events: {report_conveyor['n_low_activity_events']}")

    print("\n=== Phase 2 summary (random motion) ===")
    print(f"Tracks observed: {len(report_random['tracks'])}")
    print(f"Isolation events: {report_random['n_isolation_events']}")
    print(f"Low activity events: {report_random['n_low_activity_events']}")

    # strongest "repetitive movement" example in each scenario
    def top_repetitive(report, n=3):
        items = sorted(report["tracks"].items(), key=lambda kv: -kv[1]["repetitiveness_score"])
        return items[:n]

    print("\nTop repetitiveness (conveyor):", [(tid, r["repetitiveness_score"]) for tid, r in top_repetitive(report_conveyor)])
    print("Top repetitiveness (random):", [(tid, r["repetitiveness_score"]) for tid, r in top_repetitive(report_random)])
