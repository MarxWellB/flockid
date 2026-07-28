"""
Phase 2 stress test: run the CURRENT tracker (unmodified, the same one
already calibrated for the conveyor) on non-linear motion, and compare
against its Phase 1 performance. The goal is to measure how much the
constant-velocity assumption degrades, not to fix it yet.
"""
import json
from vision.synth.generator import ConveyorSimulator
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.detection.blob_detector import BlobDetector
from vision.evaluation.metrics import FrameResult, compute_metrics
from vision.reid.appearance import extract_batch
from vision.tracking.tracker import MultiObjectTracker


def run_scenario(sim, n_frames: int, use_appearance: bool, tracker_kwargs=None):
    tracker_kwargs = tracker_kwargs or {}
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=use_appearance, **tracker_kwargs)
    frame_results = []
    for frame_idx in range(n_frames):
        frame, gt = sim.step()
        detections = detector.detect(frame)
        embeddings = extract_batch(frame, detections) if use_appearance else None
        tracks = tracker.update(detections, frame_idx, embeddings)
        pred = {tid: (t.x, t.y, t.radius) for tid, t in tracks.items()}
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


if __name__ == "__main__":
    n_frames = 400
    seed = 42

    results = {}

    results["phase1_conveyor_no_reid"] = run_scenario(
        ConveyorSimulator(seed=seed), n_frames, use_appearance=False)
    results["phase1_conveyor_with_reid"] = run_scenario(
        ConveyorSimulator(seed=seed), n_frames, use_appearance=True)

    results["phase2_random_no_reid_default"] = run_scenario(
        RandomWalkSimulator(seed=seed, n_grains=25), n_frames, use_appearance=False)
    results["phase2_random_with_reid_default"] = run_scenario(
        RandomWalkSimulator(seed=seed, n_grains=25), n_frames, use_appearance=True)

    print("=== Phase 2 - stress test: unmodified tracker under non-linear motion ===\n")
    header = f"{'metric':<26}" + "".join(f"{k:>34}" for k in results)
    print(header)
    print("-" * len(header))
    metric_keys = list(next(iter(results.values())).keys())
    for mk in metric_keys:
        row = f"{mk:<26}" + "".join(f"{results[k][mk]:>34}" for k in results)
        print(row)

    with open("demo_out/phase2_stress_test.json", "w") as f:
        json.dump(results, f, indent=2)
