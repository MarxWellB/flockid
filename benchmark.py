"""
Benchmark: Euclidean baseline vs Kalman+Mahalanobis+prior tracker.

Runs all 4 combinations (euclidean/kalman x no-reid/with-reid) on the
SAME seed and detections, to isolate the effect of each improvement.
"""
import json
from vision.synth.generator import ConveyorSimulator
from vision.detection.blob_detector import BlobDetector
from vision.evaluation.metrics import FrameResult, compute_metrics
from vision.reid.appearance import extract_batch

from vision.tracking.legacy_euclidean_tracker import EuclideanSortTracker
from vision.tracking.tracker import MultiObjectTracker as KalmanTracker


def run_one(tracker_factory, use_appearance: bool, n_frames: int = 400, seed: int = 42):
    sim = ConveyorSimulator(seed=seed)
    detector = BlobDetector()
    tracker = tracker_factory(use_appearance)
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
    configs = {
        "euclidean_no_reid": (lambda a: EuclideanSortTracker(max_age=12, max_distance=45.0, use_appearance=a), False),
        "euclidean_with_reid": (lambda a: EuclideanSortTracker(max_age=12, max_distance=45.0, use_appearance=a), True),
        "kalman_no_reid": (lambda a: KalmanTracker(max_age=12, use_appearance=a), False),
        "kalman_with_reid": (lambda a: KalmanTracker(max_age=12, use_appearance=a), True),
    }

    results = {}
    for name, (factory, use_app) in configs.items():
        print(f"Running: {name} ...")
        results[name] = run_one(factory, use_app)

    print("\n=== Benchmark: Euclidean vs Kalman+Mahalanobis+prior (seed=42, 400 frames) ===\n")
    header = f"{'metric':<26}" + "".join(f"{k:>22}" for k in results)
    print(header)
    print("-" * len(header))
    metric_keys = list(next(iter(results.values())).keys())
    for mk in metric_keys:
        row = f"{mk:<26}" + "".join(f"{results[k][mk]:>22}" for k in results)
        print(row)

    with open("demo_out/benchmark_report.json", "w") as f:
        json.dump(results, f, indent=2)
