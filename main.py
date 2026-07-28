"""
FlockID - MVP entry point
Full pipeline: synthetic simulation -> detection -> tracking -> metrics -> demo video.

Usage:
    python main.py --frames 400 --out /mnt/user-data/outputs

This script demonstrates Phases 1-3 of the design:
    Phase 1: Detection (blob detector, YOLO placeholder)
    Phase 2: Multi-object tracking with identity preservation
    Phase 3: Identity metrics (ID switches, fragmentation, IDF1)

No physical camera or belt required -- uses a simulator that reproduces
the hard conditions (crossings, occlusions) described in the design doc.
"""
import argparse
import json
import os
import cv2
import numpy as np

from vision.synth.generator import ConveyorSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.evaluation.metrics import FrameResult, compute_metrics
from vision.reid.appearance import extract_batch


def color_for_id(track_id: int):
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(c) for c in rng.integers(80, 255, size=3))


def run(n_frames: int, out_dir: str, seed: int = 42, use_appearance: bool = True,
        save_video: bool = True, video_name: str = "flockid_demo.mp4"):
    os.makedirs(out_dir, exist_ok=True)
    sim = ConveyorSimulator(seed=seed)
    detector = BlobDetector()
    tracker = MultiObjectTracker(max_age=12, use_appearance=use_appearance)

    video_path = os.path.join(out_dir, video_name)
    width, height = sim.width, sim.height
    writer = None
    if save_video:
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

    frame_results = []
    trajectories = {}  # track_id -> list of {"frame":.., "x":.., "y":..}

    for frame_idx in range(n_frames):
        frame, gt = sim.step()
        detections = detector.detect(frame)
        embeddings = extract_batch(frame, detections) if use_appearance else None
        tracks = tracker.update(detections, frame_idx, embeddings)

        pred = {tid: (t.x, t.y, t.radius) for tid, t in tracks.items()}
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))

        if save_video:
            vis = frame.copy()
            for tid, t in tracks.items():
                color = color_for_id(tid)
                cv2.circle(vis, (int(t.x), int(t.y)), int(t.radius) + 2, color, 2)
                cv2.putText(vis, f"ID {tid}", (int(t.x) - 10, int(t.y) - int(t.radius) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                trajectories.setdefault(tid, []).append(
                    {"frame": frame_idx, "x": round(t.x, 1), "y": round(t.y, 1)}
                )
                pts = t.history[-15:]
                for k in range(1, len(pts)):
                    p1 = (int(pts[k - 1][1]), int(pts[k - 1][2]))
                    p2 = (int(pts[k][1]), int(pts[k][2]))
                    cv2.line(vis, p1, p2, color, 1)

            cv2.putText(vis, f"Frame {frame_idx}  |  Active tracks: {len(tracks)}  |  ReID: {'ON' if use_appearance else 'OFF'}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(vis)

    if writer is not None:
        writer.release()

    metrics = compute_metrics(frame_results)
    return video_path if save_video else None, trajectories, metrics


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--out", type=str, default="/mnt/user-data/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true",
                         help="Run with and without appearance re-id and compare metrics")
    parser.add_argument("--no-appearance", action="store_true",
                         help="Single run, no appearance re-id (previous baseline behavior)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.compare:
        print("=== Running WITHOUT re-identification (baseline) ===")
        _, _, metrics_off = run(args.frames, args.out, args.seed,
                                 use_appearance=False, save_video=False)
        print("=== Running WITH appearance re-identification ===")
        video_path, trajectories, metrics_on = run(
            args.frames, args.out, args.seed, use_appearance=True,
            save_video=True, video_name="flockid_demo_reid.mp4")

        save_json(trajectories, os.path.join(args.out, "trajectories.json"))
        save_json({"no_reid": metrics_off, "with_reid": metrics_on},
                   os.path.join(args.out, "metrics_comparison.json"))

        print("\n=== Comparison (same seed, same frames) ===")
        header = f"{'metric':<28}{'no reid':>12}{'with reid':>12}"
        print(header)
        print("-" * len(header))
        for k in metrics_off:
            print(f"{k:<28}{metrics_off[k]:>12}{metrics_on[k]:>12}")
        print(f"\nVideo (with reid): {video_path}")
        print(f"Comparison report: {os.path.join(args.out, 'metrics_comparison.json')}")
    else:
        use_appearance = not args.no_appearance
        video_path, trajectories, metrics = run(args.frames, args.out, args.seed,
                                                  use_appearance=use_appearance)
        save_json(trajectories, os.path.join(args.out, "trajectories.json"))
        save_json(metrics, os.path.join(args.out, "metrics_report.json"))
        print(f"=== FlockID - metrics report (ReID {'ON' if use_appearance else 'OFF'}) ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print(f"\nVideo: {video_path}")
