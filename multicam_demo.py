"""
Direct test of the hypothesis: "several cameras watching the same
space keep the animal from being lost from view".

Runs the SAME scenario (same birds, same motion, same seed) with:
(a) a single camera (baseline)
(b) four corner cameras with consensus fusion

and compares coverage, ID switches, IDF1 -- the same honest methodology
used throughout the rest of the project.
"""
import numpy as np
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.tracking.tracker import MultiObjectTracker
from vision.evaluation.metrics import FrameResult, compute_metrics
from vision.detection.base import Detection
from vision.multicam.camera_model import default_corner_cameras, SyntheticCamera, CameraSpec
from vision.multicam.consensus import fuse_multicam_detections


def run_single_camera(n_frames: int, seed: int):
    sim = RandomWalkSimulator(seed=seed, n_grains=15)
    rng = np.random.default_rng(seed + 1)
    # a single, centered camera with the same noise/range model as the
    # corner ones -- a fair comparison, same sensor type
    cam = SyntheticCamera(CameraSpec("cam_single", sim.width / 2, sim.height / 2, max_range=900), rng)
    tracker = MultiObjectTracker(use_appearance=False)

    frame_results = []
    for frame_idx in range(n_frames):
        _, gt = sim.step()
        dets = []
        for bird_id, (x, y, r) in gt.items():
            d = cam.observe(x, y, r)
            if d is not None:
                dets.append(d)
        tracks = tracker.update(dets, frame_idx)
        pred = {tid: (t.x, t.y, t.radius) for tid, t in tracks.items()}
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


def run_multi_camera(n_frames: int, seed: int, cluster_radius: float = 25.0):
    sim = RandomWalkSimulator(seed=seed, n_grains=15)
    rng = np.random.default_rng(seed + 1)
    cameras = default_corner_cameras(sim.width, sim.height, rng)
    tracker = MultiObjectTracker(use_appearance=False)

    frame_results = []
    for frame_idx in range(n_frames):
        _, gt = sim.step()
        camera_detections = []
        for cam in cameras:
            cam_dets = []
            for bird_id, (x, y, r) in gt.items():
                d = cam.observe(x, y, r)
                if d is not None:
                    cam_dets.append(d)
            camera_detections.append(cam_dets)
        fused = fuse_multicam_detections(camera_detections, cluster_radius=cluster_radius)
        tracks = tracker.update(fused, frame_idx)
        pred = {tid: (t.x, t.y, t.radius) for tid, t in tracks.items()}
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


if __name__ == "__main__":
    n_frames = 600
    seed = 42

    print("Running: 1 camera (baseline)...")
    m_single = run_single_camera(n_frames, seed)
    print("Running: 4 cameras with consensus...")
    m_multi = run_multi_camera(n_frames, seed)

    print("\n=== Hypothesis: do 4 cameras with consensus beat 1 camera? ===\n")
    header = f"{'metric':<26}{'1 camera':>14}{'4 cameras':>14}"
    print(header)
    print("-" * len(header))
    for k in m_single:
        print(f"{k:<26}{m_single[k]:>14}{m_multi[k]:>14}")

    print(f"\nCoverage: {m_single['coverage_recall']:.1%} -> {m_multi['coverage_recall']:.1%}")
    print(f"ID switches: {m_single['id_switches']} -> {m_multi['id_switches']}")
    print(f"IDF1: {m_single['idf1_simplified']:.1%} -> {m_multi['idf1_simplified']:.1%}")
