"""
Multi-camera consensus v2 -- correct architecture.

Each camera runs its own MultiObjectTracker (unmodified). The cross-
camera association engine links identities via accumulated trajectory
evidence. A GLOBAL position is built per frame (averaging all cameras
that currently agree on the same global_id) and evaluated with the same
honest methodology used throughout the rest of the project.
"""
import numpy as np
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.tracking.tracker import MultiObjectTracker
from vision.evaluation.metrics import FrameResult, compute_metrics
from vision.multicam.camera_model import default_corner_cameras
from vision.multicam.trajectory_association import CrossCameraAssociator


def run_multi_camera_v2(n_frames: int, seed: int, match_distance: float = 15.0,
                         min_obs_to_merge: int = 20):
    sim = RandomWalkSimulator(seed=seed, n_grains=15)
    rng = np.random.default_rng(seed + 1)
    cameras = default_corner_cameras(sim.width, sim.height, rng)
    trackers = {cam.spec.camera_id: MultiObjectTracker(use_appearance=False) for cam in cameras}
    associator = CrossCameraAssociator(match_distance=match_distance,
                                        min_observations_to_merge=min_obs_to_merge)

    frame_results = []
    for frame_idx in range(n_frames):
        _, gt = sim.step()

        camera_track_positions = {}
        for cam in cameras:
            cam_dets = []
            for bird_id, (x, y, r) in gt.items():
                d = cam.observe(x, y, r)
                if d is not None:
                    cam_dets.append(d)
            tracks = trackers[cam.spec.camera_id].update(cam_dets, frame_idx)
            for tid, t in tracks.items():
                camera_track_positions[(cam.spec.camera_id, tid)] = (t.x, t.y)

        associator.observe(camera_track_positions)

        # build the global position: average all (camera, track) pairs
        # that resolved to the same global_id this frame
        global_positions = {}
        global_counts = {}
        for (cam_id, tid), (x, y) in camera_track_positions.items():
            gid = associator.global_id(cam_id, tid)
            if gid not in global_positions:
                global_positions[gid] = [0.0, 0.0]
                global_counts[gid] = 0
            global_positions[gid][0] += x
            global_positions[gid][1] += y
            global_counts[gid] += 1

        pred = {}
        for gid, (sx, sy) in global_positions.items():
            n = global_counts[gid]
            # use a stable numeric id derived from the global_id string
            numeric_id = abs(hash(gid)) % 1_000_000
            pred[numeric_id] = (sx / n, sy / n, 10.0)

        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))

    metrics = compute_metrics(frame_results)
    metrics["n_camera_merges"] = associator.n_merges()
    return metrics


if __name__ == "__main__":
    from multicam_demo import run_single_camera

    n_frames, seed = 600, 42
    print("Running: 1 camera (baseline)...")
    single = run_single_camera(n_frames, seed)

    print("Running: 4 cameras, v2 architecture (trajectory + Union-Find)...")
    multi_v2 = run_multi_camera_v2(n_frames, seed)

    print("\n=== 1 camera vs. 4 cameras (correct architecture) ===\n")
    header = f"{'metric':<26}{'1 camera':>14}{'4 cams v2':>14}"
    print(header)
    print("-" * len(header))
    for k in single:
        v2 = multi_v2.get(k, "-")
        print(f"{k:<26}{single[k]:>14}{v2:>14}")
    print(f"\nConfirmed cross-camera merges: {multi_v2['n_camera_merges']}")
