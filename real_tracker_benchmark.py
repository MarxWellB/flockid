"""
Real benchmark: our MultiObjectTracker (Kalman+Mahalanobis+prior) vs the
REAL implementations of ByteTrack, BoT-SORT, OC-SORT, and DeepSORT (from
the `ultralytics` and `deep-sort-realtime` libraries, not a theoretical
table comparison). Same detector, same frames, same seed.
"""
import numpy as np
from ultralytics.cfg import get_cfg
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.oc_sort import OCSORT
from deep_sort_realtime.deepsort_tracker import DeepSort

from vision.synth.generator import ConveyorSimulator
from vision.synth.random_walk_generator import RandomWalkSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.evaluation.metrics import FrameResult, compute_metrics

BYTETRACK_CFG = "/usr/local/lib/python3.12/dist-packages/ultralytics/cfg/trackers/bytetrack.yaml"
BOTSORT_CFG = "/usr/local/lib/python3.12/dist-packages/ultralytics/cfg/trackers/botsort.yaml"
OCSORT_CFG = "/usr/local/lib/python3.12/dist-packages/ultralytics/cfg/trackers/ocsort.yaml"


def run_deepsort(sim, n_frames, max_age=12):
    detector = BlobDetector()
    ds = DeepSort(max_age=max_age, embedder=None, n_init=2)
    frame_results = []
    for frame_idx in range(n_frames):
        frame_img, gt = sim.step()
        detections = detector.detect(frame_img)
        raw_dets = [([d.x - d.radius, d.y - d.radius, d.radius * 2, d.radius * 2],
                     max(d.confidence, 0.5), "grain") for d in detections]
        embeds = [np.ones(4) for _ in raw_dets]  # constant (not zero, avoids NaN in normalization) -- no real appearance, same condition as ByteTrack/BoT-SORT/OC-SORT here
        tracks = ds.update_tracks(raw_dets, embeds=embeds) if raw_dets else ds.update_tracks([], embeds=[])
        pred = {}
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, top, w, h = t.to_ltwh()
            pred[int(t.track_id)] = (l + w / 2, top + h / 2, w / 2)
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


class SimpleResults:
    """Minimal adapter: our Detection objects -> the Results-like format
    BYTETracker/BOTSORT expect (xywh, conf, cls attributes + indexing)."""
    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return SimpleResults(self.xywh[mask], self.conf[mask], self.cls[mask])


def detections_to_results(detections):
    if not detections:
        return SimpleResults(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    xywh = np.array([[d.x, d.y, d.radius * 2, d.radius * 2] for d in detections])
    conf = np.array([max(d.confidence, 0.5) for d in detections])  # BlobDetector doesn't produce a real score
    cls = np.zeros(len(detections))
    return SimpleResults(xywh, conf, cls)


def run_ultralytics_tracker(sim, n_frames, tracker_cfg_path, tracker_cls, frame_size):
    detector = BlobDetector()
    cfg = get_cfg(tracker_cfg_path)
    tracker = tracker_cls(cfg)
    frame_results = []
    for frame_idx in range(n_frames):
        frame_img, gt = sim.step()
        detections = detector.detect(frame_img)
        results = detections_to_results(detections)
        out = tracker.update(results)  # [x1,y1,x2,y2,track_id,score,cls,idx]
        pred = {}
        for row in out:
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
            pred[tid] = ((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2)
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


def run_our_tracker(sim, n_frames):
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    frame_results = []
    for frame_idx in range(n_frames):
        frame_img, gt = sim.step()
        detections = detector.detect(frame_img)
        tracks = tracker.update(detections, frame_idx)
        pred = {tid: (t.x, t.y, t.radius) for tid, t in tracks.items()}
        frame_results.append(FrameResult(frame_idx=frame_idx, gt=dict(gt), pred=pred))
    return compute_metrics(frame_results)


def run_all(scenario_name, sim_factory, n_frames, seed):
    print(f"\n--- Scenario: {scenario_name} ---")
    results = {}
    results["our_tracker"] = run_our_tracker(sim_factory(seed), n_frames)
    results["ByteTrack_real"] = run_ultralytics_tracker(
        sim_factory(seed), n_frames, BYTETRACK_CFG, BYTETracker, None)
    results["BoT-SORT_real"] = run_ultralytics_tracker(
        sim_factory(seed), n_frames, BOTSORT_CFG, BOTSORT, None)
    results["OC-SORT_real"] = run_ultralytics_tracker(
        sim_factory(seed), n_frames, OCSORT_CFG, OCSORT, None)
    results["DeepSORT_real"] = run_deepsort(sim_factory(seed), n_frames)

    header = f"{'metric':<26}" + "".join(f"{k:>18}" for k in results)
    print(header)
    print("-" * len(header))
    for mk in results["our_tracker"]:
        print(f"{mk:<26}" + "".join(f"{results[k][mk]:>18}" for k in results))
    return results


if __name__ == "__main__":
    n_frames, seed = 400, 42
    all_results = {}
    all_results["conveyor"] = run_all("Conveyor (Phase 1)", lambda s: ConveyorSimulator(seed=s), n_frames, seed)
    all_results["random"] = run_all("Random motion (Phase 2)",
                                        lambda s: RandomWalkSimulator(seed=s, n_grains=25), n_frames, seed)

    import json
    with open("demo_out/real_tracker_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
