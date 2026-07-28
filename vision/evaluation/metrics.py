"""
Multi-object tracking identity metrics, computed against known ground
truth (used with the synthetic simulator): ID switches, track
fragmentation, coverage/recall, and a simplified IDF1. Not a
reimplementation of py-motmetrics, but measures the key quantity for
identity persistence: how often a tracked identity switches.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np


@dataclass
class FrameResult:
    frame_idx: int
    gt: Dict[int, Tuple[float, float, float]]    # gt_id -> (x, y, r)
    pred: Dict[int, Tuple[float, float, float]]   # pred_id -> (x, y, r)


def match_frame(gt, pred, max_distance=20.0):
    """Greedy nearest-distance matching between gt_id and pred_id."""
    pairs = []
    used_pred = set()
    gt_items = list(gt.items())
    for gid, (gx, gy, _) in gt_items:
        best_pid, best_d = None, max_distance
        for pid, (px, py, _) in pred.items():
            if pid in used_pred:
                continue
            d = np.hypot(gx - px, gy - py)
            if d < best_d:
                best_d, best_pid = d, pid
        if best_pid is not None:
            pairs.append((gid, best_pid))
            used_pred.add(best_pid)
    return pairs


def compute_metrics(frame_results: List[FrameResult]) -> dict:
    gt_to_pred_history: Dict[int, List[int]] = defaultdict(list)
    gt_lifespan: Dict[int, int] = Counter()
    gt_covered_frames: Dict[int, int] = Counter()

    id_switches = 0
    prev_pred_for_gt: Dict[int, int] = {}

    total_gt_instances = 0
    matched_instances = 0

    for fr in frame_results:
        pairs = match_frame(fr.gt, fr.pred)

        for gid in fr.gt:
            gt_lifespan[gid] += 1
            total_gt_instances += 1

        for gid, pid in pairs:
            matched_instances += 1
            gt_covered_frames[gid] += 1
            gt_to_pred_history[gid].append(pid)
            if gid in prev_pred_for_gt and prev_pred_for_gt[gid] != pid:
                id_switches += 1
            prev_pred_for_gt[gid] = pid

    fragmentation = {gid: len(set(hist)) for gid, hist in gt_to_pred_history.items()}
    avg_fragmentation = float(np.mean(list(fragmentation.values()))) if fragmentation else 0.0

    coverage = matched_instances / total_gt_instances if total_gt_instances else 0.0

    # Simplified IDF1: each real object's dominant predicted id counts as correct.
    idtp = 0
    for gid, hist in gt_to_pred_history.items():
        if not hist:
            continue
        _, dominant_count = Counter(hist).most_common(1)[0]
        idtp += dominant_count
    idf1 = idtp / total_gt_instances if total_gt_instances else 0.0

    return {
        "total_gt_instances": total_gt_instances,
        "matched_instances": matched_instances,
        "coverage_recall": round(coverage, 4),
        "id_switches": id_switches,
        "unique_gt_objects": len(gt_lifespan),
        "avg_track_fragmentation": round(avg_fragmentation, 3),
        "idf1_simplified": round(idf1, 4),
    }
