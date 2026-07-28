"""
Multi-camera raw-detection fusion.

Several cameras observe the same space at the same time. This module
clusters near-duplicate detections from different cameras into a single
fused detection, confidence-weighted (a simple proxy for inverse noise
variance), before handing the result to a single MultiObjectTracker.

Known limitation: distance-based clustering assumes distinct animals are
rarely closer than `cluster_radius`. In dense scenes this can incorrectly
merge two different animals. Measured against a single well-covering
camera, this approach underperformed (see trajectory_association.py for
the per-camera-tracker + trajectory-matching approach that replaced it).
It is kept here as a documented baseline, not as the recommended design.
"""
import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment

from vision.detection.base import Detection


def fuse_multicam_detections(camera_detections: List[List[Detection]],
                              cluster_radius: float = 25.0) -> List[Detection]:
    """
    camera_detections: one Detection list per camera, same frame.
    cluster_radius: max world-space distance to consider two detections
        from different cameras the same animal.
    """
    all_dets: List[Detection] = []
    for cam_dets in camera_detections:
        all_dets.extend(cam_dets)
    if not all_dets:
        return []

    n = len(all_dets)
    used = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            d = np.hypot(all_dets[i].x - all_dets[j].x, all_dets[i].y - all_dets[j].y)
            if d <= cluster_radius:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)

    fused = []
    for cluster in clusters:
        dets = [all_dets[k] for k in cluster]
        weights = np.array([max(d.confidence, 0.05) for d in dets])
        weights = weights / weights.sum()
        fx = float(np.sum([d.x * w for d, w in zip(dets, weights)]))
        fy = float(np.sum([d.y * w for d, w in zip(dets, weights)]))
        avg_radius = float(np.mean([d.radius for d in dets]))
        n_cameras = len(dets)
        fused_confidence = float(np.clip(1.0 - np.prod([1 - d.confidence for d in dets]), 0, 1))
        fused.append(Detection(x=fx, y=fy, radius=avg_radius, area=np.pi * avg_radius ** 2,
                                confidence=fused_confidence, label=f"consensus_{n_cameras}cam"))
    return fused
