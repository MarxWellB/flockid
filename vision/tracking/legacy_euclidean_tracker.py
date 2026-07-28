"""
Baseline SORT-style tracker (Euclidean distance, exponential velocity
smoothing) kept as a reference point for comparison against the
Kalman+Mahalanobis tracker (see MultiObjectTracker in tracker.py).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.detection.blob_detector import Detection


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    history: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def predict(self):
        self.x += self.vx
        self.y += self.vy

    def update(self, det: Detection, embedding: Optional[np.ndarray] = None,
               alpha: float = 0.5, emb_alpha: float = 0.3):
        new_vx = det.x - self.x
        new_vy = det.y - self.y
        self.vx = alpha * new_vx + (1 - alpha) * self.vx
        self.vy = alpha * new_vy + (1 - alpha) * self.vy
        self.x, self.y = det.x, det.y
        self.radius = det.radius
        self.time_since_update = 0
        self.hits += 1
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                self.embedding = emb_alpha * embedding + (1 - emb_alpha) * self.embedding


class EuclideanSortTracker:
    def __init__(self, max_age: int = 12, max_distance: float = 45.0,
                 use_appearance: bool = False, appearance_weight: float = 40.0):
        """
        max_age: frames a track survives without a matched detection.
        max_distance: base gate distance (px), relaxed for long-unmatched tracks.
        use_appearance: combine position distance with a cosine appearance term.
        appearance_weight: weight of the appearance term (pixel-equivalent units).
        """
        self.max_age = max_age
        self.max_distance = max_distance
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _new_track(self, det: Detection, embedding: Optional[np.ndarray]) -> Track:
        t = Track(track_id=self._next_id, x=det.x, y=det.y, vx=0.0, vy=0.0,
                   radius=det.radius, embedding=embedding)
        self._next_id += 1
        return t

    def update(self, detections: List[Detection], frame_idx: int,
               embeddings: Optional[List[np.ndarray]] = None) -> Dict[int, Track]:
        if embeddings is None:
            embeddings = [None] * len(detections)

        for t in self.tracks.values():
            t.predict()
            t.age += 1
            t.time_since_update += 1

        track_ids = list(self.tracks.keys())
        n_tracks, n_dets = len(track_ids), len(detections)

        matches = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(n_dets))

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets))
            gate = np.zeros((n_tracks, n_dets), dtype=bool)
            for i, tid in enumerate(track_ids):
                t = self.tracks[tid]
                dynamic_gate = self.max_distance * (1.0 + 0.5 * t.time_since_update)
                for j, d in enumerate(detections):
                    pos_dist = np.hypot(t.x - d.x, t.y - d.y)
                    combined = pos_dist
                    if self.use_appearance and t.embedding is not None and embeddings[j] is not None:
                        combined = pos_dist + self.appearance_weight * cosine_distance(t.embedding, embeddings[j])
                    cost[i, j] = combined
                    gate[i, j] = pos_dist <= dynamic_gate

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if gate[r, c]:
                    matches.append((track_ids[r], c))
                    unmatched_tracks.discard(track_ids[r])
                    unmatched_dets.discard(c)

        for tid, dj in matches:
            self.tracks[tid].update(detections[dj], embeddings[dj])
            self.tracks[tid].history.append((frame_idx, self.tracks[tid].x, self.tracks[tid].y))

        for dj in unmatched_dets:
            t = self._new_track(detections[dj], embeddings[dj])
            t.history.append((frame_idx, t.x, t.y))
            self.tracks[t.track_id] = t

        expired = [tid for tid in unmatched_tracks if self.tracks[tid].time_since_update > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        return self.tracks
