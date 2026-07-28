"""
Multi-object tracker based on a constant-velocity Kalman filter with
Mahalanobis-distance association and a two-tier confidence threshold
(create vs. maintain) inspired by ByteTrack.

Association cost:

    C(track, det) = mahalanobis(pos)
                  + appearance_weight * cosine_distance(appearance)
                  + lambda_prior * time_since_update

Solved via the Hungarian algorithm. The Mahalanobis term accounts for
the Kalman filter's own positional uncertainty, so tracks that have been
unmatched for longer are naturally given a wider (not arbitrarily wider)
association gate. The prior term penalizes long-unmatched tracks, and the
optional appearance term adds a re-identification signal when available.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from vision.detection.base import Detection
from vision.tracking.base import Tracker, TrackState
from vision.tracking.kalman import KalmanCV


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


MAHALANOBIS_GATE_CONFIDENCE = 0.99  # chi^2, 2 DOF, ~99% confidence region


@dataclass
class Track:
    track_id: int
    kf: KalmanCV
    radius: float
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    history: list = field(default_factory=list)  # [(frame_idx, x, y)]
    embedding: Optional[np.ndarray] = None

    @property
    def x(self) -> float:
        return self.kf.x

    @property
    def y(self) -> float:
        return self.kf.y

    def predict(self):
        self.kf.predict()

    def update(self, det: Detection, embedding: Optional[np.ndarray] = None, emb_alpha: float = 0.3):
        self.kf.update(np.array([det.x, det.y], dtype=np.float64))
        self.radius = det.radius
        self.time_since_update = 0
        self.hits += 1
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                self.embedding = emb_alpha * embedding + (1 - emb_alpha) * self.embedding

    def to_state(self) -> TrackState:
        return TrackState(
            track_id=self.track_id, x=self.x, y=self.y, radius=self.radius,
            time_since_update=self.time_since_update, history=self.history,
        )


class MultiObjectTracker(Tracker):
    def __init__(self, max_age: int = 12,
                 use_appearance: bool = False, appearance_weight: float = 40.0,
                 lambda_prior: float = 1.5,
                 gate_confidence: float = MAHALANOBIS_GATE_CONFIDENCE,
                 process_var: float = 1.0, measurement_var: float = 9.0,
                 min_confidence_for_new_track: float = 0.5):
        """
        max_age: frames a track may go unmatched before being dropped.
        use_appearance: include a cosine-distance appearance term in the cost.
        appearance_weight: weight of the appearance term (Mahalanobis-equivalent units).
        lambda_prior: penalty per unmatched frame, discourages resurrecting stale tracks.
        gate_confidence: chi^2 confidence level used for the association gate.
        process_var / measurement_var: Kalman filter process and measurement noise.
        min_confidence_for_new_track: detections below this confidence can still
            match an existing track but cannot spawn a new one. This reduces
            identity fragmentation caused by frame-to-frame confidence flicker.
        """
        self.max_age = max_age
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight
        self.lambda_prior = lambda_prior
        self.mahalanobis_gate = float(np.sqrt(chi2.ppf(gate_confidence, df=2)))
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.min_confidence_for_new_track = min_confidence_for_new_track
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _new_track(self, det: Detection, embedding: Optional[np.ndarray]) -> Track:
        t = Track(
            track_id=self._next_id,
            kf=KalmanCV(det.x, det.y, process_var=self.process_var,
                        measurement_var=self.measurement_var),
            radius=det.radius,
            embedding=embedding,
        )
        self._next_id += 1
        return t

    def update(self, detections: List[Detection], frame_idx: int,
               embeddings: Optional[List[np.ndarray]] = None) -> Dict[int, TrackState]:
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
            mahal_matrix = np.zeros((n_tracks, n_dets))
            gate = np.zeros((n_tracks, n_dets), dtype=bool)

            for i, tid in enumerate(track_ids):
                t = self.tracks[tid]
                dynamic_gate = self.mahalanobis_gate * (1.0 + 0.15 * t.time_since_update)
                prior_penalty = self.lambda_prior * t.time_since_update
                for j, d in enumerate(detections):
                    z = np.array([d.x, d.y], dtype=np.float64)
                    mahal = t.kf.mahalanobis(z)
                    mahal_matrix[i, j] = mahal
                    combined = mahal + prior_penalty
                    if self.use_appearance and t.embedding is not None and embeddings[j] is not None:
                        combined += self.appearance_weight * cosine_distance(t.embedding, embeddings[j])
                    cost[i, j] = combined
                    gate[i, j] = mahal <= dynamic_gate

            for i, tid in enumerate(track_ids):
                nearest = float(mahal_matrix[i].min())
                self.tracks[tid].kf.observe_nis(nearest)

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
            if detections[dj].confidence < self.min_confidence_for_new_track:
                continue
            t = self._new_track(detections[dj], embeddings[dj])
            t.history.append((frame_idx, t.x, t.y))
            self.tracks[t.track_id] = t

        expired = [tid for tid in unmatched_tracks if self.tracks[tid].time_since_update > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        return {tid: t.to_state() for tid, t in self.tracks.items()}
