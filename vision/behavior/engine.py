"""
Behavior Engine: consumes per-frame track positions (from any tracker
implementing the TrackState interface) and derives per-individual and
per-scene behavioral signals, emitting events with a consistent schema
(event_type, entity_id, confidence, evidence) for downstream consumption
by the Risk Engine.

Implemented signals: activity level, isolation (nearest-neighbor
distance), zone occupancy (feeding/watering/nesting), cumulative space
usage, movement repetitiveness (autocorrelation), and sustained low
activity. Gait/lameness and complex social-pattern signals are out of
scope here -- they require pose estimation and real training data.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from vision.behavior.zones import Zone
from vision.tracking.base import TrackState


@dataclass
class BehaviorEvent:
    event_type: str
    entity_id: int
    confidence: float
    evidence: dict
    frame_idx: int


def autocorrelation_repetitiveness(position_series: np.ndarray, max_lag: int = 30) -> Tuple[float, int]:
    """
    Returns (peak_strength, peak_lag) of the autocorrelation of velocity
    (the position series' first difference), excluding lag 0.

    Autocorrelation is computed on velocity rather than raw position:
    a straight-line-moving object's raw position has a near-linear trend,
    and any trending series produces spuriously high autocorrelation at
    almost every lag. Differencing removes the trend and leaves genuine
    oscillatory patterns (e.g. pacing back and forth).
    """
    velocity = np.diff(position_series)
    n = len(velocity)
    if n < max_lag * 2:
        return 0.0, 0
    velocity = velocity - velocity.mean()
    denom = np.dot(velocity, velocity)
    if denom < 1e-8:
        return 0.0, 0
    best_strength, best_lag = 0.0, 0
    for lag in range(2, min(max_lag, n // 2)):
        corr = np.dot(velocity[:-lag], velocity[lag:]) / denom
        if corr > best_strength:
            best_strength, best_lag = float(corr), lag
    return best_strength, best_lag


class BehaviorEngine:
    def __init__(self, zones: Optional[List[Zone]] = None,
                 frame_size: Tuple[int, int] = (960, 540),
                 isolation_radius: float = 80.0,
                 low_activity_threshold: float = 0.3,
                 low_activity_window: int = 30,
                 heatmap_cell_px: int = 10):
        self.zones = zones or []
        self.frame_size = frame_size
        self.isolation_radius = isolation_radius
        self.low_activity_threshold = low_activity_threshold
        self.low_activity_window = low_activity_window
        self.heatmap_cell_px = heatmap_cell_px

        h_cells = frame_size[1] // heatmap_cell_px + 1
        w_cells = frame_size[0] // heatmap_cell_px + 1
        self.heatmap = np.zeros((h_cells, w_cells), dtype=np.float64)

        self.zone_frames: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_frames_per_track: Dict[int, int] = defaultdict(int)
        self.position_history: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.events: List[BehaviorEvent] = []
        self._low_activity_streak: Dict[int, int] = defaultdict(int)
        self._low_activity_fired: set = set()
        self._isolation_streak: Dict[int, int] = defaultdict(int)
        self._isolation_fired: set = set()
        self.isolation_min_streak_frames = 15  # avoid firing on a single noisy frame

    def process_frame(self, frame_idx: int, tracks: Dict[int, TrackState]):
        positions = {tid: (t.x, t.y) for tid, t in tracks.items()}

        for tid, (x, y) in positions.items():
            self.total_frames_per_track[tid] += 1
            gy, gx = int(y // self.heatmap_cell_px), int(x // self.heatmap_cell_px)
            if 0 <= gy < self.heatmap.shape[0] and 0 <= gx < self.heatmap.shape[1]:
                self.heatmap[gy, gx] += 1
            for zone in self.zones:
                if zone.contains(x, y):
                    self.zone_frames[tid][zone.name] += 1
            self.position_history[tid].append((frame_idx, x, y))

        # Isolation: nearest-neighbor distance, debounced into sustained
        # episodes (one event per episode, not one per frame).
        ids = list(positions.keys())
        for tid in ids:
            if len(ids) <= 1:
                self._isolation_streak[tid] = 0
                continue
            x1, y1 = positions[tid]
            nn = min(np.hypot(x1 - positions[o][0], y1 - positions[o][1]) for o in ids if o != tid)
            if nn > self.isolation_radius:
                self._isolation_streak[tid] += 1
            else:
                self._isolation_streak[tid] = 0
                self._isolation_fired.discard(tid)
            if (self._isolation_streak[tid] >= self.isolation_min_streak_frames
                    and tid not in self._isolation_fired):
                self.events.append(BehaviorEvent(
                    event_type="isolation", entity_id=tid,
                    confidence=round(min(1.0, nn / (self.isolation_radius * 2)), 2),
                    evidence={"nearest_neighbor_distance_px": round(float(nn), 1),
                              "sustained_frames": self._isolation_streak[tid]},
                    frame_idx=frame_idx))
                self._isolation_fired.add(tid)

        # Sustained low activity: fires once when crossing the threshold, not every frame.
        for tid, hist in self.position_history.items():
            if len(hist) < self.low_activity_window:
                continue
            recent = hist[-self.low_activity_window:]
            total_dist = sum(
                np.hypot(recent[k][1] - recent[k - 1][1], recent[k][2] - recent[k - 1][2])
                for k in range(1, len(recent))
            )
            avg_speed = total_dist / len(recent)
            if avg_speed < self.low_activity_threshold:
                self._low_activity_streak[tid] += 1
            else:
                self._low_activity_streak[tid] = 0
                self._low_activity_fired.discard(tid)
            if self._low_activity_streak[tid] >= self.low_activity_window and tid not in self._low_activity_fired:
                self.events.append(BehaviorEvent(
                    event_type="low_activity", entity_id=tid, confidence=0.7,
                    evidence={"avg_speed_px_per_frame": round(float(avg_speed), 3),
                              "window_frames": self.low_activity_window},
                    frame_idx=frame_idx))
                self._low_activity_fired.add(tid)

    def summary(self) -> dict:
        report = {"tracks": {}}
        for tid, total in self.total_frames_per_track.items():
            zones = dict(self.zone_frames.get(tid, {}))
            zone_pct = {z: round(cnt / total, 3) for z, cnt in zones.items()}
            hist = self.position_history[tid]
            xs = np.array([p[1] for p in hist])
            speeds = np.hypot(np.diff(xs), np.diff(np.array([p[2] for p in hist]))) if len(hist) > 1 else np.array([])
            avg_speed = float(speeds.mean()) if len(speeds) else 0.0
            repetitiveness, lag = autocorrelation_repetitiveness(xs)
            report["tracks"][tid] = {
                "frames_tracked": total,
                "avg_speed_px_per_frame": round(avg_speed, 3),
                "zone_occupancy_pct": zone_pct,
                "repetitiveness_score": round(repetitiveness, 3),
                "repetitiveness_lag_frames": lag,
            }
        report["events"] = [vars(e) for e in self.events]
        report["n_events"] = len(self.events)
        report["n_isolation_events"] = sum(1 for e in self.events if e.event_type == "isolation")
        report["n_low_activity_events"] = sum(1 for e in self.events if e.event_type == "low_activity")
        return report

    def heatmap_image(self):
        import cv2
        hm = self.heatmap.copy()
        if hm.max() > 0:
            hm = (hm / hm.max() * 255).astype(np.uint8)
        else:
            hm = hm.astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_color = cv2.resize(hm_color, (self.frame_size[0], self.frame_size[1]),
                               interpolation=cv2.INTER_NEAREST)
        return hm_color
