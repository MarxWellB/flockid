"""
Abstract tracker interface. Any tracker implementation (Kalman+Mahalanobis
or a wrapper around ByteTrack/BoT-SORT) implements this so consumers don't
depend on the concrete implementation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from vision.detection.base import Detection


@dataclass
class TrackState:
    """Public, stable view of a track -- the only thing downstream code
    (rendering, evaluation) should touch, regardless of which tracker
    produced it."""
    track_id: int
    x: float
    y: float
    radius: float
    time_since_update: int
    history: list = field(default_factory=list)  # [(frame_idx, x, y)]


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], frame_idx: int,
               embeddings: Optional[List[np.ndarray]] = None) -> Dict[int, TrackState]:
        raise NotImplementedError
