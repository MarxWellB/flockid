"""
Abstract detector interface. Any detector (blob-based, YOLO, RT-DETR)
implements this. The rest of the pipeline (tracker, evaluation) only
depends on this interface, never on a concrete implementation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    radius: float
    area: float
    confidence: float = 1.0
    label: str = "object"


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return detections for a single frame. Must be stateless across calls."""
        raise NotImplementedError
