"""
Blob detector for synthetic/simulated frames (contour + minimum enclosing
circle). This is a simple placeholder used to validate the tracking
pipeline against a synthetic simulator with known ground truth -- not a
real-world object detector (see YOLODetector for that). When two objects
overlap strongly, their contours merge into a single larger blob,
simulating occlusion on purpose so the tracker has to handle it.
"""
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

from vision.detection.base import Detector, Detection as BaseDetection


@dataclass
class Detection:
    x: float
    y: float
    radius: float
    area: float


class BlobDetector(Detector):
    def __init__(self, background_value: int = 30, min_area: float = 40.0):
        self.background_value = background_value
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> List[BaseDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.background_value + 10, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            detections.append(BaseDetection(x=x, y=y, radius=radius, area=area,
                                             confidence=1.0, label="object"))
        return detections
