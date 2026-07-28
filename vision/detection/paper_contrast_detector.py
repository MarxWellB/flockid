"""
Classical two-stage detector for a high-contrast background (light paper,
dark objects). A baseline used before a trained detector was available.

Stage 1 isolates the paper region (global Otsu threshold + morphological
cleanup + largest contour), separating table from paper. Stage 2 applies
a local adaptive threshold within the paper region, since real lighting
gradients make a single global threshold fail in the darker corners.

Known limitation: still under-segments touching objects (they merge into
one large contour and get filtered by the area threshold) and is
sensitive to the circularity/area parameters -- see WatershedBeanDetector
for a partial fix, and YOLODetector for the real-world replacement.
"""
from typing import List
import cv2
import numpy as np

from vision.detection.base import Detector, Detection


class PaperContrastDetector(Detector):
    def __init__(self, min_area: float = 40.0, max_area: float = 2000.0,
                 min_circularity: float = 0.4, adaptive_block_size: int = 61,
                 adaptive_c: float = 8.0, working_width: int = 960):
        """
        working_width: area/circularity thresholds are calibrated for an
            image resized to this width -- a real object might span ~50px
            radius at 4K and ~10px at 960px wide, so the same thresholds
            behave very differently depending on camera resolution unless
            frames are normalized first.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.working_width = working_width

    def _to_working_resolution(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w == self.working_width:
            return frame, 1.0
        scale = self.working_width / w
        resized = cv2.resize(frame, (self.working_width, int(h * scale)))
        return resized, scale

    def _find_paper_mask(self, gray: np.ndarray) -> np.ndarray:
        _, paper_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray) * 255
        paper_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(paper_mask)
        cv2.drawContours(clean_mask, [paper_contour], -1, 255, -1)
        return cv2.erode(clean_mask, np.ones((5, 5), np.uint8))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        working_frame, scale = self._to_working_resolution(frame)
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        paper_mask = self._find_paper_mask(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.adaptive_block_size, self.adaptive_c)
        object_mask = cv2.bitwise_and(adaptive, paper_mask)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            circularity = area / (np.pi * radius * radius + 1e-6)
            if circularity < self.min_circularity:
                continue
            detections.append(Detection(x=x / scale, y=y / scale, radius=radius / scale,
                                         area=area / (scale ** 2),
                                         confidence=round(circularity, 2), label="object"))
        return detections
