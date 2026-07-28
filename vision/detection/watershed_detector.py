"""
Classical watershed-based blob separation, layered on PaperContrastDetector.
Used during early classical-CV experiments (uniform paper background) to
split touching objects that a simple contour would merge into one blob.
Superseded by the YOLO detector for real-world footage; kept as a
reference baseline for a controlled, high-contrast background.

Approach: distance transform on the binary foreground mask gives each
foreground pixel a value proportional to its distance from the nearest
edge, so the center of each object is a local maximum even when two
objects are touching. Thresholding that distance map yields one seed per
object, and watershed floods outward from those seeds to cut along the
boundary between neighbors.
"""
from typing import List
import cv2
import numpy as np

from vision.detection.base import Detector, Detection
from vision.detection.paper_contrast_detector import PaperContrastDetector


class WatershedBeanDetector(PaperContrastDetector):
    def __init__(self, min_area: float = 25.0, max_area: float = 3000.0,
                 adaptive_block_size: int = 61, adaptive_c: float = 8.0,
                 distance_peak_ratio: float = 0.4):
        super().__init__(min_area=min_area, max_area=max_area,
                          min_circularity=0.0,  # circularity filtering no longer applies post-watershed
                          adaptive_block_size=adaptive_block_size, adaptive_c=adaptive_c)
        self.distance_peak_ratio = distance_peak_ratio

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
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        if cv2.countNonZero(object_mask) == 0:
            return []

        dist = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []
        _, sure_fg = cv2.threshold(dist, self.distance_peak_ratio * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        sure_bg = cv2.dilate(object_mask, np.ones((5, 5), np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        cv2.watershed(working_frame, markers)

        detections = []
        for label in range(2, n_markers + 1):  # 1 = background, -1 = watershed boundaries
            component_mask = np.uint8(markers == label) * 255
            area = cv2.countNonZero(component_mask)
            if area < self.min_area or area > self.max_area:
                continue
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            detections.append(Detection(x=x / scale, y=y / scale, radius=radius / scale,
                                         area=float(area) / (scale ** 2), confidence=1.0, label="object"))
        return detections
