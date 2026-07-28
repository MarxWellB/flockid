"""
YOLO11-based detector. Fine-tuned on a real chicken dataset (1,140
images, 18 cameras, 13 dates), validated with a date-based train/val
split (not random) to measure true generalization.

Held-out results: mAP50 95.6%, mAP50-95 65.3%, precision 91.7%, recall 91.2%.
"""
import os
from typing import List
import numpy as np

from vision.detection.base import Detector, Detection

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "chicken_yolo11n.pt")


class YOLODetector(Detector):
    def __init__(self, weights_path: str = _WEIGHTS_PATH, confidence_threshold: float = 0.5,
                 nms_iou: float = 0.3):
        """
        confidence_threshold: on footage outside the training domain,
            detector confidence decays smoothly with no clear separation
            between real objects and background noise; 0.5 filters most
            of that noise while keeping recall reasonable.
        nms_iou: lowered from the ultralytics default (0.7) because the
            default did not suppress duplicate/overlapping boxes on the
            same animal.
        """
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou = nms_iou

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.confidence_threshold,
                                      iou=self.nms_iou, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            w, h = x1 - x0, y1 - y0
            cx, cy = x0 + w / 2, y0 + h / 2
            radius = max(w, h) / 2
            detections.append(Detection(x=cx, y=cy, radius=radius, area=w * h,
                                         confidence=conf, label="chicken"))
        return detections
