"""
Appearance re-identification: HSV color histogram of the object patch,
used as a lightweight substitute for a learned embedding when no labeled
training data is available. Works well when objects have distinct colors;
degrades when multiple objects share a similar color, where a trained
embedding (e.g. a Siamese network on real crops) would be the real fix.
"""
import numpy as np
import cv2


N_BINS_H = 12
N_BINS_S = 6


def extract_appearance(frame: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    h, w = frame.shape[:2]
    r = max(2, int(radius * 0.8))  # slightly smaller than the detected radius to avoid edge pixels
    x0, x1 = max(0, int(x - r)), min(w, int(x + r))
    y0, y1 = max(0, int(y - r)), min(h, int(y + r))
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(N_BINS_H + N_BINS_S, dtype=np.float32)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [N_BINS_H], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [N_BINS_S], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat = feat / norm
    return feat


def extract_batch(frame: np.ndarray, detections) -> list:
    return [extract_appearance(frame, d.x, d.y, d.radius) for d in detections]
