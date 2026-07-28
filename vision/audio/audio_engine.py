"""
Audio Engine: acoustic event detection (cough/distress) at the zone level.

No real barn audio is available in this environment. Rather than fake a
waveform and then a spectrogram, this simulates directly at the acoustic
*feature* level (energy, spectral centroid, event duration) -- the
representation any real audio pipeline would produce after its first
processing stage.

Audio is inherently zone-level, not per-individual: a microphone cannot
isolate which specific animal coughed among many. AudioEngine therefore
produces zone/session-level evidence and feeds the Risk Engine at the
population level, not per-track.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AudioEvent:
    event_type: str  # "cough" | "normal_vocalization"
    confidence: float
    window_idx: int
    zone_id: str


class SyntheticAcousticSimulator:
    """Generates 1s windows of synthetic acoustic features: normal flock
    ambience vs. cough/distress events. Also returns ground-truth labels
    for training and honest evaluation."""

    def __init__(self, seed: int = 42, cough_rate_per_min: float = 3.0):
        self.rng = np.random.default_rng(seed)
        self.cough_prob_per_window = cough_rate_per_min / 60.0

    def generate_session(self, n_windows: int) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        for _ in range(n_windows):
            is_cough = self.rng.random() < self.cough_prob_per_window
            if is_cough:
                energy = self.rng.normal(7.5, 1.2)
                spectral_centroid = self.rng.normal(0.72, 0.08)
                duration = self.rng.normal(0.35, 0.08)
            else:
                energy = self.rng.normal(3.0, 1.0)
                spectral_centroid = self.rng.normal(0.35, 0.1)
                duration = self.rng.normal(0.9, 0.2)
            features.append([energy, spectral_centroid, duration])
            labels.append(1 if is_cough else 0)
        return np.array(features), np.array(labels)


class SimpleLogisticClassifier:
    """Minimal logistic regression (manual gradient descent, no ML
    dependency) -- sufficient for a synthetic, separable 2-class problem.
    The point is to demonstrate the full pipeline (features -> model ->
    classified event -> Risk Engine evidence), not classifier sophistication."""

    def __init__(self, n_features: int = 3, lr: float = 0.1, epochs: int = 300):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.mean = None
        self.std = None

    def _normalize(self, X):
        return (X - self.mean) / (self.std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        Xn = self._normalize(X)
        n = len(y)
        for _ in range(self.epochs):
            z = Xn @ self.w + self.b
            p = 1 / (1 + np.exp(-z))
            grad_w = Xn.T @ (p - y) / n
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        z = Xn @ self.w + self.b
        return 1 / (1 + np.exp(-z))


class AudioEngine:
    def __init__(self, classifier: SimpleLogisticClassifier, confidence_threshold: float = 0.6,
                 zone_id: str = "house_zone_1"):
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.zone_id = zone_id

    def process_session(self, features: np.ndarray) -> List[AudioEvent]:
        probs = self.classifier.predict_proba(features)
        events = []
        for i, p in enumerate(probs):
            if p >= self.confidence_threshold:
                events.append(AudioEvent(event_type="cough", confidence=round(float(p), 3),
                                          window_idx=i, zone_id=self.zone_id))
        return events

    def summary(self, events: List[AudioEvent], session_duration_min: float) -> dict:
        n_coughs = len(events)
        avg_conf = float(np.mean([e.confidence for e in events])) if events else 0.0
        return {
            "zone_id": self.zone_id,
            "n_cough_events": n_coughs,
            "cough_rate_per_min": round(n_coughs / session_duration_min, 2) if session_duration_min else 0.0,
            "avg_confidence": round(avg_conf, 3),
        }
