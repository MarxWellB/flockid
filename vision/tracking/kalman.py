"""
Constant-velocity Kalman filter (state: x, y, vx, vy) with adaptive
process noise driven by Normalized Innovation Squared (NIS).

Standard theory: under a correctly-specified model, NIS ~ chi^2(2) with
expected value 2.0. A sustained NIS above that indicates the filter is
underestimating its own uncertainty (the object is less predictable than
the constant-velocity assumption implies), which should increase Q rather
than relaxing the association gate with an arbitrary multiplier.
"""
import numpy as np
from typing import Optional


class KalmanCV:
    def __init__(self, x: float, y: float, dt: float = 1.0,
                 process_var: float = 1.0, measurement_var: float = 9.0,
                 initial_velocity_var: float = 25.0,
                 adaptive: bool = True, nis_ema_alpha: float = 0.3,
                 q_scale_min: float = 1.0, q_scale_max: float = 30.0):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, initial_velocity_var, initial_velocity_var]).astype(np.float64)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.base_process_var = process_var
        self.Q = np.eye(4, dtype=np.float64) * process_var
        self.R = np.eye(2, dtype=np.float64) * measurement_var

        self.adaptive = adaptive
        self.nis_ema_alpha = nis_ema_alpha
        self.q_scale_min = q_scale_min
        self.q_scale_max = q_scale_max
        self.q_scale = 1.0
        self.nis_ema: Optional[float] = None

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def vx(self) -> float:
        return float(self.state[2])

    @property
    def vy(self) -> float:
        return float(self.state[3])

    def predict(self):
        self.state = self.F @ self.state
        effective_Q = self.Q * self.q_scale if self.adaptive else self.Q
        self.P = self.F @ self.P @ self.F.T + effective_Q

    def innovation_covariance(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self.R

    def mahalanobis(self, z: np.ndarray) -> float:
        """Mahalanobis distance (not squared) between observation z and the prediction."""
        y = z - self.H @ self.state
        S = self.innovation_covariance()
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        d2 = float(y.T @ S_inv @ y)
        return float(np.sqrt(max(d2, 0.0)))

    def observe_nis(self, mahal_probe: float):
        """
        Update the NIS EMA and adjust q_scale for the next prediction.

        mahal_probe should come from the nearest available detection in
        the frame, regardless of whether it was accepted as a match.
        Feeding this only with accepted matches biases the estimator: the
        gate itself truncates the NIS distribution, so the adaptive
        mechanism would never see the high values it needs to react to.
        """
        nis = mahal_probe ** 2
        self.nis_ema = nis if self.nis_ema is None else (
            self.nis_ema_alpha * nis + (1 - self.nis_ema_alpha) * self.nis_ema
        )
        expected_nis = 2.0
        raw_scale = self.nis_ema / expected_nis
        self.q_scale = float(np.clip(raw_scale, self.q_scale_min, self.q_scale_max))

    def update(self, z: np.ndarray):
        y = z - self.H @ self.state
        S = self.innovation_covariance()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
