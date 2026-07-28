"""
Synthetic camera model for multi-camera simulation: several cameras
observe the same scene from different positions and must agree on a
single identity per animal.

Simplification: rather than a full 3D/homography projection, each camera
operates directly in world coordinates but models two effects that matter
for coverage/consensus testing: detection probability falls off with
distance from the camera, and measurement noise grows with distance --
both realistic properties of any real camera's effective resolution.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from vision.detection.base import Detection


@dataclass
class CameraSpec:
    camera_id: str
    x: float
    y: float
    max_range: float = 700.0
    base_noise: float = 2.0
    noise_growth: float = 0.012  # extra px of noise per px of distance


class SyntheticCamera:
    def __init__(self, spec: CameraSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng

    def observe(self, true_x: float, true_y: float, radius: float) -> Optional[Detection]:
        d = np.hypot(true_x - self.spec.x, true_y - self.spec.y)
        if d > self.spec.max_range:
            return None
        detect_prob = float(np.clip(1.0 - (d / self.spec.max_range) ** 2, 0.05, 0.99))
        if self.rng.random() > detect_prob:
            return None

        noise_std = self.spec.base_noise + self.spec.noise_growth * d
        obs_x = true_x + self.rng.normal(0, noise_std)
        obs_y = true_y + self.rng.normal(0, noise_std)
        return Detection(x=obs_x, y=obs_y, radius=radius, area=np.pi * radius ** 2,
                          confidence=detect_prob, label=self.spec.camera_id)


def default_corner_cameras(width: int, height: int, rng: np.random.Generator) -> List[SyntheticCamera]:
    specs = [
        CameraSpec("cam_NW", 0, 0),
        CameraSpec("cam_NE", width, 0),
        CameraSpec("cam_SW", 0, height),
        CameraSpec("cam_SE", width, height),
    ]
    return [SyntheticCamera(s, rng) for s in specs]
