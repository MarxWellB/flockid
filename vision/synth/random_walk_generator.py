"""
Phase 2 simulator: similar objects moving RANDOMLY (no conveyor).

Unlike ConveyorSimulator (Phase 1), there is no constant belt velocity or
dominant direction here. Each grain:

- Randomly changes direction every so many frames.
- Stops completely for random intervals (simulating a bird standing
  still) and then resumes moving in a new direction.
- Bounces off the frame edges instead of disappearing (stays on screen
  the whole time -- the goal is to stress long-term identity, not
  entries/exits).

This deliberately breaks the assumption the constant-velocity Kalman
filter needs to predict well: here "physical prediction" (position =
position + velocity*time) stops being valid most of the time. That's the
point -- this phase exists to measure how much the tracker degrades, not
to avoid it.
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Grain:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    frames_until_direction_change: int
    stopped_frames_remaining: int = 0


class RandomWalkSimulator:
    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        n_grains: int = 25,
        speed: float = 3.5,
        direction_change_prob: float = 0.05,   # per-frame chance of changing heading
        stop_prob: float = 0.01,               # per-frame chance of stopping
        stop_duration_range: Tuple[int, int] = (5, 25),
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.speed = speed
        self.direction_change_prob = direction_change_prob
        self.stop_prob = stop_prob
        self.stop_duration_range = stop_duration_range
        self.rng = np.random.default_rng(seed)
        self._base_color = np.array([70, 120, 170])
        self.frame_idx = 0

        self.grains: List[Grain] = []
        for i in range(n_grains):
            self.grains.append(self._make_grain(i + 1))
        self._next_id = n_grains + 1

    def _random_direction(self) -> Tuple[float, float]:
        angle = self.rng.uniform(0, 2 * np.pi)
        return np.cos(angle) * self.speed, np.sin(angle) * self.speed

    def _make_grain(self, track_id: int) -> Grain:
        x = self.rng.uniform(30, self.width - 30)
        y = self.rng.uniform(30, self.height - 30)
        vx, vy = self._random_direction()
        radius = self.rng.uniform(9, 13)
        jitter = self.rng.integers(-8, 8, size=3)
        color = tuple(int(c) for c in np.clip(self._base_color + jitter, 0, 255))
        return Grain(
            track_id=track_id, x=x, y=y, vx=vx, vy=vy, radius=radius, color=color,
            frames_until_direction_change=self.rng.integers(10, 60),
        )

    def step(self) -> Tuple[np.ndarray, Dict[int, Tuple[float, float, float]]]:
        self.frame_idx += 1
        frame = np.full((self.height, self.width, 3), (30, 30, 30), dtype=np.uint8)

        for g in self.grains:
            if g.stopped_frames_remaining > 0:
                g.stopped_frames_remaining -= 1
            else:
                if self.rng.random() < self.stop_prob:
                    g.stopped_frames_remaining = int(self.rng.integers(*self.stop_duration_range))
                else:
                    g.frames_until_direction_change -= 1
                    if g.frames_until_direction_change <= 0 or self.rng.random() < self.direction_change_prob:
                        g.vx, g.vy = self._random_direction()
                        g.frames_until_direction_change = self.rng.integers(10, 60)
                    g.x += g.vx
                    g.y += g.vy
                    # bounce off the edges
                    if g.x - g.radius < 0 or g.x + g.radius > self.width:
                        g.vx *= -1
                        g.x = float(np.clip(g.x, g.radius, self.width - g.radius))
                    if g.y - g.radius < 0 or g.y + g.radius > self.height:
                        g.vy *= -1
                        g.y = float(np.clip(g.y, g.radius, self.height - g.radius))

        gt = {}
        for g in sorted(self.grains, key=lambda gg: gg.y):
            cv2.circle(frame, (int(g.x), int(g.y)), int(g.radius), g.color, -1)
            cv2.circle(frame, (int(g.x), int(g.y)), int(g.radius), (20, 20, 20), 1)
            gt[g.track_id] = (float(g.x), float(g.y), float(g.radius))

        return frame, gt


def generate_sequence(n_frames: int = 400, **kwargs):
    sim = RandomWalkSimulator(**kwargs)
    frames, gts = [], []
    for _ in range(n_frames):
        frame, gt = sim.step()
        frames.append(frame)
        gts.append(gt)
    return frames, gts
