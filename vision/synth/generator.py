"""
Synthetic grain-on-conveyor simulator.

Produces:
- A sequence of frames (BGR numpy arrays) simulating an overhead camera.
- Per-frame ground truth: {track_id: (x, y, radius)}, even while a grain
  is occluded (so we can later measure how well the tracker recovers it).

Includes:
- Multiple "lanes" of grains that cross laterally (simulated crossings).
- Occlusion when two grains overlap enough (their contours merge into one blob).
- Slight shape/size noise so it isn't 100% trivial for the detector.
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
    alive: bool = True


class ConveyorSimulator:
    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        belt_speed: float = 4.0,
        spawn_rate: float = 0.35,
        n_lanes: int = 5,
        seed: int = 42,
        allow_crossing: bool = True,
    ):
        self.width = width
        self.height = height
        self.belt_speed = belt_speed
        self.spawn_rate = spawn_rate
        self.n_lanes = n_lanes
        self.allow_crossing = allow_crossing
        self.rng = np.random.default_rng(seed)
        self.grains: List[Grain] = []
        self._next_id = 1
        self.frame_idx = 0
        # beige/brown tones, all very close to each other on purpose
        self._base_color = np.array([70, 120, 170])  # BGR beige

    def _spawn(self):
        lane = self.rng.integers(0, self.n_lanes)
        lane_h = self.height / self.n_lanes
        y = lane_h * lane + lane_h / 2 + self.rng.normal(0, 4)
        radius = self.rng.uniform(9, 13)
        vx = self.belt_speed + self.rng.normal(0, 0.3)
        vy = 0.0

        if self.allow_crossing and self.rng.random() < 0.25:
            # diagonal crossing into a neighboring lane
            vy = self.rng.choice([-1, 1]) * self.rng.uniform(0.6, 1.4)

        jitter = self.rng.integers(-8, 8, size=3)
        color = tuple(int(c) for c in np.clip(self._base_color + jitter, 0, 255))
        g = Grain(self._next_id, x=-radius, y=y, vx=vx, vy=vy, radius=radius, color=color)
        self._next_id += 1
        self.grains.append(g)

    def step(self) -> Tuple[np.ndarray, Dict[int, Tuple[float, float, float]]]:
        """Advance one frame. Returns (image, ground_truth)."""
        self.frame_idx += 1
        if self.rng.random() < self.spawn_rate:
            self._spawn()

        frame = np.full((self.height, self.width, 3), (30, 30, 30), dtype=np.uint8)

        # belt lines, just so the background isn't flat and has some motion texture
        offset = int(self.frame_idx * self.belt_speed) % 40
        for x in range(-40 + offset, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (45, 45, 45), 1)

        gt = {}
        alive_grains = []
        for g in self.grains:
            g.x += g.vx
            g.y += g.vy
            g.y = float(np.clip(g.y, g.radius, self.height - g.radius))
            if g.x - g.radius > self.width:
                g.alive = False
            if g.alive:
                alive_grains.append(g)
                gt[g.track_id] = (float(g.x), float(g.y), float(g.radius))
        self.grains = alive_grains

        # draw in y-order so heavily overlapping grains render consistently (visual occlusion)
        for g in sorted(self.grains, key=lambda gg: gg.y):
            cv2.circle(frame, (int(g.x), int(g.y)), int(g.radius), g.color, -1)
            cv2.circle(frame, (int(g.x), int(g.y)), int(g.radius), (20, 20, 20), 1)

        return frame, gt

    def occlusion_pairs(self, gt: Dict[int, Tuple[float, float, float]]) -> List[Tuple[int, int]]:
        """Pairs of track_ids whose current overlap is strong (>60% of the smaller radius)."""
        ids = list(gt.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                x1, y1, r1 = gt[ids[i]]
                x2, y2, r2 = gt[ids[j]]
                d = np.hypot(x1 - x2, y1 - y2)
                if d < 0.6 * (r1 + r2):
                    pairs.append((ids[i], ids[j]))
        return pairs


def generate_sequence(n_frames: int = 400, **kwargs):
    sim = ConveyorSimulator(**kwargs)
    frames = []
    gts = []
    for _ in range(n_frames):
        frame, gt = sim.step()
        frames.append(frame)
        gts.append(gt)
    return frames, gts
