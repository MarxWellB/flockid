"""
Bird simulator with periodic nest visits and an RFID reader (a proxy for
the real world: leg band + antenna under the nest + egg weight sensor).
Extends the RandomWalkSimulator pattern with a per-individual state
machine: wander -> go to nest -> stay in nest (RFID read + egg laying)
-> back to wandering.

This is what makes it possible to test the Identity Fusion Engine: vision
NEVER knows the real bird_id/tag (it just sees moving circles, same as
before), and the simulator does -- so we can measure how well the fusion
engine guesses real identity using only position + observed RFID reads,
without cheating.
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class BirdState(Enum):
    WANDER = "wander"
    GOTO_NEST = "goto_nest"
    IN_NEST = "in_nest"


@dataclass
class NestZone:
    nest_id: str
    x: float
    y: float
    radius: float = 35.0


@dataclass
class Bird:
    bird_id: int
    rfid_tag: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    state: BirdState = BirdState.WANDER
    target_nest: NestZone = None
    dwell_remaining: int = 0
    egg_laid_this_visit: bool = False
    frames_until_direction_change: int = 20


@dataclass
class RFIDRead:
    tag: str
    nest_id: str
    frame_idx: int


@dataclass
class EggEvent:
    tag: str
    nest_id: str
    frame_idx: int
    weight_g: float


class NestSimulator:
    def __init__(self, width: int = 960, height: int = 540, n_birds: int = 15,
                 speed: float = 3.0, seed: int = 42,
                 nest_visit_prob: float = 0.01, dwell_frames: Tuple[int, int] = (40, 80),
                 rfid_read_prob: float = 0.85, rfid_miss_prob_extra: float = 0.0):
        self.width, self.height = width, height
        self.speed = speed
        self.rng = np.random.default_rng(seed)
        self.frame_idx = 0
        self.nest_visit_prob = nest_visit_prob
        self.dwell_range = dwell_frames
        self.rfid_read_prob = rfid_read_prob
        self._base_color = np.array([70, 120, 170])

        self.nests = [
            NestZone("nest_A", width * 0.2, height * 0.15),
            NestZone("nest_B", width * 0.8, height * 0.15),
            NestZone("nest_C", width * 0.5, height * 0.85),
        ]

        self.birds: List[Bird] = []
        for i in range(n_birds):
            self.birds.append(self._make_bird(i + 1))

    def _random_direction(self):
        angle = self.rng.uniform(0, 2 * np.pi)
        return np.cos(angle) * self.speed, np.sin(angle) * self.speed

    def _make_bird(self, bird_id: int) -> Bird:
        x = self.rng.uniform(30, self.width - 30)
        y = self.rng.uniform(30, self.height - 30)
        vx, vy = self._random_direction()
        radius = self.rng.uniform(9, 13)
        jitter = self.rng.integers(-8, 8, size=3)
        color = tuple(int(c) for c in np.clip(self._base_color + jitter, 0, 255))
        return Bird(bird_id=bird_id, rfid_tag=f"TAG-{bird_id:03d}", x=x, y=y,
                    vx=vx, vy=vy, radius=radius, color=color)

    def step(self):
        self.frame_idx += 1
        frame = np.full((self.height, self.width, 3), (30, 30, 30), dtype=np.uint8)
        rfid_reads: List[RFIDRead] = []
        egg_events: List[EggEvent] = []

        for b in self.birds:
            if b.state == BirdState.WANDER:
                b.frames_until_direction_change -= 1
                if b.frames_until_direction_change <= 0:
                    b.vx, b.vy = self._random_direction()
                    b.frames_until_direction_change = self.rng.integers(15, 50)
                if self.rng.random() < self.nest_visit_prob:
                    b.state = BirdState.GOTO_NEST
                    b.target_nest = self.nests[self.rng.integers(0, len(self.nests))]
                else:
                    b.x += b.vx
                    b.y += b.vy

            elif b.state == BirdState.GOTO_NEST:
                dx = b.target_nest.x - b.x
                dy = b.target_nest.y - b.y
                dist = np.hypot(dx, dy)
                if dist < b.target_nest.radius * 0.5:
                    b.state = BirdState.IN_NEST
                    b.dwell_remaining = int(self.rng.integers(*self.dwell_range))
                    b.egg_laid_this_visit = False
                else:
                    b.vx, b.vy = (dx / dist) * self.speed, (dy / dist) * self.speed
                    b.x += b.vx
                    b.y += b.vy

            elif b.state == BirdState.IN_NEST:
                b.dwell_remaining -= 1
                if self.rng.random() < self.rfid_read_prob:
                    rfid_reads.append(RFIDRead(tag=b.rfid_tag, nest_id=b.target_nest.nest_id,
                                                frame_idx=self.frame_idx))
                if not b.egg_laid_this_visit and b.dwell_remaining < (self.dwell_range[0] // 2):
                    if self.rng.random() < 0.05:
                        weight = float(self.rng.normal(58, 4))
                        egg_events.append(EggEvent(tag=b.rfid_tag, nest_id=b.target_nest.nest_id,
                                                    frame_idx=self.frame_idx, weight_g=round(weight, 1)))
                        b.egg_laid_this_visit = True
                if b.dwell_remaining <= 0:
                    b.state = BirdState.WANDER
                    b.vx, b.vy = self._random_direction()

            b.x = float(np.clip(b.x, b.radius, self.width - b.radius))
            b.y = float(np.clip(b.y, b.radius, self.height - b.radius))

        for nest in self.nests:
            cv2.rectangle(frame, (int(nest.x - nest.radius), int(nest.y - nest.radius)),
                          (int(nest.x + nest.radius), int(nest.y + nest.radius)), (60, 60, 90), 2)
        for b in sorted(self.birds, key=lambda bb: bb.y):
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.radius), b.color, -1)
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.radius), (20, 20, 20), 1)

        gt = {b.bird_id: (b.x, b.y, b.radius, b.rfid_tag) for b in self.birds}
        return frame, gt, rfid_reads, egg_events
