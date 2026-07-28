"""
Identity Fusion Engine: links an anonymous visual track_id to a physical
animal's RFID tag using accumulated spatio-temporal co-occurrence.

The visual tracker maintains a persistent but anonymous track_id. An RFID
reader at a nest knows exactly which animal (leg band) is present, but
has no visibility outside the nest. When a visual track is within a
nest's radius at the same time an RFID read occurs there, that is
evidence -- accumulated over multiple visits, not a single-shot decision
-- that the track_id and the tag refer to the same animal.

Once a track_id -> tag link is resolved, the behavioral history already
computed for that track_id and the production history for that tag are
unified under a single profile.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class BirdProfile:
    resolved_tag: Optional[str] = None
    co_occurrence: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    egg_count: int = 0
    egg_weights: List[float] = field(default_factory=list)
    last_seen_frame: int = 0

    @property
    def confidence(self) -> float:
        """How dominant the winning tag is relative to alternatives."""
        total = sum(self.co_occurrence.values())
        if total == 0 or self.resolved_tag is None:
            return 0.0
        return self.co_occurrence[self.resolved_tag] / total

    @property
    def avg_egg_weight(self) -> float:
        return float(np.mean(self.egg_weights)) if self.egg_weights else 0.0


class IdentityFusionEngine:
    def __init__(self, min_observations_to_resolve: int = 3, enable_alias_merging: bool = False):
        """
        min_observations_to_resolve: co-occurrences required before
            trusting a track_id -> tag link.

        enable_alias_merging: experimental, off by default. The idea is
            to use RFID confirmation to merge track_ids that are actually
            the same animal, correcting visual tracker identity switches.
            Measured across several thresholds, it did not improve
            accuracy and reduced coverage -- left in the codebase but
            disabled, not presented as functional.
        """
        self.min_observations = min_observations_to_resolve
        self.enable_alias_merging = enable_alias_merging
        self.profiles: Dict[int, BirdProfile] = {}
        self.tag_to_canonical: Dict[str, int] = {}
        self.alias_of: Dict[int, int] = {}

    def _canonical(self, tid: int) -> int:
        return self.alias_of.get(tid, tid)

    def observe(self, frame_idx: int, track_positions: Dict[int, Tuple[float, float]],
                rfid_reads: List, egg_events: List, nests: List):
        nest_by_id = {n.nest_id: n for n in nests}

        tracks_in_nest: Dict[str, List[int]] = defaultdict(list)
        for tid, (x, y) in track_positions.items():
            for nest in nests:
                if np.hypot(x - nest.x, y - nest.y) <= nest.radius:
                    tracks_in_nest[nest.nest_id].append(tid)

        for read in rfid_reads:
            candidates = tracks_in_nest.get(read.nest_id, [])
            if not candidates:
                continue
            if len(candidates) == 1:
                raw_tid = candidates[0]
            else:
                nest = nest_by_id[read.nest_id]
                raw_tid = min(candidates, key=lambda t: np.hypot(
                    track_positions[t][0] - nest.x, track_positions[t][1] - nest.y))

            tid = self._canonical(raw_tid)
            profile = self.profiles.setdefault(tid, BirdProfile())
            profile.co_occurrence[read.tag] += 1
            profile.last_seen_frame = frame_idx
            self._resolve(tid, profile, read.tag)

        for egg in egg_events:
            target_track = self._track_for_tag(egg.tag)
            if target_track is not None:
                profile = self.profiles[target_track]
                profile.egg_count += 1
                profile.egg_weights.append(egg.weight_g)

    def _resolve(self, tid: int, profile: BirdProfile, latest_tag: str):
        total = sum(profile.co_occurrence.values())
        if total < self.min_observations:
            return
        best_tag = max(profile.co_occurrence, key=profile.co_occurrence.get)

        existing_canonical = self.tag_to_canonical.get(best_tag)
        if existing_canonical is not None:
            existing_canonical = self._canonical(existing_canonical)
        if (self.enable_alias_merging and existing_canonical is not None
                and existing_canonical != tid and existing_canonical in self.profiles):
            canonical_profile = self.profiles[existing_canonical]
            for t, c in profile.co_occurrence.items():
                canonical_profile.co_occurrence[t] += c
            canonical_profile.egg_count += profile.egg_count
            canonical_profile.egg_weights.extend(profile.egg_weights)
            canonical_profile.resolved_tag = best_tag
            self.alias_of[tid] = existing_canonical
            self.tag_to_canonical[best_tag] = existing_canonical
            del self.profiles[tid]
            return

        profile.resolved_tag = best_tag
        self.tag_to_canonical[best_tag] = tid

    def _track_for_tag(self, tag: str) -> Optional[int]:
        for tid, profile in self.profiles.items():
            if profile.resolved_tag == tag:
                return tid
        return None

    def summary(self) -> dict:
        out = {}
        for tid, p in self.profiles.items():
            out[tid] = {
                "resolved_tag": p.resolved_tag,
                "confidence": round(p.confidence, 3),
                "egg_count": p.egg_count,
                "avg_egg_weight_g": round(p.avg_egg_weight, 1),
                "total_observations": sum(p.co_occurrence.values()),
            }
        return out
