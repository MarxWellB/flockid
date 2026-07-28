"""
Cross-camera identity association via trajectory consistency.

Each camera runs its own independent MultiObjectTracker; camera noise is
already smoothed out before anything is compared across cameras.
Cross-camera links accumulate evidence over time (co-occurrence of
smoothed positions), rather than being decided from a single frame.
Identity merging uses Union-Find with path compression, which handles
transitive merges correctly (a hand-rolled alias dictionary is prone to
stale-pointer bugs when a track is merged more than once).
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent: Dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class CrossCameraAssociator:
    def __init__(self, match_distance: float = 15.0, min_observations_to_merge: int = 20):
        """
        match_distance: max world-space distance between two smoothed
            (post-Kalman) positions from different cameras to consider
            them a possible match in a given frame.
        min_observations_to_merge: sustained co-occurrence required
            before merging two tracks -- requires accumulated evidence,
            not a single-frame coincidence.
        """
        self.match_distance = match_distance
        self.min_observations = min_observations_to_merge
        self.co_occurrence: Dict[Tuple, int] = defaultdict(int)
        self.merged_pairs: set = set()
        self.uf = UnionFind()

    def observe(self, camera_track_positions: Dict[Tuple[str, int], Tuple[float, float]]):
        """camera_track_positions: {(camera_id, local_track_id): (x, y)}"""
        items = list(camera_track_positions.items())
        for i in range(len(items)):
            (cam_i, tid_i), (xi, yi) = items[i]
            for j in range(i + 1, len(items)):
                (cam_j, tid_j), (xj, yj) = items[j]
                if cam_i == cam_j:
                    continue
                d = np.hypot(xi - xj, yi - yj)
                if d <= self.match_distance:
                    key = tuple(sorted([(cam_i, tid_i), (cam_j, tid_j)]))
                    self.co_occurrence[key] += 1
                    if (self.co_occurrence[key] >= self.min_observations
                            and key not in self.merged_pairs):
                        self.uf.union(key[0], key[1])
                        self.merged_pairs.add(key)

    def global_id(self, camera_id: str, local_track_id: int) -> str:
        root = self.uf.find((camera_id, local_track_id))
        return f"global_{hash(root) % 100000}"

    def n_merges(self) -> int:
        return len(self.merged_pairs)
