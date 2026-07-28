"""Fixed rectangular zones over the camera/house plane (feeder, waterer,
nest, etc.), used for occupancy-based behavior signals."""
from dataclasses import dataclass


@dataclass
class Zone:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
