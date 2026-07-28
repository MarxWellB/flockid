"""
Environmental Engine.

Design principle: ambient conditions are not direct health evidence -- an
ammonia spike is not a health event by itself, but context that changes
how significant another signal (behavior, audio) is. Unlike Audio Engine
(an additive term on the Health Score), Environmental Engine acts as a
multiplier on the already-computed Risk Score.

No real sensors are connected here. A synthetic time series of
temperature/humidity/CO2/ammonia is generated within realistic ranges,
and a simple stress index is computed from deviation off optimal ranges.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


# Approximate optimal ranges for adult broilers (general management
# guidance, not a specific clinical source -- a reasonable placeholder to
# be calibrated with a client's veterinarian/nutritionist in production).
OPTIMAL_RANGES = {
    "temperature_c": (18.0, 24.0),
    "humidity_pct": (50.0, 70.0),
    "co2_ppm": (0, 3000),
    "ammonia_ppm": (0, 15),
}


@dataclass
class SensorReading:
    timestamp_min: float
    temperature_c: float
    humidity_pct: float
    co2_ppm: float
    ammonia_ppm: float


class EnvironmentalSimulator:
    """Generates a synthetic sensor session. `stress_level` controls how
    far from optimal the values are: 0.0 = ideal conditions, 1.0 =
    sustained strong environmental stress."""

    def __init__(self, seed: int = 42, stress_level: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.stress_level = np.clip(stress_level, 0.0, 1.0)

    def generate_session(self, n_readings: int, interval_min: float = 5.0) -> List[SensorReading]:
        readings = []
        for i in range(n_readings):
            t = i * interval_min
            temp = 21.0 + self.stress_level * self.rng.uniform(4, 10) + self.rng.normal(0, 0.8)
            humidity = 60.0 + self.stress_level * self.rng.uniform(-15, 20) + self.rng.normal(0, 3)
            co2 = 1500 + self.stress_level * self.rng.uniform(1500, 4000) + self.rng.normal(0, 200)
            ammonia = 8.0 + self.stress_level * self.rng.uniform(10, 30) + self.rng.normal(0, 1.5)
            readings.append(SensorReading(t, round(temp, 1), round(max(humidity, 0), 1),
                                           round(max(co2, 0), 0), round(max(ammonia, 0), 1)))
        return readings


class EnvironmentalEngine:
    def __init__(self, max_stress_multiplier: float = 0.4):
        """
        max_stress_multiplier: maximum amplification of risk_score under
            the worst environmental conditions (0.4 = up to +40%). A
            deliberate cap -- environment modulates the score, it does
            not dominate it.
        """
        self.max_stress_multiplier = max_stress_multiplier

    def _deviation(self, value: float, lo: float, hi: float) -> float:
        if lo <= value <= hi:
            return 0.0
        span = max(hi - lo, 1e-6)
        if value < lo:
            return min((lo - value) / span, 1.0)
        return min((value - hi) / span, 1.0)

    def summarize(self, readings: List[SensorReading]) -> dict:
        if not readings:
            return {"stress_index": 0.0, "avg_temperature_c": None, "avg_ammonia_ppm": None}

        devs = []
        for r in readings:
            d_temp = self._deviation(r.temperature_c, *OPTIMAL_RANGES["temperature_c"])
            d_hum = self._deviation(r.humidity_pct, *OPTIMAL_RANGES["humidity_pct"])
            d_co2 = self._deviation(r.co2_ppm, *OPTIMAL_RANGES["co2_ppm"])
            d_nh3 = self._deviation(r.ammonia_ppm, *OPTIMAL_RANGES["ammonia_ppm"])
            # ammonia and temperature weighted higher: most directly tied
            # to respiratory welfare in general poultry management literature
            devs.append(0.35 * d_temp + 0.15 * d_hum + 0.15 * d_co2 + 0.35 * d_nh3)

        stress_index = float(np.mean(devs))
        return {
            "stress_index": round(stress_index, 3),
            "avg_temperature_c": round(float(np.mean([r.temperature_c for r in readings])), 1),
            "avg_humidity_pct": round(float(np.mean([r.humidity_pct for r in readings])), 1),
            "avg_co2_ppm": round(float(np.mean([r.co2_ppm for r in readings])), 0),
            "avg_ammonia_ppm": round(float(np.mean([r.ammonia_ppm for r in readings])), 1),
            "n_readings": len(readings),
        }

    def risk_multiplier(self, environmental_report: dict) -> float:
        """1.0 = no effect, up to 1+max_stress_multiplier in the worst case."""
        stress = environmental_report.get("stress_index", 0.0)
        return 1.0 + self.max_stress_multiplier * stress
