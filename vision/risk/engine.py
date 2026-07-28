"""
Risk Engine v1: transparent weighted scoring.

Combines Behavior, Audio, and Environmental evidence into a Health Score,
a Behavior Score, and a final Risk Score. Designed to work with partial
evidence: any evidence source can be omitted, and the engine reports
which ones it actually used. This is a risk score for human inspection,
not a diagnosis; every score is returned with its top contributing
factors so it is never a black-box number.

Audio evidence is additive (a direct health signal). Environmental
evidence acts as a multiplier on the final score rather than an additive
term, since ambient conditions are context that changes how concerning
other signals are, not direct evidence of illness themselves.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RiskReport:
    health_score: float
    behavior_score: float
    risk_score: float
    evidence_sources: List[str]
    top_evidence: List[dict]
    raw_features: dict


class RiskEngineV1:
    def __init__(self,
                 w_isolation: float = 0.30,
                 w_low_activity: float = 0.30,
                 w_low_speed: float = 0.20,
                 w_repetitiveness: float = 0.20,
                 w_audio_distress: float = 0.35,
                 repetitiveness_anomaly_threshold: float = 0.85,
                 speed_reference_px_per_frame: float = 3.0,
                 cough_rate_reference: float = 2.0):
        """
        Weights are an engineering starting point, not a calibration —
        there is no labeled outcome data yet to fit them against.

        w_audio_distress: weight of the audio signal in the Health Score
            (not the Behavior Score) -- audio is a respiratory health
            signal, conceptually distinct from visual behavior.
        cough_rate_reference: baseline cough rate (per minute) considered
            normal background noise in a healthy flock; a placeholder,
            not clinically calibrated.
        """
        self.weights = {
            "isolation": w_isolation,
            "low_activity": w_low_activity,
            "low_speed": w_low_speed,
            "repetitiveness": w_repetitiveness,
        }
        self.w_audio_distress = w_audio_distress
        self.repetitiveness_anomaly_threshold = repetitiveness_anomaly_threshold
        self.speed_reference = speed_reference_px_per_frame
        self.cough_rate_reference = cough_rate_reference

    def _extract_features(self, behavior_report: dict) -> dict:
        tracks = behavior_report.get("tracks", {})
        n = len(tracks) or 1

        events = behavior_report.get("events", [])
        isolated_ids = {e["entity_id"] for e in events if e["event_type"] == "isolation"}
        low_activity_ids = {e["entity_id"] for e in events if e["event_type"] == "low_activity"}
        repetitive_ids = {tid for tid, t in tracks.items()
                           if t["repetitiveness_score"] >= self.repetitiveness_anomaly_threshold}

        avg_speed = float(np.mean([t["avg_speed_px_per_frame"] for t in tracks.values()])) if tracks else 0.0
        low_speed_ratio = float(np.clip(1.0 - avg_speed / self.speed_reference, 0.0, 1.0)) if self.speed_reference > 0 else 0.0

        return {
            "isolation_rate": len(isolated_ids) / n,
            "low_activity_rate": len(low_activity_ids) / n,
            "low_speed_ratio": low_speed_ratio,
            "repetitiveness_rate": len(repetitive_ids) / n,
            "avg_speed_px_per_frame": avg_speed,
            "n_tracks": len(tracks),
        }

    def score(self, behavior_report: dict,
              audio_report: Optional[dict] = None,
              environmental_report: Optional[dict] = None) -> RiskReport:
        features = self._extract_features(behavior_report)
        evidence_sources = ["behavior"]
        if audio_report is not None:
            evidence_sources.append("audio")
        if environmental_report is not None:
            evidence_sources.append("environmental")

        weighted_terms = {
            "isolation": self.weights["isolation"] * features["isolation_rate"],
            "low_activity": self.weights["low_activity"] * features["low_activity_rate"],
            "low_speed": self.weights["low_speed"] * features["low_speed_ratio"],
            "repetitiveness": self.weights["repetitiveness"] * features["repetitiveness_rate"],
        }
        total_weight = sum(self.weights.values())
        behavior_score = 100.0 * sum(weighted_terms.values()) / total_weight

        audio_distress_ratio = 0.0
        if audio_report is not None:
            cough_rate = audio_report.get("cough_rate_per_min", 0.0)
            audio_distress_ratio = float(np.clip(
                (cough_rate - self.cough_rate_reference) / max(self.cough_rate_reference, 0.1), 0.0, 1.0))
            weighted_terms["audio_distress"] = self.w_audio_distress * audio_distress_ratio
            health_score = 100.0 * (sum(weighted_terms[k] for k in
                                         ["isolation", "low_activity", "low_speed", "audio_distress"])
                                     ) / (self.weights["isolation"] + self.weights["low_activity"]
                                          + self.weights["low_speed"] + self.w_audio_distress)
        else:
            health_score = behavior_score

        risk_score = 0.5 * health_score + 0.5 * behavior_score

        environmental_multiplier = 1.0
        if environmental_report is not None:
            stress = environmental_report.get("stress_index", 0.0)
            environmental_multiplier = 1.0 + 0.4 * stress
            risk_score = min(100.0, risk_score * environmental_multiplier)

        active_weights = dict(self.weights)
        if audio_report is not None:
            active_weights["audio_distress"] = self.w_audio_distress
        active_total = sum(active_weights.get(k, 0) for k in weighted_terms)
        top_evidence = sorted(
            [{"factor": k, "contribution_pct": round(100 * v / active_total, 1) if active_total else 0.0}
             for k, v in weighted_terms.items()],
            key=lambda e: -e["contribution_pct"]
        )[:3]

        features["audio_distress_ratio"] = audio_distress_ratio
        if audio_report is not None:
            features["cough_rate_per_min"] = audio_report.get("cough_rate_per_min", 0.0)
        features["environmental_multiplier"] = round(environmental_multiplier, 3)
        if environmental_report is not None:
            features["environmental_stress_index"] = environmental_report.get("stress_index", 0.0)

        return RiskReport(
            health_score=round(health_score, 1),
            behavior_score=round(behavior_score, 1),
            risk_score=round(risk_score, 1),
            evidence_sources=evidence_sources,
            top_evidence=top_evidence,
            raw_features=features,
        )
