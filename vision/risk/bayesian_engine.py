"""
Risk Engine v2: Bayesian network fusion, using pgmpy.

Network structure:

    EnvironmentalStress -> Health -> Isolation
                                   -> LowActivity
                                   -> AudioDistress

Environmental stress modifies the prior probability of "at_risk" rather
than being observed as a direct symptom, consistent with treating ambient
conditions as context rather than direct evidence. Missing evidence
sources are handled natively by marginalization during inference, with no
special-case branching required.

The conditional probability tables are engineering estimates, not fit to
labeled outcome data -- the mechanism is principled, the calibration is
still a placeholder pending real data, same as in v1.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


@dataclass
class BayesianRiskReport:
    risk_probability: float  # P(Health = at_risk | observed evidence)
    evidence_used: Dict[str, str]
    evidence_sources: List[str]


class RiskEngineV2Bayesian:
    def __init__(self,
                 isolation_threshold: float = 0.3,
                 low_activity_threshold: float = 0.3,
                 cough_rate_threshold: float = 2.0,
                 env_stress_threshold: float = 0.3):
        self.thresholds = {
            "isolation": isolation_threshold,
            "low_activity": low_activity_threshold,
            "cough_rate": cough_rate_threshold,
            "env_stress": env_stress_threshold,
        }
        self.model = self._build_network()
        self.inference = VariableElimination(self.model)

    def _build_network(self) -> DiscreteBayesianNetwork:
        model = DiscreteBayesianNetwork([
            ("EnvironmentalStress", "Health"),
            ("Health", "Isolation"),
            ("Health", "LowActivity"),
            ("Health", "AudioDistress"),
        ])

        cpd_env = TabularCPD("EnvironmentalStress", 2, [[0.7], [0.3]],
                              state_names={"EnvironmentalStress": ["low", "high"]})

        cpd_health = TabularCPD(
            "Health", 2,
            [[0.85, 0.55],
             [0.15, 0.45]],
            evidence=["EnvironmentalStress"], evidence_card=[2],
            state_names={"Health": ["healthy", "at_risk"],
                         "EnvironmentalStress": ["low", "high"]})

        cpd_isolation = TabularCPD(
            "Isolation", 2,
            [[0.85, 0.30],
             [0.15, 0.70]],
            evidence=["Health"], evidence_card=[2],
            state_names={"Isolation": ["low", "high"], "Health": ["healthy", "at_risk"]})

        cpd_low_activity = TabularCPD(
            "LowActivity", 2,
            [[0.88, 0.35],
             [0.12, 0.65]],
            evidence=["Health"], evidence_card=[2],
            state_names={"LowActivity": ["low", "high"], "Health": ["healthy", "at_risk"]})

        cpd_audio = TabularCPD(
            "AudioDistress", 2,
            [[0.92, 0.40],
             [0.08, 0.60]],
            evidence=["Health"], evidence_card=[2],
            state_names={"AudioDistress": ["low", "high"], "Health": ["healthy", "at_risk"]})

        model.add_cpds(cpd_env, cpd_health, cpd_isolation, cpd_low_activity, cpd_audio)
        assert model.check_model()
        return model

    def _discretize(self, behavior_report: dict, audio_report: Optional[dict],
                     environmental_report: Optional[dict]) -> Dict[str, str]:
        tracks = behavior_report.get("tracks", {})
        n = len(tracks) or 1
        events = behavior_report.get("events", [])
        isolation_rate = len({e["entity_id"] for e in events if e["event_type"] == "isolation"}) / n
        low_activity_rate = len({e["entity_id"] for e in events if e["event_type"] == "low_activity"}) / n

        evidence = {
            "Isolation": "high" if isolation_rate >= self.thresholds["isolation"] else "low",
            "LowActivity": "high" if low_activity_rate >= self.thresholds["low_activity"] else "low",
        }
        if audio_report is not None:
            cough_rate = audio_report.get("cough_rate_per_min", 0.0)
            evidence["AudioDistress"] = "high" if cough_rate >= self.thresholds["cough_rate"] else "low"
        if environmental_report is not None:
            stress = environmental_report.get("stress_index", 0.0)
            evidence["EnvironmentalStress"] = "high" if stress >= self.thresholds["env_stress"] else "low"
        return evidence

    def score(self, behavior_report: dict,
              audio_report: Optional[dict] = None,
              environmental_report: Optional[dict] = None) -> BayesianRiskReport:
        evidence = self._discretize(behavior_report, audio_report, environmental_report)
        evidence_sources = ["behavior"]
        if audio_report is not None:
            evidence_sources.append("audio")
        if environmental_report is not None:
            evidence_sources.append("environmental")

        result = self.inference.query(variables=["Health"], evidence=evidence, show_progress=False)
        risk_probability = float(result.values[result.state_names["Health"].index("at_risk")])

        return BayesianRiskReport(
            risk_probability=round(risk_probability, 4),
            evidence_used=evidence,
            evidence_sources=evidence_sources,
        )
