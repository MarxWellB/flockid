"""
v1 (weighted sum) vs v2 (real Bayesian network) -- same scenarios,
different combination mechanisms. The point is not "v2 gives a better
number" (there is no real data yet to say which is "correct") -- the
point is that v2 handles partial evidence natively and combines signals
with real probability math, not invented weights.
"""
from vision.risk.engine import RiskEngineV1
from vision.risk.bayesian_engine import RiskEngineV2Bayesian

behavior_healthy = {
    "tracks": {i: {"avg_speed_px_per_frame": 3.0, "repetitiveness_score": 0.3} for i in range(20)},
    "events": [{"event_type": "isolation", "entity_id": i} for i in range(2)],  # 10% isolation
}
behavior_sick = {
    "tracks": {i: {"avg_speed_px_per_frame": 1.0, "repetitiveness_score": 0.4} for i in range(20)},
    "events": ([{"event_type": "isolation", "entity_id": i} for i in range(14)]
               + [{"event_type": "low_activity", "entity_id": i} for i in range(12)]),  # 70% isolated, 60% low activity
}

audio_normal = {"cough_rate_per_min": 0.8}
audio_elevated = {"cough_rate_per_min": 9.0}
env_ok = {"stress_index": 0.05}
env_bad = {"stress_index": 0.8}

v1 = RiskEngineV1()
v2 = RiskEngineV2Bayesian()

print("=== v1 (weighted sum) vs v2 (Bayesian network) ===\n")

scenarios = [
    ("Healthy, no audio/environment", behavior_healthy, None, None),
    ("Healthy, WITH normal audio", behavior_healthy, audio_normal, None),
    ("At-risk behavior, no audio/environment", behavior_sick, None, None),
    ("At-risk behavior + elevated audio", behavior_sick, audio_elevated, None),
    ("At-risk behavior + elevated audio + bad environment", behavior_sick, audio_elevated, env_bad),
    ("At-risk behavior + bad environment ONLY (no audio)", behavior_sick, None, env_bad),
]

for name, br, ar, er in scenarios:
    r1 = v1.score(br, audio_report=ar, environmental_report=er)
    r2 = v2.score(br, audio_report=ar, environmental_report=er)
    print(f"{name}")
    print(f"  v1: risk_score={r1.risk_score:>6}  evidence_sources={r1.evidence_sources}")
    print(f"  v2: P(at_risk)={r2.risk_probability:.1%}   evidence used={r2.evidence_used}")
    print()

print("=== What v2 does natively that v1 needed to patch around ===")
print("With partial evidence (no audio, no environment), v2 simply marginalizes")
print("those nodes out -- no need for an 'if audio_report is not None' branch.")
print("v1 DOES need that explicit branch at every point in the code (see vision/risk/engine.py).")
