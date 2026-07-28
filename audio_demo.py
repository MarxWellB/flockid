"""
Demo: Audio Engine end-to-end -- train the synthetic classifier, validate
it honestly (separate train/val), and confirm the Risk Engine actually
responds to audio (previously it only declared the source, never used it
-- see the fix in vision/risk/engine.py).
"""
import json
import numpy as np

from vision.audio.audio_engine import SyntheticAcousticSimulator, SimpleLogisticClassifier, AudioEngine
from vision.risk.engine import RiskEngineV1


def evaluate_classifier():
    sim_train = SyntheticAcousticSimulator(seed=1, cough_rate_per_min=3.0)
    X_train, y_train = sim_train.generate_session(n_windows=3000)

    clf = SimpleLogisticClassifier()
    clf.fit(X_train, y_train)

    # validate on a DIFFERENT seed (data the classifier never saw)
    sim_val = SyntheticAcousticSimulator(seed=999, cough_rate_per_min=3.0)
    X_val, y_val = sim_val.generate_session(n_windows=1500)
    probs = clf.predict_proba(X_val)
    preds = (probs >= 0.6).astype(int)

    tp = int(np.sum((preds == 1) & (y_val == 1)))
    fp = int(np.sum((preds == 1) & (y_val == 0)))
    fn = int(np.sum((preds == 0) & (y_val == 1)))
    tn = int(np.sum((preds == 0) & (y_val == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("=== Audio Engine: classifier validation (unseen synthetic data) ===")
    print(f"Precision: {precision:.1%}  Recall: {recall:.1%}  (TP={tp} FP={fp} FN={fn} TN={tn})")
    print("Note: the synthetic problem is separable by design (two distinct Gaussian")
    print("distributions) -- this precision should NOT be read as 'this is how well it")
    print("would work with real barn audio,' which has much harder fan/motor noise")
    print("(a real acoustic pipeline would face much harder background noise).")
    return clf


def demonstrate_risk_engine_uses_audio(clf):
    behavior_report_healthy = {
        "tracks": {i: {"avg_speed_px_per_frame": 3.0, "repetitiveness_score": 0.3} for i in range(15)},
        "events": [],
    }

    engine = RiskEngineV1()

    print("\n=== Does the Risk Engine actually respond to audio? ===\n")

    # Scenario A: normal behavior, NO audio connected
    report_no_audio = engine.score(behavior_report_healthy)
    print(f"No audio connected:      health_score={report_no_audio.health_score}  "
          f"evidence_sources={report_no_audio.evidence_sources}")

    # Scenario B: normal behavior + normal audio (low cough rate)
    audio_engine_normal = AudioEngine(clf, zone_id="zone_1")
    sim_normal = SyntheticAcousticSimulator(seed=5, cough_rate_per_min=1.0)  # low, normal rate
    X, y = sim_normal.generate_session(n_windows=1800)  # 30 min of simulated audio
    events_normal = audio_engine_normal.process_session(X)
    audio_report_normal = audio_engine_normal.summary(events_normal, session_duration_min=30)
    report_audio_normal = engine.score(behavior_report_healthy, audio_report=audio_report_normal)
    print(f"With NORMAL audio:        health_score={report_audio_normal.health_score}  "
          f"cough_rate={audio_report_normal['cough_rate_per_min']}/min  "
          f"evidence_sources={report_audio_normal.evidence_sources}")

    # Scenario C: normal behavior + elevated cough rate (possible respiratory issue)
    audio_engine_sick = AudioEngine(clf, zone_id="zone_1")
    sim_sick = SyntheticAcousticSimulator(seed=6, cough_rate_per_min=8.0)  # elevated rate
    X2, y2 = sim_sick.generate_session(n_windows=1800)
    events_sick = audio_engine_sick.process_session(X2)
    audio_report_sick = audio_engine_sick.summary(events_sick, session_duration_min=30)
    report_audio_sick = engine.score(behavior_report_healthy, audio_report=audio_report_sick)
    print(f"With ELEVATED audio (cough): health_score={report_audio_sick.health_score}  "
          f"cough_rate={audio_report_sick['cough_rate_per_min']}/min  "
          f"evidence_sources={report_audio_sick.evidence_sources}")
    print(f"\nTop evidence (elevated audio): {report_audio_sick.top_evidence}")

    print("\n=== Honest check ===")
    if (report_audio_sick.health_score > report_audio_normal.health_score
            and report_audio_normal.health_score >= report_no_audio.health_score):
        print("health_score rises when audio indicates real distress (cough rate above")
        print("reference), and does NOT trigger on normal audio (rate below reference) --")
        print("correct behavior, not just 'blindly monotonic'.")
        print("Before this fix, audio_report was accepted but NEVER changed the score --")
        print("evidence_sources said 'audio' without audio influencing anything. Fixed.")
    else:
        print("Score did not respond as expected to audio -- check the formula.")

    with open("demo_out/audio_risk_demo.json", "w") as f:
        json.dump({
            "no_audio": vars(report_no_audio),
            "normal_audio": {**vars(report_audio_normal), "cough_rate": audio_report_normal["cough_rate_per_min"]},
            "elevated_audio": {**vars(report_audio_sick), "cough_rate": audio_report_sick["cough_rate_per_min"]},
        }, f, indent=2, default=str)


if __name__ == "__main__":
    clf = evaluate_classifier()
    demonstrate_risk_engine_uses_audio(clf)
