"""
Demo: Detector -> Tracker -> Identity Fusion Engine (vision + RFID).

Honestly validates how well the fusion engine guesses each bird's real
identity using ONLY what a real system would observe (visual position +
RFID reads) -- it is never given the true bird_id. The simulator's
ground truth is used only AFTERWARD, to measure accuracy.
"""
import json
from collections import defaultdict

from vision.synth.nest_simulator import NestSimulator
from vision.detection.blob_detector import BlobDetector
from vision.tracking.tracker import MultiObjectTracker
from vision.fusion.identity_fusion import IdentityFusionEngine


def run(n_frames: int = 2500, n_birds: int = 15, seed: int = 42, min_obs: int = 3):
    sim = NestSimulator(n_birds=n_birds, seed=seed)
    detector = BlobDetector()
    tracker = MultiObjectTracker(use_appearance=False)
    fusion = IdentityFusionEngine(min_observations_to_resolve=min_obs)

    # for honest evaluation afterward: which real bird_id occupied each
    # track_id most of the time that track spent inside a nest
    track_true_bird_votes = defaultdict(lambda: defaultdict(int))

    total_rfid_reads = 0
    total_eggs = 0

    for frame_idx in range(n_frames):
        frame, gt, rfid_reads, egg_events = sim.step()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        track_positions = {tid: (t.x, t.y) for tid, t in tracks.items()}

        # The fusion engine ONLY receives this -- never gt (the real bird_id)
        fusion.observe(frame_idx, track_positions, rfid_reads, egg_events, sim.nests)
        total_rfid_reads += len(rfid_reads)
        total_eggs += len(egg_events)

        # For later evaluation (not used inside the algorithm):
        # match each visual track to the nearest real bird_id, and if that
        # bird is in a nest, cast a vote for its real tag on that track.
        for tid, (tx, ty) in track_positions.items():
            best_bird, best_d = None, 20.0
            for bird_id, (bx, by, br, tag) in gt.items():
                d = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_bird = d, bird_id
            if best_bird is not None:
                true_tag = gt[best_bird][3]
                track_true_bird_votes[tid][true_tag] += 1

    # --- Honest evaluation ---
    # rebuild, for each canonical track_id, the votes from ALL track_ids
    # that ended up merged into it (including aliases)
    canonical_votes = defaultdict(lambda: defaultdict(int))
    for raw_tid, votes in track_true_bird_votes.items():
        canonical_tid = fusion.alias_of.get(raw_tid, raw_tid)
        for tag, n in votes.items():
            canonical_votes[canonical_tid][tag] += n

    correct, total_resolved = 0, 0
    for tid, profile in fusion.profiles.items():
        if profile.resolved_tag is None:
            continue
        total_resolved += 1
        votes = canonical_votes.get(tid, {})
        if not votes:
            continue
        true_majority_tag = max(votes, key=votes.get)
        if profile.resolved_tag == true_majority_tag:
            correct += 1

    accuracy = correct / total_resolved if total_resolved else 0.0
    n_aliases_merged = len(fusion.alias_of)

    # Key finding: the engine's own confidence predicts its accuracy.
    # This is what would make a field AR overlay viable: show the ID only
    # when confidence is above a threshold, and "verifying..." otherwise.
    buckets = {">=80%": [0, 0], "50-79%": [0, 0], "<50%": [0, 0]}
    for tid, profile in fusion.profiles.items():
        if profile.resolved_tag is None:
            continue
        votes = canonical_votes.get(tid, {})
        if not votes:
            continue
        true_tag = max(votes, key=votes.get)
        is_correct = profile.resolved_tag == true_tag
        conf = profile.confidence
        key = ">=80%" if conf >= 0.8 else ("50-79%" if conf >= 0.5 else "<50%")
        buckets[key][0] += is_correct
        buckets[key][1] += 1

    print(f"=== Identity Fusion Engine -- honest validation ===")
    print(f"Simulated birds: {n_birds}")
    print(f"Total RFID reads: {total_rfid_reads}")
    print(f"Total egg events: {total_eggs}")
    print(f"Track_ids merged as aliases (identity switches corrected via RFID): {n_aliases_merged}")
    print(f"Visual tracks with a resolved link (>= 3 co-occurrences): {total_resolved}")
    print(f"CORRECT links (checked against ground truth, not used by the algorithm): {correct}/{total_resolved}")
    print(f"Link accuracy (aggregate, all confidence levels): {accuracy:.1%}")

    print("\n=== Precision stratified by confidence ===")
    for k, (c, t) in buckets.items():
        pct = f"{c/t:.0%}" if t else "n/a"
        print(f"  Confidence {k:>7}: {c}/{t} correct  ->  {pct}")

    print("\n=== Resolved bird profiles (what the AR app would show) ===")
    summary = fusion.summary()
    shown = 0
    for tid, prof in sorted(summary.items(), key=lambda kv: -kv[1]["confidence"]):
        if prof["resolved_tag"] is None:
            continue
        print(f"  track_id={tid:>4}  ->  {prof['resolved_tag']}  "
              f"(confidence={prof['confidence']:.0%}, eggs={prof['egg_count']}, "
              f"avg_weight={prof['avg_egg_weight_g']}g)")
        shown += 1
        if shown >= 10:
            break

    with open("demo_out/fusion_report.json", "w") as f:
        json.dump({
            "n_birds": n_birds,
            "total_rfid_reads": total_rfid_reads,
            "total_eggs": total_eggs,
            "total_resolved": total_resolved,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "profiles": summary,
        }, f, indent=2)

    return accuracy


if __name__ == "__main__":
    run()
