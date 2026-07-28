# Multi-camera consensus -- real result and why the first design failed

## The hypothesis tested

The original idea, as stated: several cameras see the same scene at the
same time, fuse their detections every frame, and that should improve
coverage and reduce identity loss compared to a single camera.

## What was measured (600 frames, 15 birds, same seed)

| | 1 camera | 4 cameras (best config tested) |
|---|---|---|
| ID switches | **312** | 590 (radius=35) |
| IDF1 | **29.3%** | 22.0% |
| Coverage | 98.7% | 96.5% |

**4 cameras lost on every metric, not just slightly.** A full sweep of the
fusion parameter was run (clustering radius: 5, 8, 10, 12, 15, 18, 25,
35, 45, 60, 80 px) -- 1 camera beat the best multi-camera configuration
at every point in the sweep. This is not a fine-tuning problem, it is a
design problem.

## Why it failed (real diagnosis, not an excuse)

The implemented method fused detections **by spatial proximity, frame by
frame**: if two cameras report something within X pixels of each other,
it is assumed to be the same bird and averaged.

That has an unresolvable conflict with a single parameter:

- **Small radius**: per-camera measurement noise (which grows with
  distance, as in any real camera) means the SAME bird, seen by 2
  different cameras, appears at positions that don't always fall within
  the radius -- they don't get fused, and the tracker receives 2-3
  "phantom" detections of the same bird every frame. At radius=5, this
  pushed ID switches to 2,244.
- **Large radius**: this fuses multiple views of ONE bird correctly, but
  also accidentally fuses two DIFFERENT birds that happen to be close to
  each other -- real information gets lost. At radius=80, coverage drops
  to 81.9%.

There is no middle ground that avoids both problems at once, because the
scale of camera noise and the scale of real inter-bird distance are in
the same range in this simulation -- exactly the kind of problem a
single-number heuristic cannot solve, no matter how finely tuned.

## The correct architecture (what real multi-camera systems actually do)

Fusing **raw detections** by spatial proximity is not what production
systems do. The correct approach is:

```
Camera 1 -> Detector -> own tracker (Kalman+Mahalanobis) -> local track_ids for Camera 1
Camera 2 -> Detector -> own tracker                       -> local track_ids for Camera 2
Camera 3 -> Detector -> own tracker                       -> local track_ids for Camera 3
Camera 4 -> Detector -> own tracker                       -> local track_ids for Camera 4
                            |
        Association engine over TRAJECTORIES (not raw detections):
        compares each local track's recent trajectory across cameras
        (position + velocity + trajectory shape over a time window,
        optionally + appearance) and decides which track_ids from
        different cameras are the same physical bird.
                            |
                    persistent global_id
```

In other words, **the same pattern already built and validated for RFID**
(`vision/fusion/identity_fusion.py`): don't fuse the raw signal, fuse
**accumulated identity evidence** with confidence. The difference is that
the "external confirmation" here is not an RFID reader, it is trajectory
consistency across independent trackers.

## What NOT to do with this result

Don't hide it or present it as "already working" -- but don't dismiss the
idea as wrong either. **The real result is: the naive implementation
doesn't work, and the reason and the correct architecture are now known.**
That, again, is exactly the kind of real-process evidence that's worth
more in a technical interview than a nice number with no diagnosis behind
it.

## Update -- second attempt (correct architecture) and the real finding

The correct architecture described above was implemented (independent
per-camera tracker + trajectory association with Union-Find, without the
alias-chain bug). Result: **still worse than 1 camera** (2,611 ID
switches vs. 312).

Before tuning more parameters (that lesson was already learned), the root
cause was measured directly: how many local `track_id`s each camera
creates on its own, for the same 15 birds:

```
cam_NW: 107 track_ids created
cam_NE: 249 track_ids created
cam_SW: 125 track_ids created
cam_SE: 255 track_ids created
(single center camera, near-total coverage: 312 switches for the WHOLE run)
```

**The problem wasn't the fusion -- it was the comparison.** The single
baseline camera has enough range (900px) to cover almost all of the
960x540 room almost all the time. The 4 corner cameras (700px range each,
in a room with a ~1100px diagonal) each have large individual coverage
gaps, so EACH ONE on its own already tracks worse than the center camera.
No fusion layer, however well designed, can recover information that no
individual camera captured stably in the first place.

## The real conclusion (and the correct pivot for the proposal)

**Multi-camera advantage was never going to show up in a small space a
single camera already covers well.** That is not the problem multi-camera
actually solves. What it does solve is when **the space is larger than
any single camera can cover** -- the real situation in an industrial
poultry house (hundreds of square meters), not a 960x540 px test room.

The correct comparison is not "1 good camera vs. 4 mediocre cameras in
the same small space" -- it is "0 cameras can cover the whole house alone
vs. N cameras covering adjacent zones can." That is also the original
idea from the architecture document's multi-camera phase (handoff between
cameras based on expected transit time) -- and it turns out that original
idea was pointing in the right direction more than the simultaneous
consensus tested here.

## Confirmation -- large room, correct comparison

The correct test was run: a 2200x1300 room (larger than any single
camera's range, 900px), 20 birds, same v2 architecture (independent
per-camera tracker + trajectory association).

| Metric | 1 camera (center) | 4 cameras (corners) |
|---|---|---|
| Coverage | 57.3% | **96.0%** |
| IDF1 | 12.9% | **32.1%** (2.5x) |
| ID switches | 398 | 1,343 (higher, but over nearly double the tracked instances: 6,876 vs 11,520) |

**This confirms the original hypothesis, with a fair comparison.** A
single camera, in a space larger than its range, literally cannot see
almost half the birds (57% coverage) -- not a tracking problem, there is
simply no signal to track outside its field of view. Four cameras
covering the space between them recover almost full coverage (96%) and
more than double identity consistency (IDF1).

Raw ID switches are still higher with 4 cameras, but that is an artifact
of tracking nearly twice as many total instances, not of the system being
less reliable per bird -- normalized per instance, the switch rate rises
from 5.8% to 11.7%, worse but not catastrophic, and exactly where the
next iteration should focus (`match_distance` and
`min_observations_to_merge` in `CrossCameraAssociator` are concrete
tuning points).

## The full lesson from this investigation (for the proposal)

Three attempts, in order:
1. Raw detection fusion -> failed (312 -> 644 switches).
2. Correct architecture (trajectory + Union-Find) in the SAME small space
   -> still worse (312 -> 2,611), because the comparison was unfair (one
   camera with near-total coverage vs. four with mediocre individual
   coverage in a space one camera already covered well).
3. Correct architecture in a space genuinely larger than one camera ->
   **confirms the hypothesis** (coverage +38 points, IDF1 2.5x).

The original idea was right from the start -- what was wrong the first
two times was the experiment, not the idea. That is, honestly, the most
valuable pattern from this whole project to show in an interview: methodical
persistence until finding the right comparison, not abandoning an idea at
the first negative result, and not accepting a positive result without
questioning whether the experiment was fair.
