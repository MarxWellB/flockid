# Phase 2 findings -- non-linear motion vs. the constant-velocity assumption

## Goal of this phase

Measure, with the tracker **unmodified** (the same one calibrated for the
conveyor), how badly it performs once motion stops being predictable --
exactly the condition expected with real birds.

## 1. Measured degradation (original config, nothing changed)

| Metric (normalized per individual) | Phase 1 (conveyor) | Phase 2 (random motion) |
|---|---|---|
| ID switches per individual | 0.66 | **2.44** (+270%) |
| Average fragmentation | 1.60 | 2.16 |

Confirmed: the constant-velocity assumption really does break down under
erratic motion, as predicted. Not a surprise -- it's the whole point of
this phase.

## 2. Unexpected finding: appearance made it worse

With the original config, turning on appearance-based re-identification
in Phase 2 raised ID switches (61 -> 68) instead of lowering them, the
opposite of what happened in Phase 1. Most likely explanation: once the
motion model is already failing (unreliable position predictions), the
association gate becomes less selective, and that's where a weak color
histogram (not very discriminative -- several grains share similar hues)
manages to "win" incorrect matches that position alone would have
rejected. Lesson: **a weak signal (color appearance) can add noise
instead of help once the strong signal (position) is already
unreliable.** This is not a flaw in "having appearance," it is a *quality*
problem with that appearance signal -- a real trained embedding (not a
histogram) would likely reverse this, but that should not be assumed
without measuring it.

## 3. Mitigation attempt: increase Kalman process noise

Raising `process_var` (how much the filter trusts the constant-velocity
model) does help in Phase 2:

| `process_var` | ID switches (Phase 2) | IDF1 (Phase 2) |
|---|---|---|
| 1 (original) | 61 | 81.6% |
| 5 | 49 | 85.6% |
| **15** | **45** | 84.7% |
| 40 | 46 | 84.1% |
| 80 | 49 | 82.9% |

`process_var=15` is a local optimum for random motion. With that plus
appearance enabled, the result improves further: IDF1 rises to 90.8%
(with some additional ID switches -- see the methodological note below on
why switches and IDF1 don't always move together).

## 4. The finding that actually matters: that improvement breaks Phase 1

```
Conveyor (Phase 1), process_var=1  -> id_switches=90,  IDF1=81.7%
Conveyor (Phase 1), process_var=15 -> id_switches=136, IDF1=78.6%   <- WORSE
```

**There is no single `process_var` value that is optimal for both
scenarios.** This is a real, actionable conclusion, not a failed
experiment: it means a fixed-parameter motion model is not enough for a
product that has to work both in predictable stretches (birds walking in
a line toward food) and unpredictable ones (birds standing still,
turning, lying down). This confirms exactly the concern flagged early in
the architecture document.

## 5. Methodological note: ID switches and IDF1 don't always agree

In point 3, with appearance enabled, ID switches rose (45->59) but IDF1
also rose (84.7%->90.8%). This is possible because IDF1 measures the
fraction of frames where the *correct* per-track association dominates,
not simply how many times the ID changed -- some switches can be
"corrective" (returning to the right identity) rather than destructive.
This is why the architecture document recommends HOTA in production: a
single metric (MOTA, or just the switch count) can lead to the wrong
conclusion if viewed in isolation.

## 6. What to do about this (a real recommendation, not aspirational)

A single hand-calibrated `process_var` does not scale to a real poultry
house, where different zones (near the feeder vs. resting) have very
different movement dynamics at the same time. The serious alternatives:

1. **IMM (Interacting Multiple Models):** run a "moving" model and a
   "still/erratic" model in parallel and blend their predictions
   probabilistically -- the standard tracking-literature answer to this
   exact problem. More implementation complexity, but the real fix.
2. **Per-track adaptive process_var:** raise process noise dynamically
   when a track's recent prediction error has been high (the filter
   "notices" its model isn't working). Simpler than IMM, a reasonable
   short-term approach.
3. **Rely more on appearance when motion is erratic -- but with a real
   embedding**, not a color histogram, since point 2 above shows a weak
   appearance signal can add noise instead of help exactly when it's
   needed most.

## 7. Addendum: implemented the adaptation (option 2) -- and it didn't work as expected

Per-track adaptive `process_var` via NIS (Normalized Innovation Squared)
was implemented, following the recommendation above. Along the way:

**First bug (found and fixed):** feeding NIS only from matches already
*accepted* by the gate biases the estimator -- by construction, an
accepted match can never have a NIS above the gate threshold (~9.2), so
the mechanism is blind to the evidence it would need to see. Fixed by
feeding NIS from the nearest available detection every frame, without
gate filtering.

**Result after the fix:** `q_scale` barely moves (mean 1.006, max 2.65)
-- well below what's needed. The actual NIS distribution in the random-
motion scenario was measured directly: median 0.16, 99th percentile just
2.1, max 5.3 across 400 frames. In other words: **frame-to-frame
prediction error is almost never large**, even in the "erratic" scenario.
The original hypothesis -- that the problem was large instantaneous
prediction error -- was wrong.

**Real explanation (revised):** at 3.5px/frame speed with gradual
direction changes, frame-to-frame prediction stays reasonably good most
of the time. What actually changes between the conveyor and random motion
is the **spatial configuration**: on the conveyor, grains are lane-
ordered, so there's rarely more than one plausible candidate near a
prediction. Under random motion, with 25 grains spread with no lane
structure, it's far more common for **several detections to land near the
same prediction at once** by chance -- that's where the tracker makes
mistakes, not from motion-model drift. Manually widening `process_var`
"helped" not because it corrected real error, but because it relaxed the
gate enough to avoid accidentally discarding the correct candidate in
those spatial-ambiguity situations.

**Honest conclusion:** single-motion-model NIS-based adaptation is not
the right tool for this problem -- it targets a symptom (prediction
error) that almost never occurs. The recommendation in section 6, item 1
(**IMM**) still holds, but for a different reason than assumed: IMM's
value here would not be "detecting when the velocity model fails," but
having an explicit notion of spatial uncertainty from candidate density,
not just temporal residual. The adaptive code stays in the repo (it does
no harm, conservative by default) but is not recommended as a solution --
it is evidence of a path that didn't work, and documenting it this way is
more useful than deleting it.

**What would be done differently next time:** before implementing a
mitigation, measure the real distribution of the signal that will drive
the adaptation (here: NIS) first -- that was done after implementing,
it should have been done before.
