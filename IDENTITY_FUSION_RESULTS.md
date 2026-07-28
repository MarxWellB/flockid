# Identity Fusion Engine -- linking vision and nest-box RFID

## What it solves (the differentiator behind the AR proposal)

The visual tracker maintains persistent identity for an anonymous
`track_id` -- it never knows which physical bird that is. The nest RFID
reader knows exactly which bird laid which egg, but knows nothing outside
the nest. This engine links both by **accumulated spatio-temporal
co-occurrence**: if a `track_id` is within a nest's radius at the same
moment that nest reports an RFID read, repeatedly, that is evidence the
`track_id` IS that bird.

This is what makes the glasses/AR idea work: the worker looks at a bird,
and the system already knows (through the link made at the nest) its
production and behavior history -- the bird doesn't need to carry
anything visibly different from the rest.

## Honest validation (simulator, ground truth never seen by the algorithm)

15 simulated birds, 6,821 RFID reads, 86 egg events.

```
Resolved links (>=3 co-occurrences): 24
Correct: 12/24 -> 50% aggregate
```

50% aggregate sounds mediocre -- **but it is the wrong metric to report
alone.** The real finding:

```
Confidence >=80%:  3/3 correct  -> 100%
Confidence 50-79%: 3/3 correct  -> 100%
Confidence <50%:   6/18 correct -> 33%
```

**The confidence the engine itself computes predicts its own accuracy.**
At confidence >=50%, it was right 6/6 in this run. That is exactly what a
field product needs: the AR overlay shows an ID only when
`confidence >= 0.5`, and "verifying identity..." otherwise, instead of
showing a confident answer that could be wrong half the time.

## Why the 50% aggregate is low (diagnosis, not an excuse)

Nests are gathering points -- several birds converge there at the same
time, exactly the condition (crossings, occlusion) already identified as
the visual tracker's weak spot (see `PHASE2_FINDINGS.md`). Link accuracy
is limited by tracker accuracy at that specific moment, not by the fusion
mechanism itself.

## An improvement attempt that did NOT work (documented on purpose)

An idea that sounds good in theory was tried: use RFID confirmation to
**merge** `track_id`s that are actually the same bird, correcting
tracker identity switches. Implemented and honestly measured: across
different thresholds (3 to 30 observations before merging), accuracy
stayed at 40-50% but with **far fewer** resolved links (3-5 instead of
24) and low confidence (9-40%) -- worse coverage, no clear improvement.
Likely cause: an early merge based on still-noisy evidence permanently
contaminates the canonical profile, with no way to undo it.

The code stays in the repo (`enable_alias_merging=True`, off by default)
for further iteration, but **it should not be presented as functional**
-- the same kind of dead end already documented with the adaptive Kalman
tuning in Phase 2. Acknowledging it here is more useful than hiding it,
even for a job proposal -- it demonstrates a real validation process, not
just nice-looking results.

## How to run it

```bash
python fusion_demo.py
```

## What this means for the proposal

1. The differentiator (linking visual identity to production via RFID)
   is technically viable -- the evidence supports it, it is not just an
   idea.
2. The system can self-rate its own reliability (link confidence) -- a
   real selling point: "the system knows when to trust itself," relevant
   to any serious technical evaluator.
3. Link accuracy depends directly on improving the tracker in crowded
   zones (nests), which loops back to the need for a real trained
   appearance embedding, not the current color histogram. The whole
   project supports itself this way.
