# Behavior Engine -- minimum viable version

## Honest scope -- read this before the numbers

This runs on **synthetic grain**, not real birds. A grain object doesn't
"isolate itself" or "feed" in any biological sense. The point of this
module is not to demonstrate real avian behavior -- it is to validate
that the **engine** works end to end: it consumes trajectories from any
tracker (it only needs `TrackState`, agnostic to which tracker produced
them), computes per-individual and per-scene signals, and emits events
with the same schema (`event_type`, `entity_id`, `confidence`,
`evidence`) that Audio Engine, Environmental Engine, and Risk Engine
consume downstream.

Once real bird tracking exists, the math doesn't change -- what changes
is which zones get defined (real feeders/waterers) and which thresholds
are clinically relevant. That requires real data and a poultry expert,
not an engineering decision -- which is why the thresholds here
(`isolation_radius=80px`, `low_activity_threshold=0.3px/frame`) are
placeholders, not calibrated values.

## Implemented signals (6 of the 10 in the full architecture list)

- Activity (average speed)
- Isolation (nearest-neighbor distance, sustained over time)
- Zone occupancy (a generalization of feeding/watering)
- Space usage (cumulative heatmap)
- Repetitive movement (autocorrelation)
- Sustained low activity (a generic proxy for "lying down")

Left out (require pose estimation or real data): gait/lameness, complex
social patterns (co-occurrence graphs), posture.

## Two real bugs found and fixed while building this

**1. Isolation event flooding.** The first version fired an event every
frame while the condition held -- 2,600 "events" across 400 frames,
useless for any real dashboard. Fixed with the same debounce pattern
already used for `low_activity` (one event per sustained episode of
>=15 frames, not one per frame). After the fix: 35 events (conveyor) / 69
events (random motion) across 400 frames -- numbers that can actually be
inspected one by one.

**2. A "repetitiveness" metric with a trend artifact.** The first version
measured autocorrelation on raw position. For an object moving in a
straight line (the conveyor), position has a near-linear trend, and
**any** trending series produces high autocorrelation at almost every
lag -- not because of a real repeating pattern, but as a mathematical
artifact. This was caught because the score came out at ~0.98 for nearly
every conveyor track, which made no sense. Fix: measure autocorrelation
on **velocity** (the position difference) instead, which removes the
trend. After the fix, the conveyor (straight-line motion) gives lower
values (~0.73) than random motion with genuine back-and-forth reversals
(~0.92-0.93) -- consistent with what would be expected.

## How to run it

```bash
python behavior_demo.py
```

For each scenario (conveyor and random motion), it generates:
- `{scenario}_behavior_report.json` -- per track: average speed, zone
  occupancy, repetitiveness score; full event list.
- `{scenario}_heatmap.png` -- cumulative space-usage heatmap.

## What's missing for this to be a real product

- Thresholds calibrated with real data, not placeholder values.
- Real zones (feeder/waterer) instead of left/right halves of the frame
  -- that comes with real cameras and a real house.
- Publishing these events to a real event bus (Redis Streams/Kafka)
  instead of a flat JSON file -- trivial to wire up, not done here
  because there is no messaging infrastructure running in this
  development environment.
- Connecting the output to Risk Engine v1 (weighted scoring) -- the
  natural next step.
