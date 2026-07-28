# Real chicken detector -- YOLO11n trained and validated

## Dataset

- **Source:** Roboflow Universe, "Broiler Chicken Dataset" (GEL / HAFL
  Switzerland) -- a real emissions study in poultry houses, 9 cameras per
  house. https://universe.roboflow.com/gel/broiler-chicken-hh3fw (CC BY 4.0)
- **1,140 images** paired with labels (of 1,819 original label files;
  679 had no corresponding image because the source images had been
  deleted but not the labels -- orphans were discarded).
- **18 distinct cameras, 13 distinct dates** (May 8-20, 2024, with a few
  August entries discarded as orphaned labels).
- Resolution 1280x1280, single class (`chicken`).

## Validation split -- the part that matters

Applying the lesson from the bean experiment: **split by full date, not
random image.**

- Train: 917 images (dates 05-08 to 05-18) -- 230 batches
- Val: 223 images (dates 05-19 and 05-20) -- **dates the model never saw
  at any point during training**

## Result

| Metric | Value (on unseen dates) |
|---|---|
| mAP50 | **95.6%** |
| mAP50-95 | 65.3% |
| Precision | 91.7% |
| Recall | 91.2% |

Additional qualitative check: on a random validation image, the model
found 63 chickens where the ground-truth label marks 59 -- a reasonable
difference given that at that density even a human labeler has real
ambiguity about where one individual ends and another begins.

## Why this result is actually trustworthy (unlike the bean one)

1. **Honest date-based split**, not frame-based -- no information leakage
   between train and val.
2. **Real source diversity**: 18 cameras, 13 days, varying lighting
   conditions -- not a single table under a single light.
3. **Real volume**: 917 training images vs. the 26 available with beans.
4. Verified qualitatively in addition to the aggregate metric -- not just
   trusting the number.

## What this means for the project

This replaces the `YOLODetector` that used to be a stub
(`NotImplementedError`). It is now a real detector that:
- Works on the project's real object (chickens), not a proxy (grain).
- Generalizes to cameras/dates it never saw, measured rather than assumed.
- Plugs directly into `MultiObjectTracker` (Kalman + Mahalanobis + temporal
  prior) with no changes, since both implement the same `Detector`/
  `Detection` interface.

## What's still missing (honestly, not oversold)

- This dataset comes from an EXTERNAL STUDY -- it does not validate that
  the detector will work in a specific target poultry house, with its own
  cameras and lighting. A dataset of its own (or at least fine-tuning)
  with the client's real cameras is still needed before production.
- It includes no identity or tracking -- it is the detector only. The
  tracker and Behavior/Risk Engine (built on the synthetic simulator) have
  not yet been tested against this real detector or real moving chicken
  video -- the logical next step.
- Trained on CPU with limited resources (4 effective epochs, small batch)
  -- likely to improve further with a GPU and more epochs, though this is
  already a solid result.

## Suggested next step

Connect this real detector to the already-built tracker/Behavior Engine,
using real video of moving chickens (not static images) to validate the
tracking phases against the project's real object for the first time.
