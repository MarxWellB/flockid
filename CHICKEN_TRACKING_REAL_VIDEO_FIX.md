# How to improve chicken detection -- real plan, ordered by impact

## 1. Already done today (code, no new data) -- confirmed improvement

**Two-tier confidence threshold (ByteTrack-style) in the custom tracker:**
high confidence required to CREATE a new identity, lower confidence
sufficient to MAINTAIN one that already exists. Directly attacks the
frame-to-frame confidence flicker diagnosed across 6 real videos.

| Video | Before | After |
|---|---|---|
| Mixed clip | 77 | **55** (-29%) |
| White #2 | 142 | **99** (-30%) |

Consistent improvement across two different videos -- not a fine-tune for
one case, a real principle (see `vision/tracking/tracker.py`,
`min_confidence_for_new_track`).

## 2. What's already known NOT to be the fix (so it isn't repeated)

- **A fixed confidence threshold doesn't transfer across videos** -- the
  first white-chicken video (covered farm) needed a much lower threshold
  than the mixed grass clip. Every scene has its own confidence
  distribution.
- **"Same color = better tracking" doesn't hold in general** -- it depends
  on contrast against that specific background, not color in the
  abstract.
- **Color-histogram appearance has a low ceiling** -- it helps a little,
  it doesn't fix the underlying problem.

## 3. The real fix, with its own evidence for why it works

**Fine-tuning on real target-domain data -- with the right variety, not
just quantity.** This was already tested twice with opposite results
that spell out exactly what's needed:

- **Beans**: 26 images from ONE clip -> 0% generalization on a new clip.
- **Chickens (GEL)**: 917 images from 18 cameras and 13 distinct dates ->
  95.6% mAP on unseen dates.

The difference wasn't quantity -- it was **variety of conditions**. For
this to work with the project's own real-world video clips, frames from
SEVERAL of the available clips need to be labeled (not just one),
covering different backgrounds/lighting/breeds, fine-tuning from the
weights that already exist (not from scratch).

## 4. Concrete steps, in order

1. **Already closed**: two-tier confidence threshold in the tracker (done).
2. **Next, cheap**: use a YOLO-based pre-annotation script over frames
   from the available real video clips to auto-label them.
3. **Label 150-300 frames spread across multiple clips** (not just one)
   in Label Studio, correcting the pre-annotations.
4. **Fine-tune** from `chicken_yolo11n.pt` (not from scratch) on that
   combined dataset.
5. **Validate with an honest split** -- by source video, not random frame
   (the recurring lesson): train on most clips, validate on the one clip
   the model never saw.
6. If the end target is cage-free layers specifically, add real cage-free
   layer frames to that dataset once available (e.g. via an academic data
   request, or a dedicated capture session).

## 5. The dedicated follow-up project for this exact bottleneck

Steps 2-5 above describe, in miniature, a full human-in-the-loop data
pipeline: auto-labeling, correction, iterative retraining, and a proper
train/val split. **FlockTrack Copilot** is the dedicated project built to
do this systematically and at scale -- turning real poultry-house video
into a specialized detection/tracking dataset through auto-labeling,
multimodal LLM review, manual correction of bounding boxes, active
learning to prioritize the next frames worth labeling, and iterative
detector retraining. It is the concrete next step for training a real
bird-identification network, not a separate, unrelated idea -- it exists
specifically because this document identified real detection failures on
real video and traced them back to the same root cause found with beans:
not enough labeled variety.

## What's not worth doing anymore

Continuing to tune tracker thresholds/parameters on the same handful of
video clips already has diminishing returns -- the available code-level
improvement has already been captured (step 1). The next real quality
jump only comes from labeled data, not more parameters.
