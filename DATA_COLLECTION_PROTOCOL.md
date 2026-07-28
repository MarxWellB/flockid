# Capture protocol -- first real dataset

## Why this matters more than continuing to tune the simulator

Everything built so far (tracker, Behavior Engine, Risk Engine) is
validated against synthetic data. That proves the **architecture**
works -- it does not prove the **detector** works on real images, which
is the requirement everything else depends on. That is the project's real
bottleneck right now, not the code.

## What to record (in this order, each a separate clip)

Based on the original grain-MVP progression:

1. **A few separated objects** (5-10 grains/beans, well spaced) -- 30s.
2. **Many objects** (fill the belt/surface) -- 30s.
3. **Crossing objects** (two streams crossing) -- 30s.
4. **Grouped/touching objects** (real occlusion) -- 30s.
5. **Variable belt speed** (if applicable) -- 15s per speed, 3 speeds.
6. If two cameras are available: the same footage from two angles with
   partial field-of-view overlap -- 30s.

**Minimum total: ~10 minutes of raw video.** No more is needed to start
labeling -- 10 well-varied minutes are more useful than 2 repetitive
hours.

## Technical specs (for the detector to train well)

| Parameter | Recommendation | Why |
|---|---|---|
| Resolution | 1080p (1920x1080) minimum | YOLO rescales anyway, but higher resolution helps with small objects |
| FPS | 30 fps | Enough for the tracker; no need for more |
| Lighting | Fixed, no flicker, no shifting hard shadows | Moving shadows confuse both the detector and the tracker |
| Background | Uniform, contrasting color with the objects | Helps both the current blob detector and a future YOLO model |
| Camera | Fixed (tripod), overhead or near-overhead | Same as the simulator -- a different angle is a domain shift |
| Format | MP4 (H.264) | Universal compatibility with OpenCV/ffmpeg |

## What to avoid (common mistakes that ruin a dataset)

- Recording handheld with a phone (introduces shake the tracker will read
  as object motion).
- Changing lighting between clips (natural light shifting with time of day).
- Patterned or textured backgrounds -- the detector can mistakenly "learn"
  the background if contrast with the objects is low.
- Recording everything as one continuous take without the variations
  listed above -- the dataset needs deliberate variety, not just volume.

## What to do with the video once recorded

1. Extract frames with `scripts/extract_frames.py` at a reasonable
   sampling rate for labeling -- full video is not labeled, a
   representative sample of frames is.
2. Label those frames in Label Studio (see `LABELING_WORKFLOW.md`).
3. Train YOLO11 on the result -- this is where `YOLODetector` (currently
   a stub for anything beyond the trained chicken model) becomes real for
   a new domain.

## Note on real birds (when that point is reached)

This whole protocol applies the same way to birds, with two differences:
- Higher / wider-angle camera (full pen in frame).
- Longer sessions due to behavioral variability (a bird staying still for
  10 minutes is normal; a grain staying still for 10 minutes never
  happens) -- likely 30-60 minutes of raw video needed, not 10.
