# Real generalization test -- result and what it means

## What was done differently this time

Previous training runs split train/val by random FRAME within the same
clip -- that is data leakage (near-identical frames end up on both sides)
and explains the 74.8% "validation" mAP seen earlier, an inflated number
with no real meaning.

This time: **training = full clip 1 (26 frames, 18 beans, white paper),
validation = full clip 2 (4 usable frames, 35 beans, wooden table,
different light, moving camera)**. A clip the model never saw in any
frame during training.

## Result

```
mAP50 on validation (clip 2, never seen): 0.0000
```

Across the 32 completed epochs (of 50 planned, cut short by the
environment's time limit), mAP stayed at zero for the entire run, with
one insignificant exception (0.00002 at epoch 32 -- numerical noise, not
real signal).

## Why this happened (diagnosis, not an excuse)

The model learned to recognize beans **under the specific conditions of
clip 1**: that white paper, that light, that camera angle. Shown a wooden
table under different lighting (clip 2), it recognizes nothing as
"grain" -- not because the concept is hard, but because 26 images from a
SINGLE scenario give the model no variation to learn from about what's
invariant (the bean's shape/texture) versus what's incidental (background,
lighting).

This is exactly what machine learning theory predicts for a single-
condition dataset, and it is why the original capture protocol called for
variety -- not out of generic caution, but because without it this was
going to happen.

## What this result does NOT mean

It does not mean YOLO11 is the wrong tool, or that the approach is flawed.
It means **26 images from a single scenario were never going to be enough
to generalize** -- a data quantity/variety limitation, not a method
limitation.

## What this means for the project, concretely

- The detector trained so far (`bean_v2`, the one with 74.8% "validation")
  **does not work outside the exact conditions of clip 1**. It should not
  be used as if it generalizes.
- For YOLO to actually generalize, real variety is needed: different
  surfaces, different lighting, different angles -- not just more beans on
  the same table under the same light.
- The realistic number mentioned from the start (200-300 labeled frames)
  assumed exactly that variety -- 2 clips of 8-9 seconds each were never
  going to reach that, and now there is a measured proof of it, not just
  the theoretical warning.

## Honest next step

Record more clips with deliberate variation: different surfaces (not just
white paper), different rooms/lighting, a fixed camera (no motion, which
introduced blur in clip 2). With 4-5 distinct scenarios like that, it
becomes worth retrying this same generalization test -- with a realistic
expectation that it takes several iterations, not one.
