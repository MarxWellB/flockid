# Labeling workflow -- Label Studio

## Why Label Studio (and not manual labeling with another tool)

It was specified in the architecture document's MLOps section, supports
bounding boxes with direct export to formats YOLO can consume, and runs
locally (no dependency on uploading video to an external service, which
matters if the data belongs to a client).

## 1. Local install

```bash
pip install label-studio
label-studio start
```
Open `http://localhost:8080`, create a new project.

## 2. Labeling config (bounding boxes)

Paste this into the project's "Labeling Setup -> Custom template":

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="grain" background="#2ecc71"/>
    <Label value="bird" background="#e74c3c"/>
    <Label value="occluded_cluster" background="#f39c12"/>
  </RectangleLabels>
</View>
```

`occluded_cluster` is intentional: when two or more objects are so close
together that not even a human can confidently separate them into
individual bounding boxes, the group gets labeled as an occluded cluster
instead of forcing invented individual boxes. The detector will learn to
recognize "this is an occlusion," which is valuable information in itself
for the tracker (matches the occlusion handling already built into the
simulator).

## 3. Which frames to label (not all of them)

`scripts/extract_frames.py` extracts 1 out of every N frames (default
N=15, ~2 frames/second at 30fps) -- enough variety without labeling
thousands of near-identical frames. On top of that, manually prioritize
labeling frames that show:
- Objects partially out of frame (edges).
- Crossing/occlusion moments (the hardest and most important case).
- Slightly different lighting/angle, if any.

**Realistic goal for the first model:** 200-300 labeled frames is enough
for a first working fine-tuned YOLO11 (not for production, but enough to
validate that the training pipeline works end to end).

## 4. Export

From Label Studio: `Export -> COCO` (or `YOLO` if the installed version
supports it natively -- some do). If exporting COCO, use
`scripts/labelstudio_to_yolo.py` to convert to the format `ultralytics`
expects.

## 5. Train (when the time comes -- don't run this without a real dataset yet)

```bash
pip install ultralytics
yolo detect train data=dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

The resulting model (`runs/detect/train/weights/best.pt`) is what
`vision/detection/yolo_detector.py` needs to stop being a stub for a new
domain.

## 6. Labeling quality control

- At least 2 people label a 10% sample of the frames independently; if
  the average IoU between their boxes is low, the labeling guide (what
  counts as "occluded" vs. "two separate boxes") is unclear and needs
  refining BEFORE labeling the rest -- fixing this after everything is
  labeled is far more expensive.
- Freeze a validation set (10-15% of frames) that nobody uses to
  iteratively tune the model -- only to measure mAP/IDF1 at the end.
