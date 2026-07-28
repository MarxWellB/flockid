"""
Generates COCO-format pre-annotations using WatershedBeanDetector over a
directory of already-extracted frames, to import into Label Studio and
CORRECT instead of drawing boxes from scratch ("model-assisted labeling").

Uso:
    python scripts/generate_preannotations.py --frames dir_con_frames --out preann.json
"""
import argparse
import json
import os
import cv2

from vision.detection.watershed_detector import WatershedBeanDetector


def generate(frames_dir: str, out_path: str):
    detector = WatershedBeanDetector()
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    images = []
    annotations = []
    ann_id = 1
    counts = []

    for img_id, fname in enumerate(files):
        path = os.path.join(frames_dir, fname)
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        detections = detector.detect(frame)
        counts.append(len(detections))
        for d in detections:
            x0 = max(0.0, d.x - d.radius)
            y0 = max(0.0, d.y - d.radius)
            bw = min(w - x0, d.radius * 2)
            bh = min(h - y0, d.radius * 2)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1), "iscrowd": 0,
                "score": d.confidence,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "grain"}],
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Frames processed: {len(files)}")
    print(f"Detections per frame: min={min(counts)} max={max(counts)} average={sum(counts)/len(counts):.1f}")
    print(f"Pre-annotations saved to: {out_path}")
    print("IMPORTANT: a starting point to CORRECT in Label Studio, not a final dataset.")
    print("We know there are 18 real beans per frame -- check especially the frames")
    print("with fewer detections (heavy clustering), that's where the most correction is needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", default="preannotations.json")
    args = parser.parse_args()
    generate(args.frames, args.out)
