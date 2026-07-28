"""
Combines pre-annotations from two different clips and builds a YOLO
dataset split BY CLIP (not by random frame) -- clip 1 in full for
training, clip 2 in full for validation. This fixes the information-
leakage problem found earlier (near-identical frames from the same clip
ending up in both train and val, inflating the validation metric without
measuring real generalization).
"""
import json
import os
import shutil
import sys

sys.path.insert(0, "/home/claude/grain-tracking")
from vision.detection.watershed_detector import WatershedBeanDetector
import cv2


def detect_and_build_coco(frames_dir, start_img_id, start_ann_id, detector):
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    images, annotations = [], []
    img_id, ann_id = start_img_id, start_ann_id
    for fname in files:
        path = os.path.join(frames_dir, fname)
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h, "_src_dir": frames_dir})
        dets = detector.detect(frame)
        for d in dets:
            x0 = max(0.0, d.x - d.radius)
            y0 = max(0.0, d.y - d.radius)
            bw = min(w - x0, d.radius * 2)
            bh = min(h - y0, d.radius * 2)
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                 "bbox": [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)],
                                 "area": round(bw * bh, 1), "iscrowd": 0})
            ann_id += 1
        img_id += 1
    return images, annotations, img_id, ann_id


def write_yolo_split(images, annotations, out_dir, split_name):
    os.makedirs(os.path.join(out_dir, "images", split_name), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split_name), exist_ok=True)
    anns_by_image = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in images:
        src = os.path.join(img["_src_dir"], img["file_name"])
        dst = os.path.join(out_dir, "images", split_name, img["file_name"])
        shutil.copy(src, dst)
        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            x, y, bw, bh = ann["bbox"]
            xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        label_path = os.path.join(out_dir, "labels", split_name,
                                   os.path.splitext(img["file_name"])[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    detector = WatershedBeanDetector()
    out_dir = "/home/claude/real_video_check/dataset_yolo_v2"

    imgs1, anns1, next_img_id, next_ann_id = detect_and_build_coco(
        "/home/claude/real_video_check/extracted", 0, 1, detector)
    imgs2, anns2, _, _ = detect_and_build_coco(
        "/home/claude/real_video_check/clip2_usable", next_img_id, next_ann_id, detector)

    write_yolo_split(imgs1, anns1, out_dir, "train")
    write_yolo_split(imgs2, anns2, out_dir, "val")

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['grain']\n")

    print(f"Train (clip1, belt/paper bean1): {len(imgs1)} images, {len(anns1)} boxes")
    print(f"Val   (clip2, bean2, real held-out): {len(imgs2)} images, {len(anns2)} boxes")
    print(f"data.yaml at {out_dir}/data.yaml")
