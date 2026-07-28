"""
Converts a Label Studio COCO export to the format `ultralytics`
(YOLO11) expects: one .txt per image with lines `class x_center y_center w h`
(normalized 0-1), plus a data.yaml.

Uso:
    python scripts/labelstudio_to_yolo.py \\
        --coco export/result.json \\
        --images export/images \\
        --out dataset/yolo \\
        --val-split 0.15
"""
import argparse
import json
import os
import random
import shutil


def convert(coco_path: str, images_dir: str, out_dir: str, val_split: float = 0.15, seed: int = 42):
    with open(coco_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    class_names = [categories[k] for k in sorted(categories)]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images_by_id.keys())
    random.Random(seed).shuffle(image_ids)
    n_val = int(len(image_ids) * val_split)
    val_ids = set(image_ids[:n_val])

    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    n_converted = 0
    for img_id, img in images_by_id.items():
        split = "val" if img_id in val_ids else "train"
        file_name = os.path.basename(img["file_name"])
        w, h = img["width"], img["height"]

        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(out_dir, "images", split, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: image not found: {src_path} -- skipping")
            continue

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = categories[ann["category_id"]]
            class_idx = class_to_idx[cat_name]
            x, y, bw, bh = ann["bbox"]  # COCO: x_min, y_min, width, height (pixels)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            label_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = os.path.join(out_dir, "labels", split,
                                   os.path.splitext(file_name)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
        n_converted += 1

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"Converted {n_converted} images. data.yaml at {data_yaml}")
    print(f"Classes: {class_names}")
    print(f"Train/val: {len(image_ids) - n_val}/{n_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to the Label Studio COCO export JSON")
    parser.add_argument("--images", required=True, help="Folder with the original images")
    parser.add_argument("--out", default="dataset/yolo", help="Output folder in YOLO format")
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()
    convert(args.coco, args.images, args.out, args.val_split)
