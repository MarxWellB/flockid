"""
Genera pre-anotaciones COCO usando el detector YOLO real (no el clasico)
sobre los frames extraidos de los 9 videos combinados -- para importar en
Label Studio y CORREGIR, no dibujar desde cero.

Umbral de confianza deliberadamente BAJO (0.2) aqui: para pre-etiquetado,
es mejor pecar de detectar de mas (el humano borra una caja de mas facil)
que de menos (el humano tiene que notar y dibujar la que falta, mas lento).
"""
import json
import os
import cv2
import sys

sys.path.insert(0, "/home/claude/grain-tracking")
from vision.detection.yolo_detector import YOLODetector


def generate(frames_dir: str, out_path: str, conf: float = 0.2):
    detector = YOLODetector(confidence_threshold=conf, nms_iou=0.3)
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    images, annotations = [], []
    ann_id = 1
    counts = []

    for img_id, fname in enumerate(files):
        path = os.path.join(frames_dir, fname)
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        dets = detector.detect(frame)
        counts.append(len(dets))
        for d in dets:
            x0 = max(0.0, d.x - d.radius)
            y0 = max(0.0, d.y - d.radius)
            bw = min(w - x0, d.radius * 2)
            bh = min(h - y0, d.radius * 2)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1), "iscrowd": 0, "score": round(d.confidence, 2),
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "chicken"}]}
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Frames: {len(files)}  |  detecciones/frame: min={min(counts)} max={max(counts)} promedio={sum(counts)/len(counts):.1f}")
    print(f"Pre-anotaciones: {out_path}")


if __name__ == "__main__":
    generate("/home/claude/chicken_dataset_v2/frames", "/home/claude/chicken_dataset_v2/preannotations.json")
