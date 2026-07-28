"""
Improves pre-annotations using TEMPORAL CONSISTENCY: for each sampled
frame, runs the tracker over a short window of neighboring (consecutive)
frames and keeps only the detections that belong to a sustained track
(>=3 hits) within that window. A detection that appears once and never
again nearby in time is, with high probability, background noise (grass/
shadow that briefly resembled a chicken) -- exactly the false-positive
pattern already diagnosed.

This reduces the human correction workload (fewer false positives to
delete), but does NOT replace human review -- it still can't distinguish
a real, partially occluded chicken (which is also a sustained track) from
a background pattern that happens to move consistently for a few frames
(rare, but possible). This limitation is documented on purpose.
"""
import json
import os
import cv2
import sys

sys.path.insert(0, "/home/claude/grain-tracking")
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.tracker import MultiObjectTracker
from vision.reid.appearance import extract_batch

WINDOW = 6  # frames before and after the sampled frame
MIN_HITS = 3  # minimum hits within the window to trust the detection


def get_consistent_detections(video_path: str, center_frame_idx: int, conf: float = 0.2):
    cap = cv2.VideoCapture(video_path)
    start = max(0, center_frame_idx - WINDOW)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    detector = YOLODetector(confidence_threshold=conf, nms_iou=0.3)
    tracker = MultiObjectTracker(use_appearance=True, appearance_weight=60.0, max_age=25,
                            process_var=25.0, min_confidence_for_new_track=0.5)

    center_frame = None
    center_local_idx = center_frame_idx - start
    track_hits_at_center = {}

    for local_idx in range(WINDOW * 2 + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if local_idx == center_local_idx:
            center_frame = frame.copy()
        dets = detector.detect(frame)
        embeddings = extract_batch(frame, dets)
        tracks = tracker.update(dets, local_idx, embeddings)
        if local_idx == center_local_idx:
            for tid, t in tracks.items():
                track_hits_at_center[tid] = (len(t.history), t.x, t.y, t.radius)
    cap.release()

    # filter: only tracks with >= MIN_HITS accumulated up to the center frame
    consistent = [(x, y, r) for tid, (hits, x, y, r) in track_hits_at_center.items() if hits >= MIN_HITS]
    return center_frame, consistent


def process_video(video_path: str, frame_indices: list, img_id_start: int, ann_id_start: int, frames_out_dir: str, prefix: str):
    images, annotations = [], []
    img_id, ann_id = img_id_start, ann_id_start
    for fi in frame_indices:
        frame, dets = get_consistent_detections(video_path, fi)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        fname = f"{prefix}_f{fi:05d}.jpg"
        cv2.imwrite(os.path.join(frames_out_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for x, y, r in dets:
            x0, y0 = max(0.0, x - r), max(0.0, y - r)
            bw, bh = min(w - x0, r * 2), min(h - y0, r * 2)
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                 "bbox": [round(x0, 1), round(y0, 1), round(bw, 1), round(bh, 1)],
                                 "area": round(bw * bh, 1), "iscrowd": 0})
            ann_id += 1
        img_id += 1
        print(f"  {fname}: {len(dets)} consistent detections")
    return images, annotations, img_id, ann_id
