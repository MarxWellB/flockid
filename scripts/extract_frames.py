"""
Extrae frames muestreados de un video real para etiquetar en Label Studio.

Uso:
    python scripts/extract_frames.py --video mi_video.mp4 --out dataset/raw_frames --every 15
"""
import argparse
import os
import cv2


def extract_frames(video_path: str, out_dir: str, every: int = 15, prefix: str = None):
    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix or os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path} | {total_frames} frames | {fps:.1f} fps")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every == 0:
            out_path = os.path.join(out_dir, f"{prefix}_frame{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        frame_idx += 1
    cap.release()

    print(f"Guardados {saved} frames en {out_dir} (1 de cada {every})")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta al video de origen")
    parser.add_argument("--out", default="dataset/raw_frames", help="Carpeta de salida")
    parser.add_argument("--every", type=int, default=15, help="Guardar 1 de cada N frames")
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.every)
