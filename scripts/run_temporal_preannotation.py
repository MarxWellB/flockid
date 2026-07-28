import sys, os, json
sys.path.insert(0, "/home/claude/grain-tracking")
from scripts.improve_preannotations_temporal import process_video

VIDEOS = {
    "106763-674268599_medium.mp4": 893,
    "111283-690770686_medium.mp4": 396,
    "190142-887464235_medium.mp4": 1335,
    "306119_medium.mp4": 501,
    "4458054-uhd_3840_2160_24fps.mp4": 262,
    "44626-439940248_medium.mp4": 1043,
    "62071-502737672_medium.mp4": 182,
    "7129-198606864_medium.mp4": 1243,
    "8386-208562358_medium.mp4": 234,
}

OUT_DIR = "/home/claude/chicken_dataset_v3"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

N_PER_VIDEO = 10


def frame_indices_for(n_total, n_pick, margin=8):
    lo, hi = margin, n_total - margin
    if hi <= lo:
        return [n_total // 2]
    step = (hi - lo) // n_pick
    return [lo + i * step for i in range(n_pick)]


if __name__ == "__main__":
    video_name = sys.argv[1]
    img_id_start = int(sys.argv[2])
    ann_id_start = int(sys.argv[3])

    n_total = VIDEOS[video_name]
    indices = frame_indices_for(n_total, N_PER_VIDEO)
    prefix = os.path.splitext(video_name)[0][:20]
    path = f"/mnt/user-data/uploads/{video_name}"

    print(f"Procesando {video_name}: {len(indices)} frames con ventana temporal...")
    images, annotations, next_img_id, next_ann_id = process_video(
        path, indices, img_id_start, ann_id_start, FRAMES_DIR, prefix)

    partial_path = os.path.join(OUT_DIR, f"partial_{prefix}.json")
    with open(partial_path, "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "next_img_id": next_img_id, "next_ann_id": next_ann_id}, f)
    print(f"Guardado: {partial_path} (next_img_id={next_img_id}, next_ann_id={next_ann_id})")
