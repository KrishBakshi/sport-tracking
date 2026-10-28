from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO, YOLOWorld

VIDEO_IN = Path("data/test-fight-1.mp4")
VIDEO_OUT = Path("output/annotated.mp4")
VIDEO_TMP = Path("output/annotated.tmp.mp4")

POSE_WEIGHTS = "yolo26m-pose.pt"
OV_WEIGHTS = "yolov8m-worldv2.pt"
OV_CLASSES = ["person", "fighter"]
MAX_FIGHTERS = 2
DET_CONF_MIN = 0.25
KPT_CONF_MIN = 0.3

POSE_EVERY = 1
DET_EVERY = 1

# COCO-17 pose skeleton edges (keypoint index pairs)
POSE_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def load_pose_model(weights: str = POSE_WEIGHTS) -> YOLO:
    return YOLO(weights)


def load_open_vocab_model(
    weights: str = OV_WEIGHTS,
    classes: list[str] | None = None,
) -> YOLOWorld:
    model = YOLOWorld(weights)
    model.set_classes(classes or OV_CLASSES)
    return model


def average_metrics(latencies_ms: list[float]) -> dict:
    """Average latency + throughput from a list of per-call latencies (ms)."""
    if not latencies_ms:
        return {
            "calls": 0,
            "avg_latency_ms": 0.0,
            "avg_throughput_fps": 0.0,
            "total_time_s": 0.0,
        }

    n = len(latencies_ms)
    avg_latency_ms = sum(latencies_ms) / n
    total_time_s = sum(latencies_ms) / 1000.0
    avg_throughput_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "calls": n,
        "avg_latency_ms": avg_latency_ms,
        "avg_throughput_fps": avg_throughput_fps,
        "total_time_s": total_time_s,
        "latencies_ms": latencies_ms,
    }


def detect_fighters(
    ov_model: YOLOWorld,
    frame,
    latencies_ms: list[float],
    max_fighters: int = MAX_FIGHTERS,
) -> list[tuple[int, int, int, int]]:
    """Open-vocab detect; return up to max_fighters boxes by confidence."""
    t0 = time.perf_counter()
    results = ov_model.predict(frame, conf=DET_CONF_MIN, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(elapsed_ms)

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    order = np.argsort(-conf)[:max_fighters]

    h, w = frame.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        px1 = int(np.clip(x1, 0, w - 1))
        py1 = int(np.clip(y1, 0, h - 1))
        px2 = int(np.clip(x2, 0, w - 1))
        py2 = int(np.clip(y2, 0, h - 1))
        if px2 > px1 and py2 > py1:
            boxes.append((px1, py1, px2, py2))
    return boxes


def pose_on_crop(
    pose_model: YOLO,
    crop,
) -> tuple[np.ndarray | None, float]:
    """Run pose on a crop; return best person's keypoints (Nx2) + conf, or None."""
    results = pose_model.predict(crop, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, 0.0

    conf = result.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(conf))
    best_conf = float(conf[best_i])
    kpts = result.keypoints.xy.cpu().numpy()[best_i]
    return kpts, best_conf


def draw_keypoints(
    frame,
    kpts: np.ndarray,
    origin: tuple[int, int],
    color: tuple[int, int, int],
):
    """Draw keypoints + skeleton in full-frame coords (origin = crop top-left)."""
    ox, oy = origin
    points: list[tuple[int, int] | None] = []
    for x, y in kpts:
        if x <= 0 and y <= 0:
            points.append(None)
            continue
        pt = (int(x + ox), int(y + oy))
        points.append(pt)
        cv2.circle(frame, pt, 3, color, -1)

    for a, b in POSE_SKELETON:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, pa, pb, color, 2)


def draw_det_boxes(frame, bboxes: list[tuple[int, int, int, int]]):
    out = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"fighter {i + 1}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


def estimate_pose(
    pose_model: YOLO,
    frame,
    det_bboxes: list[tuple[int, int, int, int]],
    latencies_ms: list[float],
):
    """Crop each det bbox -> YOLO pose -> keep highest-conf person; draw keypoints only."""
    if not det_bboxes:
        return frame

    out = draw_det_boxes(frame, det_bboxes)
    colors = ((0, 255, 0), (0, 165, 255))

    t0 = time.perf_counter()
    for i, (x1, y1, x2, y2) in enumerate(det_bboxes):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        kpts, conf = pose_on_crop(pose_model, crop)
        if kpts is None or conf < KPT_CONF_MIN:
            continue

        draw_keypoints(out, kpts, (x1, y1), colors[i % len(colors)])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(elapsed_ms)
    return out


def open_video(path: Path) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """Open input + OpenCV writer.

    Pip OpenCV cannot use avc1/libx264 (falls back to broken h264_v4l2m2m),
    so we write MPEG-4 Part 2 (mp4v) to a temp file and re-encode later.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    VIDEO_TMP.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(VIDEO_TMP),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {VIDEO_TMP}")
    return cap, writer


def reencode_h264(src: Path, dst: Path = VIDEO_OUT) -> Path:
    """Re-encode to H.264 with the project-venv ffmpeg (imageio-ffmpeg)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg H.264 encode failed:\n{result.stderr}")
    if src.resolve() != dst.resolve() and src.exists():
        src.unlink()
    return dst


def process_video(
    pose_model: YOLO,
    ov_model: YOLOWorld,
    video_in: Path = VIDEO_IN,
) -> tuple[Path, dict]:
    cap, writer = open_video(video_in)
    frame_i = 0
    det_bboxes: list[tuple[int, int, int, int]] = []

    pose_latencies_ms: list[float] = []
    det_latencies_ms: list[float] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_i % DET_EVERY == 0:
                det_bboxes = detect_fighters(ov_model, frame, det_latencies_ms)

            if frame_i % POSE_EVERY == 0:
                annotated = estimate_pose(
                    pose_model, frame, det_bboxes, pose_latencies_ms
                )
            else:
                annotated = frame

            writer.write(annotated)
            frame_i += 1
    finally:
        cap.release()
        writer.release()

    out_path = reencode_h264(VIDEO_TMP, VIDEO_OUT)

    metrics = {
        "pose": average_metrics(pose_latencies_ms),
        "open_vocab": average_metrics(det_latencies_ms),
        "frames": frame_i,
    }
    return out_path, metrics


def main() -> None:
    pose_model = load_pose_model()
    ov_model = load_open_vocab_model()
    out_path, metrics = process_video(pose_model, ov_model)

    print(f"Wrote H.264 {out_path}")
    print(
        f"YOLO pose   — calls={metrics['pose']['calls']}  "
        f"avg_latency={metrics['pose']['avg_latency_ms']:.2f} ms  "
        f"avg_throughput={metrics['pose']['avg_throughput_fps']:.2f} FPS"
    )
    print(
        f"YOLO-World  — calls={metrics['open_vocab']['calls']}  "
        f"avg_latency={metrics['open_vocab']['avg_latency_ms']:.2f} ms  "
        f"avg_throughput={metrics['open_vocab']['avg_throughput_fps']:.2f} FPS"
    )


if __name__ == "__main__":
    main()
