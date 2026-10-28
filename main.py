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
# "fighter" alone is unreliable on YOLO-World; detect people then filter.
OV_CLASSES = ["person"]
MAX_FIGHTERS = 2
DET_CONF_MIN = 0.25
KPT_CONF_MIN = 0.3
# Drop tiny/noise people (crowd, partial limbs); keep largest = fighters, not ref.
MIN_DET_AREA_FRAC = 0.02
# Pose person must cover enough of the fighter crop (rejects ref leaking into crop).
MIN_POSE_IOU = 0.25
MIN_POSE_FILL = 0.15

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


def _box_area(xyxy: np.ndarray | tuple[float, ...]) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _clip_box(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int] | None:
    px1 = int(np.clip(x1, 0, w - 1))
    py1 = int(np.clip(y1, 0, h - 1))
    px2 = int(np.clip(x2, 0, w - 1))
    py2 = int(np.clip(y2, 0, h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def select_fighter_boxes(
    xyxy: np.ndarray,
    conf: np.ndarray,
    frame_shape: tuple[int, ...],
    track_ids: np.ndarray | None = None,
    preferred_ids: list[int] | None = None,
    max_fighters: int = MAX_FIGHTERS,
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    """Select up to two fighters; keep a KO'd fighter on the floor, drop the ref.

    With BoT-SORT ids: stick to preferred_ids when those tracks are still alive.
    Otherwise:
      1) prefer wide/prone boxes (floor)
      2) fill with largest standing people (fighter before smaller standing ref)
    """
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    prone_aspect = 1.15  # width/height; lying down tends to be wider than tall

    candidates: list[dict] = []
    for i, (box, score) in enumerate(zip(xyxy, conf)):
        if float(score) < DET_CONF_MIN:
            continue
        clipped = _clip_box(*box, w, h)
        if clipped is None:
            continue
        bw = float(clipped[2] - clipped[0])
        bh = float(clipped[3] - clipped[1])
        area = bw * bh
        aspect = bw / max(bh, 1.0)
        min_area = frame_area * MIN_DET_AREA_FRAC
        if aspect >= prone_aspect:
            min_area *= 0.5
        if area < min_area:
            continue
        tid = int(track_ids[i]) if track_ids is not None else -1
        candidates.append(
            {
                "box": clipped,
                "area": area,
                "conf": float(score),
                "aspect": aspect,
                "rank": float(score) * (area**0.5),
                "track_id": tid,
            }
        )

    candidates.sort(key=lambda c: c["rank"], reverse=True)
    kept: list[dict] = []
    for cand in candidates:
        if any(
            _iou_xyxy(np.array(cand["box"], dtype=float), np.array(prev["box"], dtype=float))
            > 0.6
            for prev in kept
        ):
            continue
        kept.append(cand)

    selected: list[dict] = []
    preferred = preferred_ids or []

    # Prefer continuing the same BoT-SORT track ids across frames.
    if preferred:
        by_id = {c["track_id"]: c for c in kept if c["track_id"] >= 0}
        for tid in preferred:
            if tid in by_id and len(selected) < max_fighters:
                selected.append(by_id[tid])

    if len(selected) < max_fighters:
        selected_ids = {c["track_id"] for c in selected}
        prone = sorted(
            (c for c in kept if c["aspect"] >= prone_aspect and c["track_id"] not in selected_ids),
            key=lambda c: c["rank"],
            reverse=True,
        )
        standing = sorted(
            (c for c in kept if c["aspect"] < prone_aspect and c["track_id"] not in selected_ids),
            key=lambda c: c["area"],
            reverse=True,
        )
        for group in (prone, standing):
            for cand in group:
                selected.append(cand)
                if len(selected) >= max_fighters:
                    break
            if len(selected) >= max_fighters:
                break

    boxes = [c["box"] for c in selected]
    ids = [c["track_id"] for c in selected]
    return boxes, ids


def detect_fighters(
    ov_model: YOLOWorld,
    frame,
    latencies_ms: list[float],
    max_fighters: int = MAX_FIGHTERS,
) -> list[tuple[int, int, int, int]]:
    """Open-vocab person detect; return up to max_fighters boxes (no referee)."""
    t0 = time.perf_counter()
    results = ov_model.predict(frame, conf=DET_CONF_MIN, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(elapsed_ms)

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes, _ids = select_fighter_boxes(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        frame.shape,
        max_fighters=max_fighters,
    )
    return boxes


def pose_on_crop(
    pose_model: YOLO,
    crop,
) -> tuple[np.ndarray | None, float]:
    """Pose on crop; keep the one person that best fills this fighter bbox."""
    results = pose_model.predict(crop, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, 0.0

    ch, cw = crop.shape[:2]
    crop_box = np.array([0.0, 0.0, float(cw), float(ch)], dtype=float)
    crop_area = float(cw * ch)
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    kpts_all = result.keypoints.xy.cpu().numpy()

    best_i = None
    best_score = -1.0
    for i, (box, score) in enumerate(zip(xyxy, conf)):
        if float(score) < KPT_CONF_MIN:
            continue
        iou = _iou_xyxy(box, crop_box)
        fill = _box_area(box) / crop_area if crop_area > 0 else 0.0
        if iou < MIN_POSE_IOU or fill < MIN_POSE_FILL:
            continue
        # Prefer the person who dominates this crop (not a partial referee).
        rank = iou * 0.6 + fill * 0.3 + float(score) * 0.1
        if rank > best_score:
            best_score = rank
            best_i = i

    if best_i is None:
        return None, 0.0

    kpts = kpts_all[best_i].copy()
    # Guardrail: keypoints must stay inside this fighter crop.
    for j in range(len(kpts)):
        x, y = kpts[j]
        if x <= 0 and y <= 0:
            continue
        if not (0 <= x < cw and 0 <= y < ch):
            kpts[j] = 0.0
        else:
            kpts[j, 0] = float(np.clip(x, 0, cw - 1))
            kpts[j, 1] = float(np.clip(y, 0, ch - 1))

    return kpts, float(conf[best_i])


def draw_keypoints(
    frame,
    kpts: np.ndarray,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    clip_bbox: tuple[int, int, int, int] | None = None,
):
    """Draw keypoints + skeleton; optionally clip drawing to the fighter bbox."""
    ox, oy = origin
    x1 = y1 = 0
    x2 = frame.shape[1] - 1
    y2 = frame.shape[0] - 1
    if clip_bbox is not None:
        x1, y1, x2, y2 = clip_bbox

    def inside(pt: tuple[int, int]) -> bool:
        return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

    points: list[tuple[int, int] | None] = []
    for x, y in kpts:
        if x <= 0 and y <= 0:
            points.append(None)
            continue
        pt = (int(x + ox), int(y + oy))
        if not inside(pt):
            points.append(None)
            continue
        points.append(pt)
        cv2.circle(frame, pt, 3, color, -1)

    for a, b in POSE_SKELETON:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, pa, pb, color, 2)


def draw_det_boxes(
    frame,
    bboxes: list[tuple[int, int, int, int]],
    track_ids: list[int] | None = None,
):
    out = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        tid = track_ids[i] if track_ids and i < len(track_ids) else -1
        label = f"fighter {i + 1}" if tid < 0 else f"fighter {i + 1} id={tid}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
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
    track_ids: list[int] | None = None,
):
    """Crop each det bbox -> YOLO pose -> keep highest-conf person; draw keypoints only."""
    if not det_bboxes:
        return frame

    out = draw_det_boxes(frame, det_bboxes, track_ids=track_ids)
    colors = ((0, 255, 0), (0, 165, 255))

    t0 = time.perf_counter()
    for i, (x1, y1, x2, y2) in enumerate(det_bboxes):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        kpts, conf = pose_on_crop(pose_model, crop)
        if kpts is None or conf < KPT_CONF_MIN:
            continue

        draw_keypoints(
            out,
            kpts,
            (x1, y1),
            colors[i % len(colors)],
            clip_bbox=(x1, y1, x2, y2),
        )

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
