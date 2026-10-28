# sport-tracking

MMA clip pipeline: **YOLO-World** open-vocabulary fighter boxes → crop → **YOLO pose** keypoints → H.264 video.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py
```

Writes `output/annotated.mp4`.

## Models

- Open-vocab: `yolov8m-worldv2.pt` with classes `person`, `fighter`
- Pose: `yolo26m-pose.pt` (one person per crop, highest confidence)
