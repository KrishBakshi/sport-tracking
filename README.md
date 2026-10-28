# sport-tracking

MMA clip pipeline: **Qwen VLM** (local OpenAI-compatible) fighter boxes → crop → **YOLO pose** keypoints → H.264 video.

## Setup

```bash
uv sync
```

Run a local vision Qwen server (e.g. llama.cpp) at `http://127.0.0.1:9931/v1`.

## Run

```bash
uv run main.py
```

Writes `output/annotated.mp4`.

## Models

- VLM: Qwen via Ultralytics `LLM` (`chat.completions`, OpenAI multimodal messages)
- Pose: `yolo26m-pose.pt` (one person per VLM crop, highest confidence)
