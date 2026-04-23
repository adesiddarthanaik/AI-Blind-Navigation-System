# AI-Powered Blind Navigation System

Real-time pothole detection and navigation assistance for visually impaired users on Indian roads.

## Architecture

```
Camera Feed (Webcam / IP Cam / Video / Phone)
    │
    ├──► YOLOv5s (~0.5 GB VRAM, every frame, ~25 fps)
    │       │
    │       ├──► Distance  (calibrated polynomial from bbox bottom-y)
    │       ├──► Angle     (pinhole geometry from bbox center-x)
    │       ├──► Steps     (distance ÷ stride)
    │       └──► Path      (safest corridor from obstacle distribution)
    │               │
    │               ▼
    │       ┌───────────────┐
    │       │ Alert Manager  │  ← Priority coordination
    │       │  Cooldown      │     CRITICAL > WARNING > INFO > SCENE
    │       │  Merging       │     Scene fills gaps only
    │       │  Timing        │     Path speaks on change only
    │       └───────┬───────┘
    │               │
    ├──► Moondream2 VLM (~2 GB VRAM, int8, every 4 sec)
    │               │
    │               ▼
    │       ┌───────────────┐
    │       │ Audio Engine   │  ← Non-blocking, threaded
    │       │ (pyttsx3)      │     Priority queue
    │       └───────────────┘     CRITICAL jumps ahead
    │               │
    │               ▼
    │           🔊 Spoken alerts
    │
    └──► Annotated video output
```

## VRAM Budget (4 GB GPU)

| Module | VRAM | Status |
|--------|------|--------|
| YOLOv5s | ~0.5 GB | ✅ |
| Moondream2 (int8) | ~2.0 GB | ✅ |
| Total | ~2.5 GB | ✅ fits 4GB GPU |

## Quick Start

```bash
# 1. Place best.pt in this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py                    # webcam
python main.py --source video.mp4 # video file
python main.py --no-vlm           # without scene description (faster)
python main.py --lang hi          # Hindi audio
```

## All Options

| Flag | Description |
|------|-------------|
| `--source 0` | Laptop webcam (default) |
| `--source 1` | External USB camera |
| `--source video.mp4` | Video file |
| `--source http://ip:8080/video` | IP Webcam Android app |
| `--lang en` | English audio (default) |
| `--lang hi` | Hindi audio |
| `--no-vlm` | Disable Moondream2 |
| `--no-display` | No video window |
| `--save output.mp4` | Output video path |
| `--save none` | Don't save video |

## Calibration

Edit `config.py` → `CALIBRATION_Y` and `CALIBRATION_D`:

1. Fix camera at chest height (~1.2m)
2. Place object at 1m, 2m, 3m, 5m, 7m
3. Run YOLO, record bbox bottom-y / frame_height for each
4. Update the two lists in config.py

## Files

```
blind_nav/
├── main.py            # Entry point
├── config.py          # All settings
├── detector.py        # YOLOv5 wrapper
├── estimator.py       # Distance + angle + steps + path
├── alert_manager.py   # Priority alert coordination
├── audio_engine.py    # Threaded TTS
├── scene_describer.py # Moondream2 int8
├── annotator.py       # Frame drawing
├── requirements.txt   # Dependencies
├── best.pt            # Your YOLO weights (add yourself)
└── README.md
```

## License

MIT
