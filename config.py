"""
============================================================
  Configuration — AI Blind Navigation System
============================================================
  Edit this file to match YOUR camera and setup.
  All settings are in one place.
============================================================
"""

# ============================================================
# CAMERA
# ============================================================
# Source options:
#   0               → laptop webcam
#   1               → external USB camera
#   "video.mp4"     → video file
#   "rtsp://..."    → IP camera (RTSP)
#   "http://ip:8080/video" → IP Webcam Android app
CAMERA_SOURCE = 0
CAMERA_FOV_HORIZONTAL = 60     # degrees (most phones 58-65)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================
# YOLO
# ============================================================
YOLO_WEIGHTS = "best.pt"
YOLO_CONFIDENCE = 0.45
YOLO_IOU = 0.50

# ============================================================
# DISTANCE CALIBRATION
# ============================================================
# How to calibrate:
#   1. Fix camera at known height (chest mount ~1.2m)
#   2. Place object at 1m, 2m, 3m, 5m, 7m
#   3. Run YOLO on each, record bbox_bottom_y / frame_height
#   4. Replace values below
CALIBRATION_Y = [0.95, 0.80, 0.65, 0.45, 0.35]
CALIBRATION_D = [1.0,  2.0,  3.0,  5.0,  7.0]

# ============================================================
# STRIDE
# ============================================================
STRIDE_M = 0.50  # visually impaired adult average

# ============================================================
# ALERT THRESHOLDS (meters)
# ============================================================
DIST_CRITICAL = 2.0   # DANGER
DIST_WARNING = 5.0    # WARNING
# > DIST_WARNING = INFO

# ============================================================
# COOLDOWN (seconds)
# ============================================================
COOLDOWN_CRITICAL_SEC = 1.0
COOLDOWN_WARNING_SEC = 2.0
COOLDOWN_INFO_SEC = 3.0
COOLDOWN_PIXEL_THRESH = 50  # same pothole if within N pixels

# ============================================================
# VLM — Moondream2 (int8 quantized, ~2GB VRAM)
# ============================================================
ENABLE_VLM = True
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"
VLM_INTERVAL_SEC = 4.0    # scene description every N seconds
VLM_QUIET_GAP_SEC = 2.0   # need N sec quiet before scene speaks

# Refined prompt — focused on road obstacles and navigation
SCENE_PROMPT = (
    "You are a navigation assistant for a blind person walking on an Indian road. "
    "Look at this image and describe ONLY things that matter for safe walking. "
    "Mention: vehicles (cars, bikes, autos, trucks) and if they are moving or parked, "
    "animals (dogs, cows, goats) and their position, "
    "people or crowds blocking the path, "
    "road surface problems (water, mud, gravel, broken edges, construction), "
    "stairs, curbs, speed bumps, open drains or manholes, "
    "narrow passages or dead ends. "
    "State positions as left, right, or ahead. "
    "Use ONE short sentence. Be direct. No filler words."
)

# ============================================================
# AUDIO
# ============================================================
AUDIO_LANG = "en"   # "en" or "hi"
AUDIO_RATE = 175    # words per minute

# ============================================================
# OUTPUT
# ============================================================
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = "output_annotated.mp4"
SHOW_LIVE_WINDOW = True
WINDOW_NAME = "Blind Navigation System"
