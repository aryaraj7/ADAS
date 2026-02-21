# =============================================================
# ADAS - Advanced Driver Assistance System
# config.py — Central configuration for all modules
# Phase 1: Human / Vehicle / Animal Detection
# =============================================================

# ─────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────

# YOLOv8 model variant:
#   yolov8n.pt → fastest  (recommended for laptop real-time)
#   yolov8s.pt → balanced speed / accuracy
#   yolov8m.pt → more accurate, slower
# Model is auto-downloaded by ultralytics on first run.
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = f"models/{MODEL_NAME}"   # cached locally after first download

# Minimum confidence to show a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.45

# IoU threshold for Non-Maximum Suppression
NMS_IOU_THRESHOLD = 0.45

# Run on CPU (safe default for laptop)
# Change to "cuda" if you have an NVIDIA GPU with CUDA installed
DEVICE = "cpu"


# ─────────────────────────────────────────
# CAMERA / VIDEO SOURCE
# ─────────────────────────────────────────

# 0 = default webcam
# 1, 2 ... = external cameras
# "path/to/video.mp4" = video file
CAMERA_SOURCE = 0

# Processing resolution (smaller = faster)
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720


# ─────────────────────────────────────────
# TARGET CLASSES  (COCO dataset IDs)
# Only these classes will be detected and displayed.
# ─────────────────────────────────────────

# Format:
#   coco_id: {
#       "name"    : display label
#       "category": "human" | "vehicle" | "animal"
#       "color"   : BGR tuple for bounding box
#   }

DETECTION_CLASSES = {
    # ── Humans ──────────────────────────────
    0:  {"name": "person",    "category": "human",   "color": (0,   220,   0)},

    # ── Vehicles ────────────────────────────
    1:  {"name": "bicycle",   "category": "vehicle", "color": (0,  165, 255)},
    2:  {"name": "car",       "category": "vehicle", "color": (0,   0,  255)},
    3:  {"name": "motorcycle","category": "vehicle", "color": (0,  165, 255)},
    5:  {"name": "bus",       "category": "vehicle", "color": (0,  60,  255)},
    7:  {"name": "truck",     "category": "vehicle", "color": (0,  60,  255)},

    # ── Animals ─────────────────────────────
    14: {"name": "bird",      "category": "animal",  "color": (255,  0, 200)},
    15: {"name": "cat",       "category": "animal",  "color": (255,  0, 200)},
    16: {"name": "dog",       "category": "animal",  "color": (255,  0, 200)},
    17: {"name": "horse",     "category": "animal",  "color": (255,  0, 200)},
    18: {"name": "sheep",     "category": "animal",  "color": (255,  0, 200)},
    19: {"name": "cow",       "category": "animal",  "color": (255,  0, 200)},
    20: {"name": "elephant",  "category": "animal",  "color": (200,  0, 200)},
    21: {"name": "bear",      "category": "animal",  "color": (200,  0, 200)},
    22: {"name": "zebra",     "category": "animal",  "color": (255,  0, 200)},
    23: {"name": "giraffe",   "category": "animal",  "color": (255,  0, 200)},
}

# Quick lookup: set of COCO IDs we care about (used for fast filtering)
TARGET_CLASS_IDS = set(DETECTION_CLASSES.keys())


# ─────────────────────────────────────────
# DISTANCE ESTIMATION  (bounding-box approximation)
# ─────────────────────────────────────────

# Approximate real-world heights (metres) for each category.
# Used with the pinhole camera model:
#   distance = (real_height × focal_length) / pixel_height
REAL_WORLD_HEIGHTS = {
    "human":   1.75,   # average person
    "vehicle": 1.50,   # average car roof height
    "animal":  0.60,   # mid-size animal
}

# Camera focal length in pixels (calibrate for your webcam for accuracy).
# Default: typical 1080p laptop webcam ≈ 1000 px focal length.
FOCAL_LENGTH_PX = 1000

# Distance thresholds for colour-coded risk levels (metres)
DISTANCE_DANGER  = 5.0    # red   — immediate risk
DISTANCE_WARNING = 15.0   # yellow — caution zone
# Beyond DISTANCE_WARNING → green (safe)


# ─────────────────────────────────────────
# ALERT SETTINGS
# ─────────────────────────────────────────

# Print alerts to console
CONSOLE_ALERTS = True

# Minimum frames between repeated alerts for the same object
ALERT_COOLDOWN_FRAMES = 30

# Category priority for alerts (highest first)
ALERT_PRIORITY = ["human", "animal", "vehicle"]


# ─────────────────────────────────────────
# DISPLAY SETTINGS
# ─────────────────────────────────────────

SHOW_FPS        = True
SHOW_DISTANCE   = True
SHOW_CONFIDENCE = True
SHOW_CATEGORY   = True

# Bounding box line thickness
BOX_THICKNESS = 2

# Font scale for labels
FONT_SCALE = 0.55

# Show a legend panel in the top-right corner
SHOW_LEGEND = True


# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────

ENABLE_LOGGING  = True
LOG_FILE        = "logs/detections.csv"
