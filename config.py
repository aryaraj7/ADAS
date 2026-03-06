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

# Pose model for body-part keypoints (runs alongside detection model)
POSE_MODEL_NAME = "yolov8n-pose.pt"
POSE_MODEL_PATH = f"models/{POSE_MODEL_NAME}"
ENABLE_POSE = True  # draw body keypoints (eyes, nose, shoulders, etc.)
ENABLE_HANDS = True  # detect hands via MediaPipe

# Custom model path — set to a .pt file to use your own trained model
# Example: CUSTOM_MODEL_PATH = "models/custom_trained.pt"
CUSTOM_MODEL_PATH = None

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
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480


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

DETECT_ALL_CLASSES = True  # True = detect all 80 COCO classes

# Target FPS — detection loop will pace itself to this rate
TARGET_FPS = 10

DETECTION_CLASSES = {
    # ── Humans ──────────────────────────────
    0:  {"name": "person",      "category": "human",   "color": (0, 220, 0)},
    # ── Vehicles ────────────────────────────
    1:  {"name": "bicycle",     "category": "vehicle", "color": (0, 165, 255)},
    2:  {"name": "car",         "category": "vehicle", "color": (0, 0, 255)},
    3:  {"name": "motorcycle",  "category": "vehicle", "color": (0, 165, 255)},
    4:  {"name": "airplane",    "category": "vehicle", "color": (0, 100, 255)},
    5:  {"name": "bus",         "category": "vehicle", "color": (0, 60, 255)},
    6:  {"name": "train",       "category": "vehicle", "color": (0, 60, 255)},
    7:  {"name": "truck",       "category": "vehicle", "color": (0, 60, 255)},
    8:  {"name": "boat",        "category": "vehicle", "color": (0, 130, 255)},
    # ── Outdoor ─────────────────────────────
    9:  {"name": "traffic light",  "category": "object", "color": (0, 255, 255)},
    10: {"name": "fire hydrant",   "category": "object", "color": (0, 255, 200)},
    11: {"name": "stop sign",      "category": "object", "color": (0, 200, 255)},
    12: {"name": "parking meter",  "category": "object", "color": (0, 200, 200)},
    13: {"name": "bench",          "category": "object", "color": (150, 200, 150)},
    # ── Animals ─────────────────────────────
    14: {"name": "bird",        "category": "animal", "color": (255, 0, 200)},
    15: {"name": "cat",         "category": "animal", "color": (255, 0, 200)},
    16: {"name": "dog",         "category": "animal", "color": (255, 0, 200)},
    17: {"name": "horse",       "category": "animal", "color": (255, 0, 200)},
    18: {"name": "sheep",       "category": "animal", "color": (255, 0, 200)},
    19: {"name": "cow",         "category": "animal", "color": (255, 0, 200)},
    20: {"name": "elephant",    "category": "animal", "color": (200, 0, 200)},
    21: {"name": "bear",        "category": "animal", "color": (200, 0, 200)},
    22: {"name": "zebra",       "category": "animal", "color": (255, 0, 200)},
    23: {"name": "giraffe",     "category": "animal", "color": (255, 0, 200)},
    # ── Accessories ─────────────────────────
    24: {"name": "backpack",    "category": "object", "color": (180, 180, 50)},
    25: {"name": "umbrella",    "category": "object", "color": (180, 180, 50)},
    26: {"name": "handbag",     "category": "object", "color": (180, 180, 50)},
    27: {"name": "tie",         "category": "object", "color": (180, 180, 50)},
    28: {"name": "suitcase",    "category": "object", "color": (180, 180, 50)},
    # ── Sports ──────────────────────────────
    29: {"name": "frisbee",        "category": "object", "color": (100, 255, 100)},
    30: {"name": "skis",           "category": "object", "color": (100, 255, 100)},
    31: {"name": "snowboard",      "category": "object", "color": (100, 255, 100)},
    32: {"name": "sports ball",    "category": "object", "color": (100, 255, 100)},
    33: {"name": "kite",           "category": "object", "color": (100, 255, 100)},
    34: {"name": "baseball bat",   "category": "object", "color": (100, 255, 100)},
    35: {"name": "baseball glove", "category": "object", "color": (100, 255, 100)},
    36: {"name": "skateboard",     "category": "object", "color": (100, 255, 100)},
    37: {"name": "surfboard",      "category": "object", "color": (100, 255, 100)},
    38: {"name": "tennis racket",  "category": "object", "color": (100, 255, 100)},
    # ── Kitchen ─────────────────────────────
    39: {"name": "bottle",     "category": "object", "color": (255, 200, 100)},
    40: {"name": "wine glass", "category": "object", "color": (255, 200, 100)},
    41: {"name": "cup",        "category": "object", "color": (255, 200, 100)},
    42: {"name": "fork",       "category": "object", "color": (255, 200, 100)},
    43: {"name": "knife",      "category": "object", "color": (255, 200, 100)},
    44: {"name": "spoon",      "category": "object", "color": (255, 200, 100)},
    45: {"name": "bowl",       "category": "object", "color": (255, 200, 100)},
    # ── Food ────────────────────────────────
    46: {"name": "banana",     "category": "object", "color": (100, 255, 255)},
    47: {"name": "apple",      "category": "object", "color": (100, 255, 255)},
    48: {"name": "sandwich",   "category": "object", "color": (100, 255, 255)},
    49: {"name": "orange",     "category": "object", "color": (100, 255, 255)},
    50: {"name": "broccoli",   "category": "object", "color": (100, 255, 255)},
    51: {"name": "carrot",     "category": "object", "color": (100, 255, 255)},
    52: {"name": "hot dog",    "category": "object", "color": (100, 255, 255)},
    53: {"name": "pizza",      "category": "object", "color": (100, 255, 255)},
    54: {"name": "donut",      "category": "object", "color": (100, 255, 255)},
    55: {"name": "cake",       "category": "object", "color": (100, 255, 255)},
    # ── Furniture ───────────────────────────
    56: {"name": "chair",        "category": "object", "color": (200, 150, 100)},
    57: {"name": "couch",        "category": "object", "color": (200, 150, 100)},
    58: {"name": "potted plant", "category": "object", "color": (200, 150, 100)},
    59: {"name": "bed",          "category": "object", "color": (200, 150, 100)},
    60: {"name": "dining table", "category": "object", "color": (200, 150, 100)},
    61: {"name": "toilet",       "category": "object", "color": (200, 150, 100)},
    # ── Electronics ─────────────────────────
    62: {"name": "tv",         "category": "object", "color": (255, 100, 100)},
    63: {"name": "laptop",     "category": "object", "color": (255, 100, 100)},
    64: {"name": "mouse",      "category": "object", "color": (255, 100, 100)},
    65: {"name": "remote",     "category": "object", "color": (255, 100, 100)},
    66: {"name": "keyboard",   "category": "object", "color": (255, 100, 100)},
    67: {"name": "cell phone", "category": "object", "color": (255, 100, 100)},
    # ── Appliances ──────────────────────────
    68: {"name": "microwave",    "category": "object", "color": (200, 100, 255)},
    69: {"name": "oven",         "category": "object", "color": (200, 100, 255)},
    70: {"name": "toaster",      "category": "object", "color": (200, 100, 255)},
    71: {"name": "sink",         "category": "object", "color": (200, 100, 255)},
    72: {"name": "refrigerator", "category": "object", "color": (200, 100, 255)},
    # ── Other ───────────────────────────────
    73: {"name": "book",       "category": "object", "color": (150, 150, 255)},
    74: {"name": "clock",      "category": "object", "color": (150, 150, 255)},
    75: {"name": "vase",       "category": "object", "color": (150, 150, 255)},
    76: {"name": "scissors",   "category": "object", "color": (150, 150, 255)},
    77: {"name": "teddy bear", "category": "object", "color": (150, 150, 255)},
    78: {"name": "hair drier", "category": "object", "color": (150, 150, 255)},
    79: {"name": "toothbrush", "category": "object", "color": (150, 150, 255)},
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
    "object":  0.50,   # generic object fallback
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
SHOW_DISTANCE   = False
SHOW_CONFIDENCE = True
SHOW_CATEGORY   = False

# Bounding box line thickness
BOX_THICKNESS = 2

# Font scale for labels
FONT_SCALE = 0.55

# Show a legend panel in the top-right corner
SHOW_LEGEND = True


# ─────────────────────────────────────────
# MIDAS DEPTH ESTIMATION
# ─────────────────────────────────────────

# Set True to use MiDaS AI depth instead of bounding-box pinhole model
USE_MIDAS = False

# MiDaS model variant:
#   "MiDaS_small"  → fastest, good for CPU (recommended)
#   "DPT_Hybrid"   → more accurate, ~4x slower (needs GPU for real-time)
#   "DPT_Large"    → most accurate, very slow on CPU
# Note: v3.1 models (LeViT, SwinV2, BEiT) need GPU — too slow on CPU
MIDAS_MODEL_TYPE = "MiDaS_small"

# Run MiDaS every N frames to save CPU (reuse last depth map in between)
MIDAS_EVERY_N_FRAMES = 3

# Depth range mapping (metres)
# MiDaS gives relative depth (0=far, 1=close), mapped to this range
MIDAS_MIN_RANGE = 0.5    # closest distance (metres)
MIDAS_MAX_RANGE = 20.0   # farthest distance (metres)


# ─────────────────────────────────────────
# ULTRASONIC SENSOR (ESP32 + HC-SR04)
# ─────────────────────────────────────────

# Enable ultrasonic sensor reading from ESP32 serial
ENABLE_ULTRASONIC = True

# Serial port — None = auto-detect ESP32 USB, or set manually e.g. "/dev/cu.usbserial-0001"
ULTRASONIC_PORT = None

# Baud rate (must match ESP32 firmware)
ULTRASONIC_BAUDRATE = 115200


# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────

ENABLE_LOGGING  = True
LOG_FILE        = "logs/detections.csv"
