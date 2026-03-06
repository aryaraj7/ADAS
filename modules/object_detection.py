# =============================================================
# ADAS — Phase 1
# modules/object_detection.py
#
# Responsibilities:
#   • Load YOLOv8 detection + pose models
#   • Run inference on each frame
#   • Detect ALL 80 COCO classes + body-part keypoints
#   • Estimate distance using bounding-box height or MiDaS
#   • Draw annotated bounding boxes + skeleton on the frame
#   • Support custom-trained models
# =============================================================

import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

import config
from utils.distance_estimator import estimate_distance
from modules.depth_estimator import DepthEstimator

# YOLOv8-pose keypoint names (17 COCO keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections for drawing limbs
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

KEYPOINT_COLOR = (0, 255, 255)   # cyan dots
SKELETON_COLOR = (255, 255, 0)   # yellow limbs
HAND_COLOR     = (255, 255, 0)   # cyan for hand bounding box


class ObjectDetector:
    """
    Wraps YOLOv8 inference for ADAS detection.
    Supports: all 80 COCO classes, pose keypoints, and custom models.
    """

    def __init__(self):
        self.model = self._load_model()
        self.pose_model = self._load_pose_model()
        self.hand_detector = self._load_hand_detector()
        self.class_info = config.DETECTION_CLASSES
        self.target_ids = config.TARGET_CLASS_IDS
        self.detect_all = getattr(config, "DETECT_ALL_CLASSES", False)

        # MiDaS depth estimator (loaded only if enabled in config)
        self.depth_estimator = DepthEstimator() if config.USE_MIDAS else None

    # ─────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────

    def _load_model(self) -> YOLO:
        # Use custom model if configured
        custom = getattr(config, "CUSTOM_MODEL_PATH", None)
        if custom and os.path.exists(custom):
            print(f"[ObjectDetector] Loading CUSTOM model from {custom}")
            model = YOLO(custom)
            model.to(config.DEVICE)
            return model

        local_path = config.MODEL_PATH
        if os.path.exists(local_path):
            print(f"[ObjectDetector] Loading model from {local_path}")
            model = YOLO(local_path)
        else:
            print(f"[ObjectDetector] Downloading {config.MODEL_NAME} …")
            model = YOLO(config.MODEL_NAME)
            os.makedirs("models", exist_ok=True)
            model.save(local_path)
            print(f"[ObjectDetector] Model saved to {local_path}")

        model.to(config.DEVICE)
        print(f"[ObjectDetector] Running on device: {config.DEVICE}")
        return model

    def _load_pose_model(self):
        if not getattr(config, "ENABLE_POSE", False):
            return None

        pose_path = getattr(config, "POSE_MODEL_PATH", "models/yolov8n-pose.pt")
        pose_name = getattr(config, "POSE_MODEL_NAME", "yolov8n-pose.pt")

        if os.path.exists(pose_path):
            print(f"[ObjectDetector] Loading pose model from {pose_path}")
            model = YOLO(pose_path)
        else:
            print(f"[ObjectDetector] Downloading {pose_name} …")
            model = YOLO(pose_name)
            os.makedirs("models", exist_ok=True)
            model.save(pose_path)
            print(f"[ObjectDetector] Pose model saved to {pose_path}")

        model.to(config.DEVICE)
        print(f"[ObjectDetector] Pose model ready")
        return model

    def _load_hand_detector(self):
        if not getattr(config, "ENABLE_HANDS", True):
            return None
        print("[ObjectDetector] Loading MediaPipe Hands …")
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        print("[ObjectDetector] Hand detector ready")
        return hands

    # ─────────────────────────────────────────
    # Main detection entry point
    # ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        annotated = frame.copy()

        # Run MiDaS once per frame
        depth_map = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(frame)

        if results.boxes is not None and len(results.boxes) > 0:
            # Get YOLO's own class names for unmapped classes
            model_names = results.names  # {0: 'person', 1: 'bicycle', ...}

            for box in results.boxes:
                class_id = int(box.cls[0].item())

                # If not detecting all, filter to target IDs only
                if not self.detect_all and class_id not in self.target_ids:
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get class info from our config, or build a default
                if class_id in self.class_info:
                    info = self.class_info[class_id]
                    name = info["name"]
                    category = info["category"]
                    color = info["color"]
                else:
                    name = model_names.get(class_id, f"class_{class_id}")
                    category = "object"
                    color = (200, 200, 200)

                # Distance estimation
                if depth_map is not None:
                    distance_m = self.depth_estimator.get_distance(depth_map, [x1, y1, x2, y2])
                else:
                    pixel_height = y2 - y1
                    distance_m = estimate_distance(category, pixel_height)
                risk = self._risk_level(distance_m)

                # Real-world size estimation (width × height in metres)
                size_w, size_h = None, None
                if distance_m is not None and distance_m > 0:
                    focal = config.FOCAL_LENGTH_PX
                    px_w = x2 - x1
                    px_h = y2 - y1
                    size_w = round((px_w * distance_m) / focal, 2)
                    size_h = round((px_h * distance_m) / focal, 2)

                detection = {
                    "class_id":   class_id,
                    "name":       name,
                    "category":   category,
                    "confidence": confidence,
                    "box":        [x1, y1, x2, y2],
                    "distance_m": distance_m,
                    "risk":       risk,
                    "size_w":     size_w,
                    "size_h":     size_h,
                }
                detections.append(detection)
                self._draw_detection(annotated, detection, color)

        # Run pose estimation for body keypoints
        if self.pose_model is not None:
            self._draw_pose(annotated, frame)

        # Run hand detection (MediaPipe)
        if self.hand_detector is not None:
            hand_dets = self._detect_hands(frame, depth_map)
            for hdet in hand_dets:
                detections.append(hdet)
                self._draw_detection(annotated, hdet, HAND_COLOR)

        return detections, annotated

    # ─────────────────────────────────────────
    # Pose / body-part keypoints
    # ─────────────────────────────────────────

    def _draw_pose(self, annotated: np.ndarray, frame: np.ndarray) -> None:
        pose_results = self.pose_model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        if pose_results.keypoints is None:
            return

        kpts_data = pose_results.keypoints.data  # shape: (N, 17, 3) — x, y, conf
        for person_kpts in kpts_data:
            points = []
            for i, kpt in enumerate(person_kpts):
                x, y, conf = int(kpt[0].item()), int(kpt[1].item()), float(kpt[2].item())
                points.append((x, y, conf))
                if conf > 0.5:
                    cv2.circle(annotated, (x, y), 4, KEYPOINT_COLOR, -1)
                    # Label key body parts
                    if i in (0, 5, 6, 9, 10, 15, 16):  # nose, shoulders, wrists, ankles
                        cv2.putText(annotated, KEYPOINT_NAMES[i],
                                    (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3, KEYPOINT_COLOR, 1, cv2.LINE_AA)

            # Draw skeleton limbs
            for (a, b) in SKELETON:
                if points[a][2] > 0.5 and points[b][2] > 0.5:
                    cv2.line(annotated,
                             (points[a][0], points[a][1]),
                             (points[b][0], points[b][1]),
                             SKELETON_COLOR, 2, cv2.LINE_AA)

    # ─────────────────────────────────────────
    # Hand detection (MediaPipe)
    # ─────────────────────────────────────────

    def _detect_hands(self, frame: np.ndarray, depth_map) -> list[dict]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb)

        hand_detections = []
        if not results.multi_hand_landmarks:
            return hand_detections

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Get bounding box from all 21 landmarks
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1 = max(int(min(xs)) - 10, 0)
            y1 = max(int(min(ys)) - 10, 0)
            x2 = min(int(max(xs)) + 10, w)
            y2 = min(int(max(ys)) + 10, h)

            hand_label = handedness.classification[0].label  # "Left" or "Right"
            confidence = handedness.classification[0].score

            # Distance estimation
            if depth_map is not None and self.depth_estimator is not None:
                distance_m = self.depth_estimator.get_distance(depth_map, [x1, y1, x2, y2])
            else:
                pixel_height = y2 - y1
                distance_m = estimate_distance("object", pixel_height)
            risk = self._risk_level(distance_m)

            # Size estimation
            size_w, size_h = None, None
            if distance_m is not None and distance_m > 0:
                focal = config.FOCAL_LENGTH_PX
                size_w = round(((x2 - x1) * distance_m) / focal, 2)
                size_h = round(((y2 - y1) * distance_m) / focal, 2)

            hand_detections.append({
                "class_id":   -1,
                "name":       f"Hand ({hand_label})",
                "category":   "human",
                "confidence": confidence,
                "box":        [x1, y1, x2, y2],
                "distance_m": distance_m,
                "risk":       risk,
                "size_w":     size_w,
                "size_h":     size_h,
            })

        return hand_detections

    # ─────────────────────────────────────────
    # Risk classification
    # ─────────────────────────────────────────

    @staticmethod
    def _risk_level(distance_m: float | None) -> str:
        if distance_m is None:
            return "unknown"
        if distance_m <= config.DISTANCE_DANGER:
            return "danger"
        if distance_m <= config.DISTANCE_WARNING:
            return "warning"
        return "safe"

    # ─────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────

    def _draw_detection(self, frame: np.ndarray, det: dict, base_color: tuple) -> None:
        x1, y1, x2, y2 = det["box"]

        risk_colors = {
            "danger":  (0,   0, 255),
            "warning": (0, 165, 255),
            "safe":    (0, 220,   0),
            "unknown": base_color,
        }
        draw_color = risk_colors.get(det["risk"], base_color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, config.BOX_THICKNESS)

        parts = [f"{det['name']}"]
        if config.SHOW_CONFIDENCE:
            parts.append(f"{det['confidence']:.0%}")
        if config.SHOW_DISTANCE and det["distance_m"] is not None:
            parts.append(f"depth:{det['distance_m']:.1f}m")
        if det.get("size_w") is not None and det.get("size_h") is not None:
            parts.append(f"size:{det['size_w']:.1f}x{det['size_h']:.1f}m")
        if config.SHOW_CATEGORY:
            parts.append(f"[{det['category']}]")
        label = "  ".join(parts)

        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = config.FONT_SCALE
        thickness = 1
        (lw, lh), baseline = cv2.getTextSize(label, font, scale, thickness)
        pad = 4
        label_y1 = max(y1 - lh - baseline - pad * 2, 0)
        label_y2 = label_y1 + lh + baseline + pad * 2

        cv2.rectangle(frame, (x1, label_y1), (x1 + lw + pad * 2, label_y2),
                      draw_color, -1)
        cv2.putText(frame, label,
                    (x1 + pad, label_y2 - baseline - pad // 2),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
