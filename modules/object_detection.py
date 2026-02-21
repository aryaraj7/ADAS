# =============================================================
# ADAS — Phase 1
# modules/object_detection.py
#
# Responsibilities:
#   • Load YOLOv8 pre-trained model (auto-download on first run)
#   • Run inference on each frame
#   • Filter results to ONLY our target classes
#     (humans, vehicles, animals)
#   • Estimate distance using bounding-box height
#   • Draw annotated bounding boxes on the frame
#   • Return structured detection list to main pipeline
# =============================================================

import os
import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils.distance_estimator import estimate_distance


class ObjectDetector:
    """
    Wraps YOLOv8 inference and post-processing for Phase 1 ADAS detection.

    Usage:
        detector = ObjectDetector()
        detections, annotated_frame = detector.detect(frame)
    """

    def __init__(self):
        self.model = self._load_model()
        self.class_info = config.DETECTION_CLASSES
        self.target_ids = config.TARGET_CLASS_IDS

    # ─────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────

    def _load_model(self) -> YOLO:
        """
        Load YOLOv8 weights.
        • If the model file exists in models/, load from there.
        • Otherwise ultralytics auto-downloads it from the official
          Ultralytics GitHub release and caches it locally.
        """
        local_path = config.MODEL_PATH
        if os.path.exists(local_path):
            print(f"[ObjectDetector] Loading model from {local_path}")
            model = YOLO(local_path)
        else:
            print(f"[ObjectDetector] Downloading {config.MODEL_NAME} …")
            model = YOLO(config.MODEL_NAME)
            # Save to our models/ folder so future runs are instant
            os.makedirs("models", exist_ok=True)
            model.save(local_path)
            print(f"[ObjectDetector] Model saved to {local_path}")

        model.to(config.DEVICE)
        print(f"[ObjectDetector] Running on device: {config.DEVICE}")
        return model

    # ─────────────────────────────────────────
    # Main detection entry point
    # ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """
        Run detection on a single BGR frame.

        Returns
        -------
        detections : list of dicts  — one entry per valid detection
            {
              "class_id"   : int,
              "name"       : str,
              "category"   : str,          # human / vehicle / animal
              "confidence" : float,
              "box"        : [x1,y1,x2,y2],
              "distance_m" : float | None, # estimated metres
              "risk"       : str,          # "danger" | "warning" | "safe"
            }
        annotated_frame : np.ndarray — frame with drawn boxes
        """
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        annotated = frame.copy()

        if results.boxes is None or len(results.boxes) == 0:
            return detections, annotated

        for box in results.boxes:
            class_id = int(box.cls[0].item())

            # Skip classes we don't care about
            if class_id not in self.target_ids:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            info     = self.class_info[class_id]
            name     = info["name"]
            category = info["category"]
            color    = info["color"]

            # Distance estimation
            pixel_height = y2 - y1
            distance_m   = estimate_distance(category, pixel_height)
            risk         = self._risk_level(distance_m)

            detection = {
                "class_id":   class_id,
                "name":       name,
                "category":   category,
                "confidence": confidence,
                "box":        [x1, y1, x2, y2],
                "distance_m": distance_m,
                "risk":       risk,
            }
            detections.append(detection)

            # Draw on frame
            self._draw_detection(annotated, detection, color)

        return detections, annotated

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

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: dict,
        base_color: tuple,
    ) -> None:
        """Draw bounding box + label for one detection."""
        x1, y1, x2, y2 = det["box"]

        # Override colour based on risk level
        risk_colors = {
            "danger":  (0,   0, 255),   # red
            "warning": (0, 165, 255),   # orange
            "safe":    (0, 220,   0),   # green
            "unknown": base_color,
        }
        draw_color = risk_colors.get(det["risk"], base_color)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, config.BOX_THICKNESS)

        # Build label string
        parts = [f"{det['name']}"]
        if config.SHOW_CONFIDENCE:
            parts.append(f"{det['confidence']:.0%}")
        if config.SHOW_DISTANCE and det["distance_m"] is not None:
            parts.append(f"{det['distance_m']:.1f}m")
        if config.SHOW_CATEGORY:
            parts.append(f"[{det['category']}]")
        label = "  ".join(parts)

        # Label background pill
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
