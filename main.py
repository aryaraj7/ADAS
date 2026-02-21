# =============================================================
# ADAS — main.py
# Phase 1: Human / Vehicle / Animal Detection
#
# Run:
#   python main.py                    # live webcam
#   python main.py --source video.mp4 # video file
# =============================================================

import argparse
import time
import sys

import cv2

import config
from modules.object_detection import ObjectDetector
from utils.alert_system import AlertSystem


# ─────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────

def draw_fps(frame, fps: float):
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def draw_legend(frame):
    """Small legend in the top-right corner."""
    items = [
        ("Human",   (0, 220,   0)),
        ("Vehicle", (0,   0, 255)),
        ("Animal",  (255, 0, 200)),
        ("DANGER",  (0,   0, 255)),
        ("WARNING", (0, 165, 255)),
        ("SAFE",    (0, 220,   0)),
    ]
    h, w = frame.shape[:2]
    x_start = w - 145
    y_start = 12
    box_w, box_h, gap = 130, 20, 5

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x_start - 8, y_start - 5),
                  (w - 8, y_start + len(items) * (box_h + gap) + 5),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (label, color) in enumerate(items):
        y = y_start + i * (box_h + gap)
        cv2.rectangle(frame, (x_start, y), (x_start + 14, y + 14), color, -1)
        cv2.putText(frame, label, (x_start + 20, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                    cv2.LINE_AA)


def draw_summary(frame, detections: list[dict]):
    """
    Bottom-left summary: count per category + any DANGER alerts.
    """
    counts = {"human": 0, "vehicle": 0, "animal": 0}
    for d in detections:
        counts[d["category"]] = counts.get(d["category"], 0) + 1

    h = frame.shape[0]
    lines = [
        f"Persons : {counts['human']}",
        f"Vehicles: {counts['vehicle']}",
        f"Animals : {counts['animal']}",
    ]

    # Flash DANGER banner
    dangers = [d for d in detections if d["risk"] == "danger"]
    if dangers:
        names = ", ".join(d["name"] for d in dangers[:3])
        lines.append(f"!! DANGER: {names} !!")

    for i, line in enumerate(lines):
        y = h - 15 - (len(lines) - 1 - i) * 22
        color = (0, 0, 255) if "DANGER" in line else (200, 200, 200)
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────

def run(source):
    detector = ObjectDetector()
    alerter  = AlertSystem()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("\n[ADAS] Phase 1 running — press Q to quit\n")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ADAS] Stream ended or frame read failed.")
                break

            # ── Detection ────────────────────────────
            detections, annotated = detector.detect(frame)

            # ── Alerts ───────────────────────────────
            alerter.process(detections)

            # ── Overlays ─────────────────────────────
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now

            if config.SHOW_FPS:
                draw_fps(annotated, fps)
            if config.SHOW_LEGEND:
                draw_legend(annotated)
            draw_summary(annotated, detections)

            # ── Display ──────────────────────────────
            cv2.imshow("ADAS — Phase 1 Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[ADAS] Quit signal received.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        alerter.close()
        print("[ADAS] Session closed. Detections saved to", config.LOG_FILE)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS Phase 1 — Object Detection")
    parser.add_argument(
        "--source",
        default=config.CAMERA_SOURCE,
        help="Camera index (0, 1 …) or path to a video file",
    )
    args = parser.parse_args()

    # Allow passing integer camera index from CLI
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(source)
