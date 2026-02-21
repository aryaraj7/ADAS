# =============================================================
# ADAS — utils/alert_system.py
#
# Manages real-time alerts for detected objects.
# Prevents alert spam using a per-object cooldown counter.
# =============================================================

import csv
import os
import time
from datetime import datetime
from collections import defaultdict

import config


class AlertSystem:
    """
    Generates console alerts for dangerous detections and
    optionally logs all events to CSV.
    """

    # Risk level → console colour (ANSI)
    _COLORS = {
        "danger":  "\033[91m",   # bright red
        "warning": "\033[93m",   # bright yellow
        "safe":    "\033[92m",   # bright green
        "reset":   "\033[0m",
    }

    def __init__(self):
        # cooldown_counter[object_name] → frames remaining before next alert
        self._cooldown: dict[str, int] = defaultdict(int)
        self._csv_file  = None
        self._csv_writer = None

        if config.ENABLE_LOGGING:
            self._init_csv()

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────

    def process(self, detections: list[dict]) -> None:
        """
        Call once per frame with the detection list.
        Handles alerts + logging.
        """
        # Decrement all cooldowns
        for key in list(self._cooldown.keys()):
            if self._cooldown[key] > 0:
                self._cooldown[key] -= 1

        # Sort by priority then by distance (closest first)
        priority_order = {cat: i for i, cat in enumerate(config.ALERT_PRIORITY)}
        sorted_dets = sorted(
            detections,
            key=lambda d: (
                priority_order.get(d["category"], 99),
                d["distance_m"] if d["distance_m"] is not None else 9999,
            ),
        )

        for det in sorted_dets:
            if config.CONSOLE_ALERTS:
                self._maybe_alert(det)
            if config.ENABLE_LOGGING:
                self._log(det)

    def close(self):
        """Call when the main loop exits to flush the CSV."""
        if self._csv_file:
            self._csv_file.close()

    # ─────────────────────────────────────────
    # Console alerting
    # ─────────────────────────────────────────

    def _maybe_alert(self, det: dict) -> None:
        """Print an alert if the cooldown has expired."""
        key = f"{det['name']}_{det['risk']}"
        if self._cooldown[key] > 0:
            return

        risk   = det["risk"]
        dist   = det["distance_m"]
        name   = det["name"].upper()
        cat    = det["category"]
        conf   = det["confidence"]

        color = self._COLORS.get(risk, "")
        reset = self._COLORS["reset"]

        dist_str = f"{dist:.1f}m" if dist is not None else "??m"
        ts       = datetime.now().strftime("%H:%M:%S")

        if risk == "danger":
            msg = (f"{color}[{ts}] ⚠  DANGER  — {name} detected "
                   f"at ~{dist_str}  ({cat}, conf {conf:.0%}){reset}")
        elif risk == "warning":
            msg = (f"{color}[{ts}] ⚡ WARNING — {name} detected "
                   f"at ~{dist_str}  ({cat}, conf {conf:.0%}){reset}")
        else:
            return  # no alert for safe objects

        print(msg)
        self._cooldown[key] = config.ALERT_COOLDOWN_FRAMES

    # ─────────────────────────────────────────
    # CSV logging
    # ─────────────────────────────────────────

    def _init_csv(self) -> None:
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        file_exists = os.path.exists(config.LOG_FILE)
        self._csv_file   = open(config.LOG_FILE, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(
                ["timestamp", "name", "category", "confidence",
                 "distance_m", "risk", "x1", "y1", "x2", "y2"]
            )

    def _log(self, det: dict) -> None:
        if self._csv_writer is None:
            return
        x1, y1, x2, y2 = det["box"]
        self._csv_writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            det["name"],
            det["category"],
            f"{det['confidence']:.3f}",
            det["distance_m"],
            det["risk"],
            x1, y1, x2, y2,
        ])
