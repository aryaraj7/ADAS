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
        Handles logging. Console alerts are now handled by the GUI
        using ultrasonic sensor distance.
        """
        # Decrement all cooldowns
        for key in list(self._cooldown.keys()):
            if self._cooldown[key] > 0:
                self._cooldown[key] -= 1

        for det in detections:
            if config.ENABLE_LOGGING:
                self._log(det)

    def close(self):
        """Call when the main loop exits to flush the CSV."""
        if self._csv_file:
            self._csv_file.close()

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
                 "x1", "y1", "x2", "y2"]
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
            x1, y1, x2, y2,
        ])
