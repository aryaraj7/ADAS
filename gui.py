# =============================================================
# ADAS — gui.py
# CustomTkinter GUI for Phase 1 Detection
#
# Run:  python gui.py
# =============================================================

import threading
import queue
import time
from datetime import datetime

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog

import config
from modules.object_detection import ObjectDetector
from modules.ultrasonic import UltrasonicReader
from utils.alert_system import AlertSystem


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VIDEO_W, VIDEO_H = 800, 500
MAX_ALERT_LINES = 200


class ADASApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ADAS — Advanced Driver Assistance System")
        self.geometry("1180x720")
        self.minsize(960, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._det_queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._detector: ObjectDetector | None = None
        self._alerter: AlertSystem | None = None
        self._ultrasonic: UltrasonicReader | None = None
        self._video_source = None
        self._alert_count = 0

        self._build_ui()
        self._poll_queue()

    # ─────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────

    def _build_ui(self):
        # Main layout: top (video + controls) and bottom (alert log)
        self._top_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._top_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._top_frame.grid_columnconfigure(0, weight=1)
        self._top_frame.grid_columnconfigure(1, weight=0)
        self._top_frame.grid_rowconfigure(0, weight=1)

        self._build_video_panel()
        self._build_control_panel()
        self._build_alert_log()

    def _build_video_panel(self):
        self._video_frame = ctk.CTkFrame(self._top_frame, corner_radius=8)
        self._video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._video_label = ctk.CTkLabel(
            self._video_frame, text="Press Start to begin detection",
            font=ctk.CTkFont(size=16), text_color="gray60",
        )
        self._video_label.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_control_panel(self):
        panel = ctk.CTkScrollableFrame(self._top_frame, width=280, corner_radius=8)
        panel.grid(row=0, column=1, sticky="nsew")

        # ── Source ──
        ctk.CTkLabel(panel, text="SOURCE", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(12, 4))

        src_frame = ctk.CTkFrame(panel, fg_color="transparent")
        src_frame.pack(fill="x", padx=12)

        self._source_var = ctk.StringVar(value="0")
        self._source_entry = ctk.CTkEntry(src_frame, textvariable=self._source_var, width=160)
        self._source_entry.pack(side="left", padx=(0, 6))
        ctk.CTkButton(src_frame, text="Browse", width=70, command=self._browse_file).pack(side="left")

        # ── Start / Stop ──
        btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(10, 4))

        self._start_btn = ctk.CTkButton(
            btn_frame, text="▶  Start", fg_color="#2ea043", hover_color="#3fb950",
            command=self._start_detection)
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._stop_btn = ctk.CTkButton(
            btn_frame, text="■  Stop", fg_color="#da3633", hover_color="#f85149",
            command=self._stop_detection, state="disabled")
        self._stop_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # ── Settings ──
        ctk.CTkLabel(panel, text="SETTINGS", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(16, 4))

        # Confidence threshold
        self._conf_var = ctk.DoubleVar(value=config.CONFIDENCE_THRESHOLD)
        self._conf_label = ctk.CTkLabel(panel, text=f"Confidence: {config.CONFIDENCE_THRESHOLD:.2f}")
        self._conf_label.pack(anchor="w", padx=12)
        ctk.CTkSlider(
            panel, from_=0.1, to=0.95, variable=self._conf_var,
            command=self._on_conf_change,
        ).pack(fill="x", padx=12, pady=(0, 6))

        # Danger distance
        self._danger_var = ctk.DoubleVar(value=config.DISTANCE_DANGER)
        self._danger_label = ctk.CTkLabel(panel, text=f"Ultrasonic danger: {config.DISTANCE_DANGER:.1f} m")
        self._danger_label.pack(anchor="w", padx=12)
        ctk.CTkSlider(
            panel, from_=1.0, to=20.0, variable=self._danger_var,
            command=self._on_danger_change,
        ).pack(fill="x", padx=12, pady=(0, 6))

        # Warning distance
        self._warn_var = ctk.DoubleVar(value=config.DISTANCE_WARNING)
        self._warn_label = ctk.CTkLabel(panel, text=f"Ultrasonic warning: {config.DISTANCE_WARNING:.1f} m")
        self._warn_label.pack(anchor="w", padx=12)
        ctk.CTkSlider(
            panel, from_=5.0, to=50.0, variable=self._warn_var,
            command=self._on_warn_change,
        ).pack(fill="x", padx=12, pady=(0, 6))

        # Checkboxes
        self._fps_var = ctk.BooleanVar(value=config.SHOW_FPS)
        ctk.CTkCheckBox(
            panel, text="Show FPS", variable=self._fps_var,
            command=lambda: setattr(config, "SHOW_FPS", self._fps_var.get()),
        ).pack(anchor="w", padx=12, pady=2)

        self._legend_var = ctk.BooleanVar(value=config.SHOW_LEGEND)
        ctk.CTkCheckBox(
            panel, text="Show Legend", variable=self._legend_var,
            command=lambda: setattr(config, "SHOW_LEGEND", self._legend_var.get()),
        ).pack(anchor="w", padx=12, pady=2)

        # ── Stats ──
        ctk.CTkLabel(panel, text="STATS", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(16, 4))

        stats_frame = ctk.CTkFrame(panel, corner_radius=6)
        stats_frame.pack(fill="x", padx=12, pady=(0, 12))

        self._fps_stat = ctk.CTkLabel(stats_frame, text="FPS: —", anchor="w")
        self._fps_stat.pack(anchor="w", padx=10, pady=(8, 2))
        self._person_stat = ctk.CTkLabel(stats_frame, text="Persons:  0", anchor="w")
        self._person_stat.pack(anchor="w", padx=10, pady=2)
        self._vehicle_stat = ctk.CTkLabel(stats_frame, text="Vehicles: 0", anchor="w")
        self._vehicle_stat.pack(anchor="w", padx=10, pady=2)
        self._animal_stat = ctk.CTkLabel(stats_frame, text="Animals:  0", anchor="w")
        self._animal_stat.pack(anchor="w", padx=10, pady=2)
        self._object_stat = ctk.CTkLabel(stats_frame, text="Objects:  0", anchor="w")
        self._object_stat.pack(anchor="w", padx=10, pady=2)

        # Ultrasonic sensor readout
        self._ultra_stat = ctk.CTkLabel(
            stats_frame, text="Ultrasonic: —", anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._ultra_stat.pack(anchor="w", padx=10, pady=(6, 8))

    def _build_alert_log(self):
        log_frame = ctk.CTkFrame(self, corner_radius=8)
        log_frame.pack(fill="x", padx=8, pady=(4, 8))

        ctk.CTkLabel(log_frame, text="ALERT LOG", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(8, 4))

        self._alert_box = ctk.CTkTextbox(log_frame, height=120, state="disabled", font=ctk.CTkFont(family="Courier", size=12))
        self._alert_box.pack(fill="x", padx=8, pady=(0, 8))
        self._alert_box.tag_config("danger", foreground="#f85149")
        self._alert_box.tag_config("warning", foreground="#d29922")
        self._alert_box.tag_config("info", foreground="#8b949e")

    # ─────────────────────────────────────────
    # Controls callbacks
    # ─────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self._source_var.set(path)

    def _on_conf_change(self, val):
        val = round(val, 2)
        config.CONFIDENCE_THRESHOLD = val
        self._conf_label.configure(text=f"Confidence: {val:.2f}")

    def _on_danger_change(self, val):
        val = round(val, 1)
        config.DISTANCE_DANGER = val
        self._danger_label.configure(text=f"Ultrasonic danger: {val:.1f} m")

    def _on_warn_change(self, val):
        val = round(val, 1)
        config.DISTANCE_WARNING = val
        self._warn_label.configure(text=f"Ultrasonic warning: {val:.1f} m")

    # ─────────────────────────────────────────
    # Start / Stop detection
    # ─────────────────────────────────────────

    def _start_detection(self):
        if self._running:
            return

        # Parse source
        src = self._source_var.get().strip()
        source = int(src) if src.isdigit() else src

        # Test capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._add_alert_text(f"Cannot open source: {source}", "danger")
            cap.release()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        self._running = True
        self._stop_event.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._source_entry.configure(state="disabled")

        self._video_source = cap
        self._detector = ObjectDetector()
        self._alerter = AlertSystem()

        # Start ultrasonic sensor if enabled
        if getattr(config, "ENABLE_ULTRASONIC", False):
            self._ultrasonic = UltrasonicReader(
                port=getattr(config, "ULTRASONIC_PORT", None),
                baudrate=getattr(config, "ULTRASONIC_BAUDRATE", 115200),
            )
            if self._ultrasonic.start():
                self._add_alert_text("Ultrasonic sensor connected.", "info")
            else:
                self._add_alert_text("Ultrasonic sensor not found — running without it.", "warning")
                self._ultrasonic = None

        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        self._add_alert_text("Detection started.", "info")

    def _stop_detection(self):
        if not self._running:
            return
        self._stop_event.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        if self._video_source:
            self._video_source.release()
            self._video_source = None

        if self._ultrasonic:
            self._ultrasonic.stop()
            self._ultrasonic = None

        if self._alerter:
            self._alerter.close()
            self._alerter = None

        self._detector = None

        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._source_entry.configure(state="normal")
        self._add_alert_text("Detection stopped.", "info")

    # ─────────────────────────────────────────
    # Background detection loop (worker thread)
    # ─────────────────────────────────────────

    def _detection_loop(self):
        cap = self._video_source
        detector = self._detector
        alerter = self._alerter
        prev_time = time.time()
        target_spf = 1.0 / getattr(config, "TARGET_FPS", 10)

        while not self._stop_event.is_set():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                self._stop_event.set()
                break

            detections, annotated = detector.detect(frame)
            alerter.process(detections)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now

            # Pace to target FPS
            elapsed = time.time() - frame_start
            if elapsed < target_spf:
                time.sleep(target_spf - elapsed)

            # Read ultrasonic distance
            ultra_dist_m = -1.0
            if self._ultrasonic and self._ultrasonic.connected:
                ultra_dist_m = self._ultrasonic.get_distance_m()

            # Draw overlays on annotated frame (reuse main.py helpers)
            if config.SHOW_FPS:
                _draw_fps(annotated, fps)
            if config.SHOW_LEGEND:
                _draw_legend(annotated)
            if ultra_dist_m >= 0:
                _draw_ultrasonic(annotated, ultra_dist_m)

            # Push to GUI (drop old frames if queue full)
            try:
                self._frame_queue.put_nowait(annotated)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put_nowait(annotated)

            try:
                self._det_queue.put_nowait((detections, fps, ultra_dist_m))
            except queue.Full:
                try:
                    self._det_queue.get_nowait()
                except queue.Empty:
                    pass
                self._det_queue.put_nowait((detections, fps, ultra_dist_m))

    # ─────────────────────────────────────────
    # GUI polling (main thread)
    # ─────────────────────────────────────────

    def _poll_queue(self):
        try:
            frame = self._frame_queue.get_nowait()
            self._update_frame(frame)
        except queue.Empty:
            pass

        try:
            detections, fps, ultra_dist_m = self._det_queue.get_nowait()
            self._update_stats(detections, fps, ultra_dist_m)
            self._process_alerts(detections, ultra_dist_m)
        except queue.Empty:
            pass

        self.after(30, self._poll_queue)

    def _update_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Scale to fit video label
        label_w = self._video_label.winfo_width()
        label_h = self._video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            scale = min(label_w / img.width, label_h / img.height)
            new_w = max(int(img.width * scale), 1)
            new_h = max(int(img.height * scale), 1)
        else:
            new_w, new_h = VIDEO_W, VIDEO_H

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
        self._video_label.configure(image=ctk_img, text="")
        self._video_label._ctk_image = ctk_img  # prevent garbage collection

    def _update_stats(self, detections: list[dict], fps: float, ultra_dist_m: float = -1.0):
        counts = {"human": 0, "vehicle": 0, "animal": 0, "object": 0}
        for d in detections:
            cat = d["category"]
            counts[cat] = counts.get(cat, 0) + 1

        self._fps_stat.configure(text=f"FPS: {fps:.1f}")
        self._person_stat.configure(text=f"Persons:  {counts['human']}")
        self._vehicle_stat.configure(text=f"Vehicles: {counts['vehicle']}")
        self._animal_stat.configure(text=f"Animals:  {counts['animal']}")
        self._object_stat.configure(text=f"Objects:  {counts['object']}")

        # Ultrasonic distance
        if ultra_dist_m >= 0:
            if ultra_dist_m < 0.3:
                color = "#f85149"   # red — very close
            elif ultra_dist_m < 1.0:
                color = "#d29922"   # orange — warning
            else:
                color = "#3fb950"   # green — safe
            self._ultra_stat.configure(
                text=f"Ultrasonic: {ultra_dist_m:.2f} m",
                text_color=color,
            )
        else:
            self._ultra_stat.configure(text="Ultrasonic: —", text_color="gray60")

    def _process_alerts(self, detections: list[dict], ultra_dist_m: float = -1.0):
        if ultra_dist_m < 0:
            return  # No ultrasonic reading, no distance alerts

        # Determine risk from ultrasonic distance
        if ultra_dist_m <= config.DISTANCE_DANGER:
            risk = "danger"
        elif ultra_dist_m <= config.DISTANCE_WARNING:
            risk = "warning"
        else:
            return  # safe, no alert needed

        # Alert once per risk level change (not per detection)
        ts = datetime.now().strftime("%H:%M:%S")
        icon = "⚠" if risk == "danger" else "⚡"
        level = risk.upper()
        obj_names = ", ".join(set(d["name"].upper() for d in detections)) if detections else "OBSTACLE"
        msg = f"[{ts}] {icon} {level} — {obj_names} at {ultra_dist_m:.2f}m (ultrasonic)"
        self._add_alert_text(msg, risk)

    def _add_alert_text(self, text: str, tag: str = "info"):
        self._alert_box.configure(state="normal")
        self._alert_box.insert("end", text + "\n", tag)
        self._alert_count += 1
        # Trim old lines
        if self._alert_count > MAX_ALERT_LINES:
            self._alert_box.delete("1.0", "2.0")
            self._alert_count -= 1
        self._alert_box.see("end")
        self._alert_box.configure(state="disabled")

    # ─────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────

    def _on_closing(self):
        self._stop_detection()
        self.destroy()


# ─────────────────────────────────────────
# Overlay helpers (copied from main.py to avoid import coupling)
# ─────────────────────────────────────────

def _draw_fps(frame, fps: float):
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_legend(frame):
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


def _draw_ultrasonic(frame, dist_m: float):
    if dist_m < 0.3:
        color = (0, 0, 255)       # red
        label = f"ULTRASONIC: {dist_m:.2f}m  DANGER"
    elif dist_m < 1.0:
        color = (0, 165, 255)     # orange
        label = f"ULTRASONIC: {dist_m:.2f}m  WARNING"
    else:
        color = (0, 220, 0)       # green
        label = f"ULTRASONIC: {dist_m:.2f}m"

    h = frame.shape[0]
    cv2.putText(frame, label, (12, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    app = ADASApp()
    app.mainloop()
