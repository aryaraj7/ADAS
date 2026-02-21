# ADAS — Setup & Session Log

## Project Overview

**ADAS** (Advanced Driver Assistance System) — Phase 1: Human, Vehicle, and Animal Detection using YOLOv8 and OpenCV.

---

## Project Structure

```
ADAS/
├── main.py                  # Entry point — detection loop, overlays, display
├── config.py                # Config (camera source, resolution, FPS, log file)
├── requirements.txt         # Python dependencies
├── modules/
│   └── object_detection.py  # YOLOv8-based ObjectDetector class
├── utils/
│   ├── alert_system.py      # Risk alerting (DANGER / WARNING / SAFE)
│   └── distance_estimator.py
└── logs/                    # Detection session logs
```

---

## How to Run

### Prerequisite — activate the correct conda environment

```bash
conda activate yolo-env1
```

> Your terminal prompt must show `(yolo-env1)` before running the project.

### Run with live webcam (default)

```bash
cd /Users/macbook/ADAS
python main.py
```

### Run with a video file

```bash
python main.py --source video.mp4
```

### Controls

| Key | Action      |
|-----|-------------|
| `Q` | Quit / stop |

---

## Dependencies

| Package         | Version      | Purpose                        |
|----------------|--------------|--------------------------------|
| `ultralytics`  | >= 8.2.0     | YOLOv8 object detection        |
| `opencv-python`| >= 4.9.0     | Video capture & frame display  |
| `numpy`        | >= 1.24.0    | Numerical computing            |
| `torch`        | >= 1.8.0     | PyTorch backend for YOLO       |

---

## Environment Setup (Troubleshooting)

### Problem 1 — `ModuleNotFoundError: No module named 'ultralytics'`

**Cause:** Running with the `base` conda environment (Python 3.13), which does not have `ultralytics` or `torch` installed.

**Why `pip install -r requirements.txt` failed in base:**
- Python 3.13 has limited PyTorch support
- Dependency conflict between `ultralytics` and existing packages in `base`

**Solution:** Use the pre-existing `yolo-env1` conda environment, which already has all required packages installed (Python 3.12, ultralytics 8.3.203, torch, etc.).

```bash
conda activate yolo-env1
python main.py
```

### Available Conda Environments

| Environment   | Python | Notes                              |
|--------------|--------|------------------------------------|
| `base`        | 3.13   | No torch/ultralytics — do NOT use  |
| `yolo-env1`   | 3.12   | ultralytics 8.3.203 — **use this** |
| `pcb_yolo`    | 3.10   | ultralytics 8.3.233                |
| `pcb_vlm_env` | —      | PCB vision-language model env      |
| `ssd_env`     | —      | SSD model env                      |
| `ssd_torch`   | —      | SSD + torch env                    |

---

## What the App Does

- Opens webcam (or video file) via OpenCV
- Runs YOLOv8 inference on each frame via `ObjectDetector`
- Classifies detections into: **Human**, **Vehicle**, **Animal**
- Assigns risk levels: **DANGER**, **WARNING**, **SAFE**
- Displays overlays:
  - FPS counter (top-left)
  - Legend (top-right)
  - Detection counts + DANGER alerts (bottom-left)
- Logs detections to file on exit (`config.LOG_FILE`)
