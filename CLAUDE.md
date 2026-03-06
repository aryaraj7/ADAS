# ADAS — Advanced Driver Assistance System

## Project Overview
Advanced Driver Assistance System for a rover, running on a local machine with a USB webcam. Performs real-time object detection, depth estimation, object sizing, body-part keypoints, drowsiness monitoring, hand detection, TTC (time-to-collision), and radar integration. Includes RL training for autonomous navigation and 3D room scanning.

## Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8n (`models/yolov8n.pt`) — COCO 80-class |
| Object Tracking | ByteTrack (Ultralytics built-in) |
| Pose Estimation | YOLOv8n-pose — 17 body keypoints + skeleton |
| Depth Estimation | MiDaS_small (`torch.hub`) |
| Hand Detection | MediaPipe Hands (bounding box only) |
| Drowsiness | MediaPipe FaceMesh + EAR (Eye Aspect Ratio) |
| Optical Flow / TTC | Lucas-Kanade sparse flow + bbox looming |
| Radar | TI AWR1843 77GHz FMCW — TLV binary protocol |
| GUI | CustomTkinter (dark theme, scrollable) |
| Video I/O | OpenCV (cv2) — capture, drawing, color conversion |
| Display Pipeline | OpenCV frame → PIL → CTkImage |
| RL Training | Stable-Baselines3 PPO + Gymnasium |
| 3D Scanning | MiDaS DPT_Large + Open3D + Three.js |
| Runtime | Python 3.12 (conda env `adas`), CPU-only (torch 2.2.2, numpy 1.26.4) |

## Project Structure
```
├── gui.py                  # CustomTkinter GUI entry point
├── main.py                 # CLI entry point (original)
├── config.py               # All settings (model, classes, thresholds, FPS, display)
├── train_custom.py         # Custom object training pipeline
├── modules/
│   ├── object_detection.py # YOLOv8 detection + pose + drawing + size estimation
│   └── depth_estimator.py  # MiDaS depth estimation
├── utils/
│   ├── alert_system.py     # Console alerts + CSV logging
│   └── distance_estimator.py # Pinhole camera distance fallback
├── models/                 # .pt weights (gitignored, auto-downloaded)
├── logs/                   # detections.csv (gitignored)
└── requirements.txt
```

## Running
```bash
conda activate adas
python gui.py          # GUI mode
python main.py         # CLI mode
```

## Key Architecture Decisions

- **Threading**: Detection runs in a background thread; GUI polls via `after(30ms)` + `queue.Queue`
- **FPS control**: `config.TARGET_FPS = 10` — detection loop uses `perf_counter` pacing (no drift)
- **All classes**: `config.DETECT_ALL_CLASSES = True` — detects all 80 COCO classes
- **Size estimation**: `real_size = (pixel_size × depth) / focal_length` using MiDaS depth
- **Depth colour coding**: red <1m (danger), orange 1–3m (warning), green >3m (safe)
- **Config as runtime state**: GUI sliders mutate `config.*` attributes directly; takes effect next frame

### Why MiDaS over DepthAnything v2?
DepthAnything v2 was the bottleneck causing 3 FPS actual throughput. MiDaS_small is significantly faster while still providing usable depth for size estimation.

### Why MediaPipe Hands as bounding box only?
Hand landmarks are computed internally (MediaPipe), but only the bounding box around all 21 landmarks is drawn — treated like any other YOLO detection.

### TTC (Time-To-Collision)
```
looming_rate = d(bbox_area)/dt / bbox_area
TTC = 1 / looming_rate  (seconds)
```

### AWR1843 — ESP32 Integration Path
```
AWR1843 UART (TLV) → serial port → Python radar.py → nearest() → process_frame()
AWR1843 UART → ESP32 TWAI → SN65HVD230 CAN transceiver → CAN Bus → Jetson Nano (ROS 2)
```

## Environment Notes
- Python 3.13 does NOT work (no PyTorch wheels). Must use Python 3.12.
- numpy must be <2.0 (torch 2.2.2 incompatible with numpy 2.x)
- opencv-python 4.10.x works with numpy 1.x despite pip warning
- conda env: `/opt/anaconda3/envs/adas/bin/python`
- USB webcam hardware cap: ~15 fps actual throughput
- Full pipeline (camera + YOLO + MiDaS + pose): achieves 10 FPS target

## FPS Notes
- Target: 10 FPS (enforced by perf_counter timer in detection loop)
- YOLOv8n: ~25–30 FPS on CPU at 640×480
- MiDaS_small adds ~20–30ms per frame
- Resolution: 640×480 (reduced from 1280×720 for performance)

## Custom Training
```bash
python train_custom.py --setup --classes "my_object"
# Label images with labelImg, then:
python train_custom.py --data datasets/my_objects/dataset.yaml --epochs 50
# Set CUSTOM_MODEL_PATH in config.py to use trained model
```

## Bugs Fixed

| Bug | Fix |
|-----|-----|
| 3 FPS throughput | Removed DepthAnything v2, reduced resolution to 640×480 |
| numpy 2.x crash with torch 2.2.2 | Pin `numpy<2` (`numpy==1.26.4`) |
| Python 3.13 no torch wheels | Use Python 3.12 conda env |
| `fetch()` CORS error on file:// | Embed base64 point cloud data directly in HTML as JS variable |
| transformers/torch version conflict | Avoided `transformers` pipeline; used `torch.hub.load` for MiDaS |

## Dependencies
```
ultralytics
opencv-python
torch torchvision
numpy<2
customtkinter
Pillow
timm
PyYAML
```
