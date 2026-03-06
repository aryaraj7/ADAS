#!/usr/bin/env python3
"""
ADAS — Custom Object Training Script
=====================================

Train YOLOv8 to detect YOUR own objects (e.g. "fire extinguisher", "helmet", etc.)

STEP-BY-STEP GUIDE:
────────────────────

1. COLLECT IMAGES (50–200 per class minimum)
   - Take photos/screenshots of your object from different angles
   - Put them in: datasets/my_objects/images/train/
   - Optionally add validation images in: datasets/my_objects/images/val/

2. LABEL IMAGES
   Install and use labelImg (free tool):
     pip install labelImg
     labelImg datasets/my_objects/images/train/

   - Draw bounding boxes around your objects
   - Save as YOLO format (.txt files)
   - Labels go in: datasets/my_objects/labels/train/

   Each .txt label file has one line per object:
     <class_id> <x_center> <y_center> <width> <height>
   All values normalized 0-1. Example:
     0 0.5 0.5 0.3 0.4

3. CREATE dataset.yaml (auto-generated below if you run --setup)

4. RUN TRAINING:
     python train_custom.py --data datasets/my_objects/dataset.yaml --epochs 50

5. USE YOUR MODEL:
   After training, update config.py:
     CUSTOM_MODEL_PATH = "runs/detect/train/weights/best.pt"
   Then launch the GUI as normal:
     python gui.py

EXAMPLES:
────────────────────
  # Setup folder structure for 2 classes
  python train_custom.py --setup --classes "helmet,fire_extinguisher"

  # Train on your labeled data
  python train_custom.py --data datasets/my_objects/dataset.yaml --epochs 50

  # Train starting from pre-trained weights (recommended — faster + better)
  python train_custom.py --data datasets/my_objects/dataset.yaml --epochs 100 --base yolov8n.pt

  # Resume interrupted training
  python train_custom.py --resume runs/detect/train/weights/last.pt
"""

import argparse
import os
import sys
import yaml

from ultralytics import YOLO


def setup_dataset(classes: list[str], base_dir: str = "datasets/my_objects"):
    """Create the folder structure and dataset.yaml for custom training."""
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    dataset_yaml = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path = f"{base_dir}/dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\n[SETUP COMPLETE]")
    print(f"  Folder structure created at: {base_dir}/")
    print(f"  Dataset config: {yaml_path}")
    print(f"  Classes: {classes}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Put training images in: {base_dir}/images/train/")
    print(f"  2. Put validation images in: {base_dir}/images/val/")
    print(f"  3. Label them with: pip install labelImg && labelImg {base_dir}/images/train/")
    print(f"     (Save labels as YOLO format)")
    print(f"  4. Train: python train_custom.py --data {yaml_path} --epochs 50")


def train(data_yaml: str, epochs: int, base_model: str, imgsz: int, batch: int):
    """Run YOLOv8 training."""
    print(f"\n[TRAINING]")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Base model: {base_model}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print()

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",  # change to 0 for GPU
        verbose=True,
    )

    best_path = "runs/detect/train/weights/best.pt"
    print(f"\n[TRAINING COMPLETE]")
    print(f"  Best weights: {best_path}")
    print(f"\nTo use your model, edit config.py:")
    print(f'  CUSTOM_MODEL_PATH = "{best_path}"')
    print(f"Then run: python gui.py")
    return results


def resume(weights_path: str):
    """Resume interrupted training."""
    print(f"\n[RESUMING] from {weights_path}")
    model = YOLO(weights_path)
    model.train(resume=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 to detect custom objects for ADAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python train_custom.py --setup --classes "helmet,vest"
  python train_custom.py --data datasets/my_objects/dataset.yaml --epochs 50
  python train_custom.py --resume runs/detect/train/weights/last.pt
        """,
    )
    parser.add_argument("--setup", action="store_true",
                        help="Create dataset folder structure")
    parser.add_argument("--classes", type=str, default="custom_object",
                        help="Comma-separated class names (used with --setup)")
    parser.add_argument("--data", type=str,
                        help="Path to dataset.yaml for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--base", type=str, default="yolov8n.pt",
                        help="Base model to fine-tune from (default: yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--resume", type=str,
                        help="Resume training from weights file")

    args = parser.parse_args()

    if args.setup:
        classes = [c.strip() for c in args.classes.split(",")]
        setup_dataset(classes)
    elif args.resume:
        resume(args.resume)
    elif args.data:
        train(args.data, args.epochs, args.base, args.imgsz, args.batch)
    else:
        parser.print_help()
        print("\n[TIP] Start with: python train_custom.py --setup --classes \"your_object\"")


if __name__ == "__main__":
    main()
