# =============================================================
# ADAS — modules/depth_estimator.py
#
# MiDaS AI Monocular Depth Estimation
#
# How it works:
#   1. Each frame is passed through the MiDaS neural network
#   2. MiDaS outputs a relative inverse-depth map (higher = closer)
#   3. The map is normalized to [0, 1] and mapped to metres using
#      config.MIDAS_MIN_RANGE and config.MIDAS_MAX_RANGE
#   4. For each detected object, the median depth in the center
#      of its bounding box is used as the distance estimate
#
# Model auto-downloads on first run via torch.hub (~100 MB).
# =============================================================

import cv2
import numpy as np
import torch

import config


class DepthEstimator:
    """
    Wraps MiDaS for per-frame monocular depth estimation.

    Usage:
        depth_est = DepthEstimator()
        depth_map = depth_est.estimate(frame)          # full depth map
        dist = depth_est.get_distance(depth_map, box)  # metres for one box
    """

    def __init__(self):
        print(f"[DepthEstimator] Loading MiDaS model: {config.MIDAS_MODEL_TYPE} ...")
        self.device = torch.device(config.DEVICE)
        self.model, self.transform = self._load_model()
        self.model.to(self.device).eval()
        print(f"[DepthEstimator] Ready on device: {config.DEVICE}")

    # ─────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────

    def _load_model(self):
        model_type = config.MIDAS_MODEL_TYPE

        model = torch.hub.load(
            "intel-isl/MiDaS",
            model_type,
            trust_repo=True,
        )

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )

        if model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        return model, transform

    # ─────────────────────────────────────────
    # Depth map generation
    # ─────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Run MiDaS on a BGR frame.

        Returns
        -------
        depth_map : np.ndarray (float32, same H×W as frame)
            Values in [0, 1] where 1.0 = closest, 0.0 = farthest.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]  (MiDaS: higher raw value = closer)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map[:] = 0.0

        return depth_map

    # ─────────────────────────────────────────
    # Per-object distance query
    # ─────────────────────────────────────────

    def get_distance(self, depth_map: np.ndarray, box: list) -> float | None:
        """
        Estimate distance (metres) for one detection bounding box.

        Strategy: sample the median depth value in a small patch at the
        center of the bounding box (more stable than a single pixel).

        Parameters
        ----------
        depth_map : normalized depth map from estimate()
        box       : [x1, y1, x2, y2]

        Returns
        -------
        Distance in metres, or None if the region is invalid.
        """
        x1, y1, x2, y2 = box
        h, w = depth_map.shape[:2]

        # Center patch — 20% of box size, minimum 5 px
        bw, bh = x2 - x1, y2 - y1
        pw = max(int(bw * 0.2), 5)
        ph = max(int(bh * 0.2), 5)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        px1 = max(cx - pw, 0)
        px2 = min(cx + pw, w)
        py1 = max(cy - ph, 0)
        py2 = min(cy + ph, h)

        patch = depth_map[py1:py2, px1:px2]
        if patch.size == 0:
            return None

        depth_val = float(np.median(patch))   # 1.0 = closest, 0.0 = farthest

        # Map to metres:
        #   depth_val=1.0  →  MIDAS_MIN_RANGE (closest)
        #   depth_val=0.0  →  MIDAS_MAX_RANGE (farthest)
        lo = config.MIDAS_MIN_RANGE
        hi = config.MIDAS_MAX_RANGE
        distance = hi - (hi - lo) * depth_val

        return round(distance, 2)
