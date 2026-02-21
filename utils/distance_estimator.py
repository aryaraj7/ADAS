# =============================================================
# ADAS — utils/distance_estimator.py
#
# Pinhole camera distance estimation using bounding-box height.
#
# Formula:
#   distance (m) = (real_height_m × focal_length_px) / pixel_height
#
# This is an approximation — accuracy improves with proper
# camera calibration (see config.py → FOCAL_LENGTH_PX).
# =============================================================

import config


def estimate_distance(category: str, pixel_height: int) -> float | None:
    """
    Estimate distance in metres from the camera to a detected object.

    Parameters
    ----------
    category     : "human" | "vehicle" | "animal"
    pixel_height : bounding box height in pixels

    Returns
    -------
    Estimated distance in metres, or None if pixel_height is 0.
    """
    if pixel_height <= 0:
        return None

    real_height = config.REAL_WORLD_HEIGHTS.get(category)
    if real_height is None:
        return None

    distance = (real_height * config.FOCAL_LENGTH_PX) / pixel_height
    return round(distance, 2)
