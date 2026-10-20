"""Colour conversion, numpy only.

OpenCV's uint8 BGR->HSV is 86 MB of compiled image-processing code we use exactly
one function from, so it is reimplemented here. It matches cv2.cvtColor
behaviourally: on every fixture in tests/ the resulting "is this a lit gauge arc"
mask is identical pixel for pixel, and the detector's segment counts are unchanged.

The remaining +-1 differences against cv2 on real photographs are rounding ties
that never straddle a threshold (H 108/142, S 100), which is why the equivalence
test asserts on the mask rather than on the channels.
"""
from __future__ import annotations

import numpy as np


def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 BGR -> (H, W, 3) uint8 HSV with H in 0..179.

    Follows OpenCV's formulation: the hue sector is computed in units of 60
    degrees, wrapped into [0, 6) before scaling by 30, and every rounding is
    round-half-up (floor(x + 0.5)), which is what reproduces cv2 exactly on an
    exhaustive set of channel triples.
    """
    blue = frame[..., 0].astype(np.int64)
    green = frame[..., 1].astype(np.int64)
    red = frame[..., 2].astype(np.int64)

    cmax = np.maximum(np.maximum(red, green), blue)
    cmin = np.minimum(np.minimum(red, green), blue)
    diff = cmax - cmin

    safe = np.where(diff == 0, 1, diff).astype(np.float64)
    sector = np.where(red == cmax, (green - blue) / safe,
                      np.where(green == cmax, 2.0 + (blue - red) / safe,
                               4.0 + (red - green) / safe))
    sector = np.where(sector < 0, sector + 6.0, sector)
    hue = np.floor(sector * 30.0 + 0.5).astype(np.int64) % 180
    hue = np.where(diff == 0, 0, hue)

    sat = np.floor(255.0 * diff / np.maximum(cmax, 1) + 0.5).astype(np.int64)
    sat = np.where(cmax == 0, 0, sat)

    return np.stack([hue, sat, cmax], axis=-1).astype(np.uint8)
