"""Connected components, without OpenCV."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Blob:
    area: int
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float


def label_blobs(mask: np.ndarray) -> list[Blob]:
    """8-connectivity components of a boolean mask, largest first.

    A plain flood fill rather than a vectorised scheme: the mask is the set of
    lit gauge pixels, typically a couple of thousand inside a 200k-pixel band,
    so walking the set bits is both faster and far easier to trust than an
    iterative label-propagation loop.
    """
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    blobs: list[Blob] = []
    for sy, sx in np.argwhere(mask):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        queue = deque([(sy, sx)])
        ys: list[int] = []
        xs: list[int] = []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= w or seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        ay, ax = np.asarray(ys), np.asarray(xs)
        blobs.append(Blob(area=ay.size, x=int(ax.min()), y=int(ay.min()),
                          w=int(ax.max() - ax.min() + 1), h=int(ay.max() - ay.min() + 1),
                          cx=float(ax.mean()), cy=float(ay.mean())))
    blobs.sort(key=lambda b: -b.area)
    return blobs
