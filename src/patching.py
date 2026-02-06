from __future__ import annotations

import functools
import typing as tp
from dataclasses import dataclass

# numpy & related imports
import numpy as np


@dataclass(frozen=True)
class Patch:
    x: int
    y: int
    w: int
    h: int


@functools.lru_cache
def build_patches(W: int, H: int, nw: int, nh: int, centroidal: bool = True) -> list[Patch]:
    pw, ph = max(1, W // nw), max(1, H // nh)
    base = [Patch(i * pw, j * ph, pw, ph) for j in range(nh) for i in range(nw)]

    if centroidal and (nw > 1 and nh > 1):
        for j in range(nh - 1):
            for i in range(nw - 1):
                x = int((i + 0.5) * pw)
                y = int((j + 0.5) * ph)
                x = max(0, min(x, W - pw))
                y = max(0, min(y, H - ph))
                base.append(Patch(x, y, pw, ph))

    return base


def filter_patch_points(patch: Patch, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_xy is None or points_xy.size == 0:
        return np.empty((0,), np.int32), np.empty((0, 2), np.float32)

    px, py = points_xy[:, 0], points_xy[:, 1]
    mask = (
        (patch.x <= px) & (px <= patch.x + patch.w) &
        (patch.y <= py) & (py <= patch.y + patch.h)
    )
    idx = np.where(mask)[0]
    return idx.astype(np.int32), points_xy[idx].astype(np.float32, copy=False)


def remove_duplicate_points(points_xy: np.ndarray, radius: float = 3.0) -> np.ndarray:
    if points_xy is None or points_xy.size == 0:
        return np.empty((0, 2), np.float32)

    r = max(1.0, float(radius))
    q = np.floor(points_xy / r).astype(np.int32)

    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for i, (qx, qy) in enumerate(q):
        key = (int(qx), int(qy))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    return points_xy[np.array(keep, dtype=np.int32)].astype(np.float32, copy=False)


def remove_duplicate_points_from_second_array(
    old_points_xy: np.ndarray,
    new_points_xy: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    if new_points_xy is None or new_points_xy.size == 0:
        return np.empty((0, 2), np.float32)

    r = max(1.0, float(radius))
    old_q = (
        np.floor(old_points_xy / r).astype(np.int32)
        if (old_points_xy is not None and old_points_xy.size)
        else np.empty((0, 2), np.int32)
    )
    new_q = np.floor(new_points_xy / r).astype(np.int32)

    seen = {(int(qx), int(qy)) for qx, qy in old_q}
    keep: list[int] = []
    for i, (qx, qy) in enumerate(new_q):
        key = (int(qx), int(qy))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    return new_points_xy[np.array(keep, dtype=np.int32)].astype(np.float32, copy=False)