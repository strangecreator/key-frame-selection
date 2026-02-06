from __future__ import annotations

import functools
import typing as tp

# numpy & related imports
import cv2 as cv
import numpy as np

# local imports
from .patching import build_patches, remove_duplicate_points


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-z)))


def _bilinear_sample(map2d: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    h, w = map2d.shape[:2]
    x = np.clip(points_xy[:, 0], 0, w - 1 - 1e-6)
    y = np.clip(points_xy[:, 1], 0, h - 1 - 1e-6)
    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = x - x0
    dy = y - y0

    Ia = map2d[y0, x0]
    Ib = map2d[y0, x1]
    Ic = map2d[y1, x0]
    Id = map2d[y1, x1]
    return Ia * (1 - dx) * (1 - dy) + Ib * dx * (1 - dy) + Ic * (1 - dx) * dy + Id * dx * dy


def _corner_response_map(
    gray_u8: np.ndarray,
    blockSize: int,
    gradientSize: int,
    use_harris: bool,
    k: float,
) -> np.ndarray:
    gradientSize = int(max(3, gradientSize) | 1)  # odd
    blockSize = int(max(2, blockSize))
    if use_harris:
        return cv.cornerHarris(gray_u8, blockSize=blockSize, ksize=gradientSize, k=float(k))
    return cv.cornerMinEigenVal(gray_u8, blockSize=blockSize, ksize=gradientSize)


def round_robin_interleave_scored(lists_scored: list[np.ndarray], max_total: int | None = None) -> np.ndarray:
    out: list[np.ndarray] = []
    rank = 0
    while True:
        added = False
        for li in lists_scored:
            if rank < len(li):
                out.append(li[rank])
                added = True
                if max_total is not None and len(out) >= max_total:
                    return np.vstack(out).astype(np.float32)
        if not added:
            break
        rank += 1
    return np.vstack(out).astype(np.float32) if out else np.empty((0, 3), np.float32)


def custom_sorting_by_quality(lists_scored: list[np.ndarray], temperature: float = 0.2, stable: bool = False) -> np.ndarray:
    keys: list[float] = []
    for _list in lists_scored:
        for i, (score, _x, _y) in enumerate(_list):
            keys.append(float(score - temperature * i * (1 - sigmoid(float(score) - 0.3))))

    arr = np.vstack(lists_scored).astype(np.float32, copy=False)
    order = np.argsort(-np.array(keys, dtype=np.float32), kind=("mergesort" if stable else "quicksort"))
    return arr[order]


@functools.lru_cache(maxsize=64)
def _cached_patch_rois_and_masks(
    W: int,
    H: int,
    nw: int,
    nh: int,
    centroidal: bool,
    blockSize: int,
    gradientSize: int,
) -> tuple[tuple[int, int, int, int, np.ndarray], ...]:
    """
    Returns tuples of (rx0, ry0, rx1, ry1, mask_roi_uint8) for each patch.
    mask_roi_uint8 selects ONLY the original patch area inside the padded ROI.
    """
    patches = build_patches(W, H, nw, nh, centroidal=centroidal)

    g = int(max(3, gradientSize) | 1)
    b = int(max(2, blockSize))
    pad = (g // 2) + (b // 2) + 2

    out: list[tuple[int, int, int, int, np.ndarray]] = []
    for p in patches:
        rx0 = max(0, p.x - pad)
        ry0 = max(0, p.y - pad)
        rx1 = min(W, p.x + p.w + pad)
        ry1 = min(H, p.y + p.h + pad)

        roi_w = rx1 - rx0
        roi_h = ry1 - ry0

        ix0 = p.x - rx0
        iy0 = p.y - ry0

        m = np.zeros((roi_h, roi_w), dtype=np.uint8)
        m[iy0 : iy0 + p.h, ix0 : ix0 + p.w] = 255
        m.setflags(write=False)

        out.append((rx0, ry0, rx1, ry1, m))

    return tuple(out)


def shi_tomasi_patched_sorted_fast(
    gray_u8: np.ndarray,
    *,
    nw: int = 5,
    nh: int = 5,
    centroidal: bool = True,
    max_corners_patch: int = 10,
    maxCorners: int = 25,
    dedup_radius: float = 5.0,
    custom_sort: tp.Callable | None = None,
    **gftt_kwargs,
) -> np.ndarray:
    """
    Fast patchwise GFTT:
    - ROI + cached masks
    - scores using response map bilinear sampling
    - custom_sort merges scored candidates
    """
    if maxCorners <= 0 or max_corners_patch <= 0:
        return np.empty((0, 2), np.float32)

    if custom_sort is None:
        custom_sort = round_robin_interleave_scored

    H, W = gray_u8.shape[:2]
    blockSize = int(gftt_kwargs.get("blockSize", 3))
    gradientSize = int(gftt_kwargs.get("gradientSize", 3))

    roi_specs = _cached_patch_rois_and_masks(W, H, nw, nh, bool(centroidal), int(blockSize), int(gradientSize))

    per_patch_pts: list[np.ndarray] = []
    any_pts = False

    for (rx0, ry0, rx1, ry1, mask_roi) in roi_specs:
        roi = gray_u8[ry0:ry1, rx0:rx1]
        cand = cv.goodFeaturesToTrack(
            roi,
            mask=mask_roi,
            maxCorners=int(max_corners_patch),
            **gftt_kwargs,
        )
        if cand is None:
            per_patch_pts.append(np.empty((0, 2), np.float32))
            continue

        pts = cand.reshape(-1, 2).astype(np.float32, copy=False)
        pts[:, 0] += float(rx0)
        pts[:, 1] += float(ry0)
        per_patch_pts.append(pts)
        any_pts = True

    if not any_pts:
        return np.empty((0, 2), np.float32)

    response_map = _corner_response_map(
        gray_u8,
        blockSize=int(blockSize),
        gradientSize=int(gradientSize),
        use_harris=bool(gftt_kwargs.get("useHarrisDetector", False)),
        k=float(gftt_kwargs.get("k", 0.04)),
    )

    lists_scored: list[np.ndarray] = []
    for cand in per_patch_pts:
        if cand.size == 0:
            lists_scored.append(np.empty((0, 3), np.float32))
            continue
        scores = _bilinear_sample(response_map, cand)
        order = np.argsort(-scores)
        lists_scored.append(np.column_stack([scores[order], cand[order]]).astype(np.float32, copy=False))

    merged = custom_sort(lists_scored)
    if merged.size == 0:
        return np.empty((0, 2), np.float32)

    xy = merged[:, 1:3].astype(np.float32, copy=False)
    xy = remove_duplicate_points(xy, radius=float(dedup_radius))
    return xy[: int(maxCorners)].astype(np.float32, copy=False)