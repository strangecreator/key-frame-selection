from __future__ import annotations

import math
import pathlib
import typing as tp

import cv2 as cv
import numpy as np


_INTERP: dict[str, int] = {
    "nearest": cv.INTER_NEAREST,
    "linear": cv.INTER_LINEAR,
    "cubic": cv.INTER_CUBIC,
    "area": cv.INTER_AREA,
    "lanczos": cv.INTER_LANCZOS4,
}


def read_bgr(path: str | pathlib.Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_bgr(img: np.ndarray, path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv.imwrite(str(p), img)
    if not ok:
        raise IOError(f"Failed to write image: {p}")


def to_gray_u8(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if img_bgr_or_gray.ndim == 2:
        gray = img_bgr_or_gray
    else:
        gray = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        return gray

    # normalize to uint8 deterministically
    out = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    return out.astype(np.uint8)


def resize_image(
    img: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    keep_aspect: bool = False,
    interpolation: str = "linear",
) -> np.ndarray:
    if width is None and height is None:
        return img

    H, W = img.shape[:2]
    inter = _INTERP.get(str(interpolation).lower(), cv.INTER_LINEAR)

    if keep_aspect:
        sx = (width / W) if width else math.inf
        sy = (height / H) if height else math.inf
        s = min(sx, sy)
        if not math.isfinite(s):
            s = sx if math.isfinite(sx) else sy
        newW = max(1, int(round(W * s)))
        newH = max(1, int(round(H * s)))
        return cv.resize(img, (newW, newH), interpolation=inter)

    targetW = int(width) if width is not None else W
    targetH = int(height) if height is not None else H
    return cv.resize(img, (targetW, targetH), interpolation=inter)


def robust_minmax(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    """
    Same behavior as your current code:
    - percentile scaling
    - if degenerate, widen by 1e-6
    """
    if arr.size == 0:
        return arr
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi <= lo:
        hi = lo + 1e-6
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)