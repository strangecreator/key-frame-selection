from __future__ import annotations

import json
import pathlib
import typing as tp

# numpy & related imports
import cv2 as cv
import numpy as np
from tqdm import tqdm

# local imports
from .types import PipelineConfig
from .selection_psfr import select
from .tracking_lk import lucas_kanade_track
from .image_ops import resize_image, to_gray_u8, robust_minmax
from .features_shi_tomasi import shi_tomasi_patched_sorted_fast, custom_sorting_by_quality
from .patching import build_patches, filter_patch_points, remove_duplicate_points_from_second_array


def _gray_entropy(gray_u8: np.ndarray) -> float:
    hist = cv.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / max(1.0, float(hist.sum()))
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _edge_density(gray_u8: np.ndarray, t1: int, t2: int) -> float:
    edges = cv.Canny(gray_u8, int(t1), int(t2))
    return float((edges > 0).sum()) / float(gray_u8.size)


def _hsv_hist_cosine(bgr: np.ndarray, bins: tuple[int, int, int] = (12, 6, 6)) -> np.ndarray:
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).astype(np.float32)
    hist = hist.ravel()
    norm = float(np.linalg.norm(hist)) + 1e-8
    return (hist / norm).astype(np.float32, copy=False)


def _center_rect(W: int, H: int, frac: float) -> tuple[int, int, int, int]:
    cw = max(1, int(W * frac))
    ch = max(1, int(H * frac))
    x = (W - cw) // 2
    y = (H - ch) // 2
    return x, y, cw, ch


def _preprocess_frame(bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    rz = cfg.resize
    return resize_image(
        bgr,
        width=rz.width,
        height=rz.height,
        keep_aspect=rz.keep_aspect,
        interpolation=rz.interpolation,
    )


def _gftt_points(gray_u8: np.ndarray, cfg: PipelineConfig, maxCorners: int) -> np.ndarray:
    shi = cfg.shi_tomasi
    patch = cfg.patching
    return shi_tomasi_patched_sorted_fast(
        gray_u8,
        nw=patch.nw,
        nh=patch.nh,
        centroidal=patch.centroidal,
        max_corners_patch=int(shi.max_corners_patch),
        maxCorners=int(maxCorners),
        dedup_radius=float(shi.dedup_radius),
        qualityLevel=float(shi.qualityLevel),
        minDistance=float(shi.minDistance),
        useHarrisDetector=bool(shi.useHarrisDetector),
        blockSize=int(shi.blockSize),
        gradientSize=int(shi.gradientSize),
        k=float(shi.k),
        custom_sort=custom_sorting_by_quality,
    )


def compute_scores_from_video(
    video_path: str | pathlib.Path,
    cfg: PipelineConfig | None = None,
) -> dict[str, tp.Any]:
    cfg = cfg or PipelineConfig()
    video_path = pathlib.Path(video_path)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    T = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if T <= 0:
        cap.release()
        raise ValueError(f"No frames in video: {video_path}")

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise ValueError(f"Failed to read first frame: {video_path}")

    first_img = _preprocess_frame(first_frame, cfg)
    grey0 = to_gray_u8(first_img)

    H, W = grey0.shape[:2]
    patches = build_patches(W, H, cfg.patching.nw, cfg.patching.nh, cfg.patching.centroidal)

    base_pts = _gftt_points(grey0, cfg, maxCorners=cfg.shi_tomasi.corners_limit_per_image)

    init_counts = np.zeros(len(patches), dtype=np.int32)
    for pid, p in enumerate(patches):
        idx, _ = filter_patch_points(p, base_pts)
        init_counts[pid] = int(idx.size)

    lk = cfg.lucas_kanade
    lk_kwargs = dict(
        winSize=tuple(int(x) for x in lk.winSize),
        maxLevel=int(lk.maxLevel),
        criteria=(int(lk.criteria[0]), int(lk.criteria[1]), float(lk.criteria[2])),
    )

    sel = cfg.selection

    corner_count = np.zeros(T, dtype=np.float32)
    center_corner = np.zeros(T, dtype=np.float32)
    edge_density_arr = np.zeros(T, dtype=np.float32)
    entropy_arr = np.zeros(T, dtype=np.float32)
    lowret_count = np.zeros(T, dtype=np.float32)
    hsv_hists: list[np.ndarray] = [None] * T  # type: ignore

    corner_count[0] = float(len(base_pts))
    x, y, cw, ch = _center_rect(W, H, sel.center_frac)
    if len(base_pts):
        in_center = (
            (base_pts[:, 0] >= x) & (base_pts[:, 0] <= x + cw) &
            (base_pts[:, 1] >= y) & (base_pts[:, 1] <= y + ch)
        )
        center_corner[0] = float(np.count_nonzero(in_center))
    edge_density_arr[0] = float(_edge_density(grey0, sel.canny_t1, sel.canny_t2))
    entropy_arr[0] = float(_gray_entropy(grey0))
    lowret_count[0] = 0.0
    hsv_hists[0] = _hsv_hist_cosine(first_img)

    prevg = grey0

    for i in tqdm(range(1, T), desc=f"scoring {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret or frame is None:
            # stop early if decoding ended
            corner_count = corner_count[:i]
            center_corner = center_corner[:i]
            edge_density_arr = edge_density_arr[:i]
            entropy_arr = entropy_arr[:i]
            lowret_count = lowret_count[:i]
            hsv_hists = hsv_hists[:i]
            T = i
            break

        curr = _preprocess_frame(frame, cfg)
        currg = to_gray_u8(curr)

        tracked_idx, tracked_xy = lucas_kanade_track(
            prevg,
            currg,
            base_pts,
            nw=cfg.patching.nw,
            nh=cfg.patching.nh,
            centroidal=cfg.patching.centroidal,
            max_error=float(lk.max_error),
            **lk_kwargs,
        )

        prevg = currg

        tracked_mask = np.zeros(base_pts.shape[0], dtype=np.uint8)
        tracked_mask[tracked_idx] = 1

        per_patch_tracked = np.zeros(len(patches), dtype=np.int32)
        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, base_pts)
            if idx.size:
                per_patch_tracked[pid] = int(tracked_mask[idx].sum())

        low_ret = 0
        for pid in range(len(patches)):
            denom = int(init_counts[pid])
            if denom > 0 and (per_patch_tracked[pid] / float(denom)) < float(sel.retention_tau):
                low_ret += 1
        lowret_count[i] = float(low_ret)

        refill = max(0, int(cfg.shi_tomasi.corners_limit_per_image) - int(tracked_xy.shape[0]))
        if refill <= 0:
            new_pts = np.empty((0, 2), np.float32)
        else:
            new_pts = _gftt_points(currg, cfg, maxCorners=refill)

        new_pts = remove_duplicate_points_from_second_array(
            tracked_xy,
            new_pts,
            radius=float(cfg.shi_tomasi.dedup_radius),
        )
        base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy

        corner_count[i] = float(len(base_pts))
        x, y, cw, ch = _center_rect(W, H, sel.center_frac)
        if len(base_pts):
            in_center = (
                (base_pts[:, 0] >= x) & (base_pts[:, 0] <= x + cw) &
                (base_pts[:, 1] >= y) & (base_pts[:, 1] <= y + ch)
            )
            center_corner[i] = float(np.count_nonzero(in_center))
        edge_density_arr[i] = float(_edge_density(currg, sel.canny_t1, sel.canny_t2))
        entropy_arr[i] = float(_gray_entropy(currg))
        hsv_hists[i] = _hsv_hist_cosine(curr)

    cap.release()

    # normalize (robust minmax)
    corner_n = robust_minmax(corner_count.astype(np.float32))
    center_n = robust_minmax(center_corner.astype(np.float32))
    edge_n = robust_minmax(edge_density_arr.astype(np.float32))
    entr_n = robust_minmax(entropy_arr.astype(np.float32))
    lowret_n = robust_minmax(lowret_count.astype(np.float32))

    scores_5 = np.stack([corner_n, center_n, edge_n, entr_n, lowret_n], axis=1).astype(np.float32, copy=False)
    hsv_arr = np.array(hsv_hists, dtype=np.float32)

    return {
        "scores": scores_5,
        "hsv_hists": hsv_arr,
        "num_frames": int(T),
    }


def compute_scores_from_frames_dir(
    frames_dir: str | pathlib.Path,
    cfg: PipelineConfig | None = None,
    *,
    glob_pattern: str = "*",
) -> dict[str, tp.Any]:
    cfg = cfg or PipelineConfig()
    frames_dir = pathlib.Path(frames_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in sorted(frames_dir.glob(glob_pattern)) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} (pattern={glob_pattern})")

    first_img = _preprocess_frame(cv.imread(str(paths[0]), cv.IMREAD_COLOR), cfg)
    if first_img is None:
        raise FileNotFoundError(f"Cannot read frame: {paths[0]}")
    grey0 = to_gray_u8(first_img)

    H, W = grey0.shape[:2]
    patches = build_patches(W, H, cfg.patching.nw, cfg.patching.nh, cfg.patching.centroidal)
    base_pts = _gftt_points(grey0, cfg, maxCorners=cfg.shi_tomasi.corners_limit_per_image)

    init_counts = np.zeros(len(patches), dtype=np.int32)
    for pid, p in enumerate(patches):
        idx, _ = filter_patch_points(p, base_pts)
        init_counts[pid] = int(idx.size)

    lk = cfg.lucas_kanade
    lk_kwargs = dict(
        winSize=tuple(int(x) for x in lk.winSize),
        maxLevel=int(lk.maxLevel),
        criteria=(int(lk.criteria[0]), int(lk.criteria[1]), float(lk.criteria[2])),
    )
    sel = cfg.selection

    T = len(paths)
    corner_count = np.zeros(T, dtype=np.float32)
    center_corner = np.zeros(T, dtype=np.float32)
    edge_density_arr = np.zeros(T, dtype=np.float32)
    entropy_arr = np.zeros(T, dtype=np.float32)
    lowret_count = np.zeros(T, dtype=np.float32)
    hsv_hists: list[np.ndarray] = [None] * T  # type: ignore

    corner_count[0] = float(len(base_pts))
    x, y, cw, ch = _center_rect(W, H, sel.center_frac)
    if len(base_pts):
        in_center = (
            (base_pts[:, 0] >= x) & (base_pts[:, 0] <= x + cw) &
            (base_pts[:, 1] >= y) & (base_pts[:, 1] <= y + ch)
        )
        center_corner[0] = float(np.count_nonzero(in_center))
    edge_density_arr[0] = float(_edge_density(grey0, sel.canny_t1, sel.canny_t2))
    entropy_arr[0] = float(_gray_entropy(grey0))
    lowret_count[0] = 0.0
    hsv_hists[0] = _hsv_hist_cosine(first_img)

    prevg = grey0

    for i in tqdm(range(1, T), desc=f"scoring frames in {frames_dir.name}", leave=False):
        bgr = cv.imread(str(paths[i]), cv.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read frame: {paths[i]}")
        curr = _preprocess_frame(bgr, cfg)
        currg = to_gray_u8(curr)

        tracked_idx, tracked_xy = lucas_kanade_track(
            prevg,
            currg,
            base_pts,
            nw=cfg.patching.nw,
            nh=cfg.patching.nh,
            centroidal=cfg.patching.centroidal,
            max_error=float(lk.max_error),
            **lk_kwargs,
        )
        prevg = currg

        tracked_mask = np.zeros(base_pts.shape[0], dtype=np.uint8)
        tracked_mask[tracked_idx] = 1

        per_patch_tracked = np.zeros(len(patches), dtype=np.int32)
        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, base_pts)
            if idx.size:
                per_patch_tracked[pid] = int(tracked_mask[idx].sum())

        low_ret = 0
        for pid in range(len(patches)):
            denom = int(init_counts[pid])
            if denom > 0 and (per_patch_tracked[pid] / float(denom)) < float(sel.retention_tau):
                low_ret += 1
        lowret_count[i] = float(low_ret)

        refill = max(0, int(cfg.shi_tomasi.corners_limit_per_image) - int(tracked_xy.shape[0]))
        if refill <= 0:
            new_pts = np.empty((0, 2), np.float32)
        else:
            new_pts = _gftt_points(currg, cfg, maxCorners=refill)

        new_pts = remove_duplicate_points_from_second_array(
            tracked_xy,
            new_pts,
            radius=float(cfg.shi_tomasi.dedup_radius),
        )
        base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy

        corner_count[i] = float(len(base_pts))
        x, y, cw, ch = _center_rect(W, H, sel.center_frac)
        if len(base_pts):
            in_center = (
                (base_pts[:, 0] >= x) & (base_pts[:, 0] <= x + cw) &
                (base_pts[:, 1] >= y) & (base_pts[:, 1] <= y + ch)
            )
            center_corner[i] = float(np.count_nonzero(in_center))
        edge_density_arr[i] = float(_edge_density(currg, sel.canny_t1, sel.canny_t2))
        entropy_arr[i] = float(_gray_entropy(currg))
        hsv_hists[i] = _hsv_hist_cosine(curr)

    corner_n = robust_minmax(corner_count.astype(np.float32))
    center_n = robust_minmax(center_corner.astype(np.float32))
    edge_n = robust_minmax(edge_density_arr.astype(np.float32))
    entr_n = robust_minmax(entropy_arr.astype(np.float32))
    lowret_n = robust_minmax(lowret_count.astype(np.float32))

    scores_5 = np.stack([corner_n, center_n, edge_n, entr_n, lowret_n], axis=1).astype(np.float32, copy=False)
    hsv_arr = np.array(hsv_hists, dtype=np.float32)

    return {
        "scores": scores_5,
        "hsv_hists": hsv_arr,
        "num_frames": int(T),
        "frame_paths": [str(p) for p in paths],
    }


def select_from_video(
    video_path: str | pathlib.Path,
    *,
    max_frames: int = 16,
    cfg: PipelineConfig | None = None,
) -> list[int]:
    cfg = cfg or PipelineConfig()
    cfg = PipelineConfig(
        resize=cfg.resize,
        patching=cfg.patching,
        shi_tomasi=cfg.shi_tomasi,
        lucas_kanade=cfg.lucas_kanade,
        selection=type(cfg.selection)(
            max_frames=int(max_frames),
            canny_t1=cfg.selection.canny_t1,
            canny_t2=cfg.selection.canny_t2,
            center_frac=cfg.selection.center_frac,
            retention_tau=cfg.selection.retention_tau,
        ),
    )

    data = compute_scores_from_video(video_path, cfg)
    T = int(data["num_frames"])
    scores = data["scores"]
    hsv = data["hsv_hists"]
    allowed = list(range(T))
    return select(scores, hsv, allowed, max_frames=int(max_frames))


def select_from_frames_dir(
    frames_dir: str | pathlib.Path,
    *,
    max_frames: int = 16,
    cfg: PipelineConfig | None = None,
    glob_pattern: str = "*",
    return_paths: bool = False,
) -> list[int] | list[str]:
    cfg = cfg or PipelineConfig()
    data = compute_scores_from_frames_dir(frames_dir, cfg, glob_pattern=glob_pattern)
    T = int(data["num_frames"])
    scores = data["scores"]
    hsv = data["hsv_hists"]
    allowed = list(range(T))
    idx = select(scores, hsv, allowed, max_frames=int(max_frames))

    if return_paths:
        paths = tp.cast(list[str], data["frame_paths"])
        return [paths[i] for i in idx]
    return idx


def dump_scores_json(path: str | pathlib.Path, payload: dict[str, tp.Any]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # numpy -> lists
    out: dict[str, tp.Any] = dict(payload)
    if "scores" in out and isinstance(out["scores"], np.ndarray):
        out["scores"] = out["scores"].tolist()
    if "hsv_hists" in out and isinstance(out["hsv_hists"], np.ndarray):
        out["hsv_hists"] = out["hsv_hists"].tolist()

    with p.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)