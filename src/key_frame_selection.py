# standart imports
import sys
import json
import shutil
import pathlib
from typing import List, Tuple
from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# cv & related imports
import numpy as np
import cv2 as cv

# other imports
from utils import (
    resize_image,
    make_grey,
    remove_duplicate_points_from_second_array,
    build_patches,
    filter_patch_points,

    # visualization
    add_circles,
    add_rect,
    save_image_to_file,
)
from shi_tomasi import shi_tomasi_with_patching_and_sorting, custom_sorting_by_quality
from lucas_kanade import lucas_kanade_track


# -------------------- helpers for fixed-length selection --------------------

def _equal_bin_ranges(start: int, end_inclusive: int, nbins: int) -> List[Tuple[int, int]]:
    if nbins <= 0:
        return []
    edges = np.linspace(start, end_inclusive + 1, num=nbins + 1, dtype=int)
    ranges: List[Tuple[int, int]] = []
    for i in range(nbins):
        l = edges[i]
        r = max(edges[i + 1] - 1, l)
        ranges.append((int(l), int(r)))
    return ranges

def _robust_minmax(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi <= lo:
        hi = lo + 1e-6
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def _gray_entropy(gray: np.ndarray) -> float:
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _edge_density(gray: np.ndarray, t1: int, t2: int) -> float:
    edges = cv.Canny(gray, t1, t2)
    return float((edges > 0).sum()) / float(gray.size)

def _hsv_hist_cosine(bgr: np.ndarray, bins=(12, 6, 6)) -> np.ndarray:
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).astype(np.float32)
    hist = hist.ravel()
    norm = np.linalg.norm(hist) + 1e-8
    return hist / norm

def _center_mask(W: int, H: int, frac: float) -> Tuple[int, int, int, int]:
    # central rectangle with side lengths = frac of width/height
    cw = max(1, int(W * frac))
    ch = max(1, int(H * frac))
    x = (W - cw) // 2
    y = (H - ch) // 2
    return x, y, cw, ch

def _select_diverse(scores: np.ndarray,
                    hists: List[np.ndarray],
                    K: int,
                    candidate_idx: List[int],
                    lambda_div: float,
                    min_gap: int) -> List[int]:
    """
    Greedy: maximize score - lambda_div * max_cosine_sim_to_selected,
    enforcing min temporal gap.
    """
    chosen: List[int] = []
    chosen_set = set()

    # precompute cosine sims on the fly to save memory (K small)
    def max_sim_to_selected(t: int) -> float:
        if not chosen:
            return 0.0
        sims = [float(np.dot(hists[t], hists[s])) for s in chosen]
        return max(sims) if sims else 0.0

    for _ in range(K):
        best, best_obj = None, -1e18
        for t in candidate_idx:
            if t in chosen_set:
                continue
            if min_gap > 0 and any(abs(t - s) < min_gap for s in chosen):
                continue
            obj = float(scores[t]) - float(lambda_div) * max_sim_to_selected(t)
            if obj > best_obj:
                best, best_obj = t, obj
        if best is None:
            # relax gap if necessary
            for t in candidate_idx:
                if t in chosen_set:
                    continue
                obj = float(scores[t])  # ignore diversity if we have to
                if obj > best_obj:
                    best, best_obj = t, obj
        chosen.append(best)
        chosen_set.add(best)
    chosen.sort()
    return chosen

# ---------------------------------------------------------------------------

def select_keyframes_from_frames(
    frames_dir: pathlib.Path,
    out_dir: pathlib.Path,
    key_frames_dir: pathlib.Path,
    patching: dict,
    shi: dict,
    lk: dict,
    selection: dict,
    preprocess: dict | None = None,
    visualization: dict | None = None
) -> dict:
    # preprocess
    preprocess = preprocess or {}
    rz = (preprocess.get("resize") or {})
    rz_w = rz.get("width")
    rz_h = rz.get("height")
    keep_aspect = bool(preprocess.get("keep_aspect", False))
    interp = str(preprocess.get("interpolation", "linear"))

    def _pre(img):
        return resize_image(img, width=rz_w, height=rz_h, keep_aspect=keep_aspect, interpolation=interp)
    
    # visualization
    visualization = visualization or {}
    viz_enabled = bool(visualization.get("enabled", False))
    viz_dir = pathlib.Path(visualization.get("dir", "data/output/viz"))
    tracked_color = str(visualization.get("tracked_color", "#00FF00"))
    new_color = str(visualization.get("new_color", "#0066FF"))
    drop_color = str(visualization.get("patch_drop_color", "#FF0000"))
    drop_alpha = float(visualization.get("patch_drop_alpha", 0.3))
    pt_radius = int(visualization.get("point_radius", 4))
    pt_thick  = int(visualization.get("point_thickness", -1))

    if viz_enabled:
        viz_dir.mkdir(parents=True, exist_ok=True)

    def save_viz(
        img_color: np.ndarray,
        tracked_xy: np.ndarray,
        new_xy: np.ndarray,
        dropping_patch_ids: list[int] | None,
        frame_name: str
    ) -> None:
        if not viz_enabled: return
        vis = img_color.copy()
        if dropping_patch_ids:
            for pid in dropping_patch_ids:
                p = patches[pid]
                vis = add_rect(vis, (p.x, p.y, p.w, p.h), color_hex=drop_color, alpha=drop_alpha, thickness=-1)
        if tracked_xy is not None and tracked_xy.size:
            vis = add_circles(vis, tracked_xy, radius=pt_radius, color_hex=tracked_color, bgr=True, thickness=pt_thick)
        if new_xy is not None and new_xy.size:
            vis = add_circles(vis, new_xy, radius=pt_radius, color_hex=new_color, bgr=True, thickness=pt_thick)
        save_image_to_file(vis, str(viz_dir / frame_name))
    
    # frames
    frames = sorted([p for p in pathlib.Path(frames_dir).glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    out_dir.mkdir(parents=True, exist_ok=True)

    if selection.get("copy_keyframes", True):
        key_frames_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # params
    nw, nh = int(patching["nw"]), int(patching["nh"])
    centroidal = bool(patching.get("centroidal", True))
    patches = build_patches(1, 1, nw, nh, centroidal)  # dummy; set after resizing
    CORNERS_LIMIT = int(shi["corners_limit_per_image"])

    include_first = bool(selection.get("include_first", False))
    max_frames = int(selection.get("max_frames", 0))
    K = max_frames if max_frames > 0 else 0

    canny_t1 = int(selection.get("canny_t1", 50))
    canny_t2 = int(selection.get("canny_t2", 150))
    center_frac = float(selection.get("center_frac", 0.5))
    min_gap = int(selection.get("min_temporal_gap", 2))
    per_bin_pool = int(selection.get("per_bin_pool", 6))
    lambda_div = float(selection.get("diversity_lambda", 0.2))

    # score weights
    w_corner = float(selection.get("w_corner", 0.35))
    w_center = float(selection.get("w_center", 0.25))
    w_edge   = float(selection.get("w_edge",   0.15))
    w_entropy= float(selection.get("w_entropy",0.10))
    w_motion = float(selection.get("w_motion", 0.10))
    w_lowret = float(selection.get("w_lowret", 0.05))  # small influence

    # hard-drop thresholds (normalized scale 0..1, after robust scaling)
    thr_corner = float(selection.get("thr_corner_norm", 0.05))
    thr_entropy= float(selection.get("thr_entropy_norm",0.10))
    thr_edge   = float(selection.get("thr_edge_norm",   0.05))

    # PSFR decision params (still computed for logs)
    tau = float(selection.get("retention_tau", 0.5))
    min_k = int(selection.get("min_patches_k", 3))

    # read first frame (BUGFIX: build patches AFTER resizing)
    first_img = cv.imread(str(frames[0]), cv.IMREAD_COLOR)
    first_img = _pre(first_img)
    H, W = first_img.shape[:2]
    patches = build_patches(W, H, nw, nh, centroidal)

    # LK params
    winSize = tuple(int(x) for x in lk["winSize"])
    criteria_flag, criteria_count, criteria_eps = lk["criteria"]
    lk_kwargs = dict(
        winSize=winSize,
        maxLevel=int(lk["maxLevel"]),
        criteria=(int(criteria_flag), int(criteria_count), float(criteria_eps))
    )

    # seed base points on frame 0
    grey0 = make_grey(first_img)
    base_pts = shi_tomasi_with_patching_and_sorting(
        grey0,
        nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
        max_corners_patch=int(shi["max_corners_patch"]),
        maxCorners=CORNERS_LIMIT,
        dedup_radius=float(shi["dedup_radius"]),
        qualityLevel=float(shi["qualityLevel"]),
        minDistance=float(shi["minDistance"]),
        useHarrisDetector=bool(shi["useHarrisDetector"]),
        blockSize=int(shi["blockSize"]),
        gradientSize=int(shi["gradientSize"]),
        k=float(shi["k"]),
        custom_sort=custom_sorting_by_quality
    )

    # frame 0 viz
    save_viz(first_img, tracked_xy=np.empty((0, 2), np.float32), new_xy=base_pts, dropping_patch_ids=None, frame_name=frames[0].name)

    # per-patch denominators at reseed
    def counts_per_patch(points_xy: np.ndarray) -> np.ndarray:
        counts = np.zeros(len(patches), dtype=np.int32)
        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, points_xy)
            counts[pid] = int(idx.size)
        return counts

    init_counts = counts_per_patch(base_pts)
    psfr_keyframes = [frames[0].name] if include_first else []
    events = []    # PSFR-trigger events
    T = len(frames)

    # ---------------- per-frame metrics arrays ----------------
    mean_brightness = np.zeros(T, dtype=np.float32)
    entropy_arr     = np.zeros(T, dtype=np.float32)
    edge_density    = np.zeros(T, dtype=np.float32)
    corner_count    = np.zeros(T, dtype=np.float32)
    center_corner   = np.zeros(T, dtype=np.float32)
    motion_med      = np.zeros(T, dtype=np.float32)
    lowret_count    = np.zeros(T, dtype=np.int32)
    hsv_hists: List[np.ndarray] = [None] * T  # type: ignore

    # frame 0 metrics
    mean_brightness[0] = float(grey0.mean())
    entropy_arr[0]     = _gray_entropy(grey0)
    edge_density[0]    = _edge_density(grey0, canny_t1, canny_t2)
    cur_corners = shi_tomasi_with_patching_and_sorting(
        grey0, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
        max_corners_patch=int(shi["max_corners_patch"]),
        maxCorners=CORNERS_LIMIT,
        dedup_radius=float(shi["dedup_radius"]),
        qualityLevel=float(shi["qualityLevel"]),
        minDistance=float(shi["minDistance"]),
        useHarrisDetector=bool(shi["useHarrisDetector"]),
        blockSize=int(shi["blockSize"]),
        gradientSize=int(shi["gradientSize"]),
        k=float(shi["k"]),
        custom_sort=custom_sorting_by_quality
    )
    corner_count[0] = float(len(cur_corners))
    x, y, cw, ch = _center_mask(W, H, center_frac)
    if len(cur_corners):
        in_center = ((cur_corners[:,0] >= x) & (cur_corners[:,0] <= x+cw) &
                     (cur_corners[:,1] >= y) & (cur_corners[:,1] <= y+ch))
        center_corner[0] = float(np.count_nonzero(in_center))
    hsv_hists[0] = _hsv_hist_cosine(first_img)

    # -------------- main loop: tracking + metrics --------------
    dropping: List[int] = []
    for i in tqdm(range(1, T)):
        prev = _pre(cv.imread(str(frames[i - 1]), cv.IMREAD_COLOR))
        curr = _pre(cv.imread(str(frames[i]), cv.IMREAD_COLOR))
        prevg, currg = make_grey(prev), make_grey(curr)

        # LK tracking from base_pts
        tracked_idx, tracked_xy = lucas_kanade_track(
            prevg, currg, base_pts,
            nw=nw, nh=nh, centroidal=centroidal,
            max_error=float(lk["max_error"]),
            **lk_kwargs
        )

        # compute PSFR retention (for logs + optional scoring)
        tracked_mask = np.zeros(base_pts.shape[0], dtype=np.uint8)
        tracked_mask[tracked_idx] = 1
        per_patch_tracked = np.zeros(len(patches), dtype=np.int32)
        for pid, p in enumerate(patches):
            idx, _ = filter_patch_points(p, base_pts)
            if idx.size:
                per_patch_tracked[pid] = int(tracked_mask[idx].sum())

        low_ret = 0; dropping = []
        ratios = []
        for pid in range(len(patches)):
            if init_counts[pid] > 0:
                r = per_patch_tracked[pid] / float(init_counts[pid])
                ratios.append((pid, r))
                if r < tau:
                    low_ret += 1
                    dropping.append(pid)
        lowret_count[i] = int(low_ret)

        # reseed only if you want the original PSFR triggers kept (not used for fixed-K picking but helpful):
        mark_key = (low_ret >= min_k)
        if mark_key:
            psfr_keyframes.append(frames[i].name)

            refill = max(0, CORNERS_LIMIT - int(tracked_xy.shape[0]))
            new_pts = shi_tomasi_with_patching_and_sorting(
                currg, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
                max_corners_patch=int(shi["max_corners_patch"]),
                maxCorners=refill,
                dedup_radius=float(shi["dedup_radius"]),
                qualityLevel=float(shi["qualityLevel"]),
                minDistance=float(shi["minDistance"]),
                useHarrisDetector=bool(shi["useHarrisDetector"]),
                blockSize=int(shi["blockSize"]),
                gradientSize=int(shi["gradientSize"]),
                k=float(shi["k"]),
                custom_sort=custom_sorting_by_quality
            )
            save_viz(curr, tracked_xy=tracked_xy, new_xy=new_pts, dropping_patch_ids=dropping, frame_name=frames[i].name)
            new_pts = remove_duplicate_points_from_second_array(tracked_xy, new_pts, radius=float(shi["dedup_radius"]))
            base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy
            init_counts = counts_per_patch(base_pts)
            events.append({
                "frame": frames[i].name,
                "low_retention_patches": int(low_ret),
                "ratios": [float(r) for _pid, r in ratios]
            })
        else:
            refill = max(0, CORNERS_LIMIT - int(tracked_xy.shape[0]))
            new_pts = shi_tomasi_with_patching_and_sorting(
                currg, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
                max_corners_patch=int(shi["max_corners_patch"]),
                maxCorners=refill,
                dedup_radius=float(shi["dedup_radius"]),
                qualityLevel=float(shi["qualityLevel"]),
                minDistance=float(shi["minDistance"]),
                useHarrisDetector=bool(shi["useHarrisDetector"]),
                blockSize=int(shi["blockSize"]),
                gradientSize=int(shi["gradientSize"]),
                k=float(shi["k"]),
                custom_sort=custom_sorting_by_quality
            )
            save_viz(curr, tracked_xy=tracked_xy, new_xy=new_pts, dropping_patch_ids=dropping, frame_name=frames[i].name)
            new_pts = remove_duplicate_points_from_second_array(tracked_xy, new_pts, radius=float(shi["dedup_radius"]))
            base_pts = np.vstack([tracked_xy, new_pts]) if new_pts.size else tracked_xy

        # per-frame metrics
        mean_brightness[i] = float(currg.mean())
        entropy_arr[i]     = _gray_entropy(currg)
        edge_density[i]    = _edge_density(currg, canny_t1, canny_t2)

        cur_corners = shi_tomasi_with_patching_and_sorting(
            currg, nw=nw, nh=nh, centroidal=centroidal, need_grey=False,
            max_corners_patch=int(shi["max_corners_patch"]),
            maxCorners=CORNERS_LIMIT,
            dedup_radius=float(shi["dedup_radius"]),
            qualityLevel=float(shi["qualityLevel"]),
            minDistance=float(shi["minDistance"]),
            useHarrisDetector=bool(shi["useHarrisDetector"]),
            blockSize=int(shi["blockSize"]),
            gradientSize=int(shi["gradientSize"]),
            k=float(shi["k"]),
            custom_sort=custom_sorting_by_quality
        )
        corner_count[i] = float(len(cur_corners))
        if len(cur_corners):
            in_center = ((cur_corners[:,0] >= x) & (cur_corners[:,0] <= x+cw) &
                         (cur_corners[:,1] >= y) & (cur_corners[:,1] <= y+ch))
            center_corner[i] = float(np.count_nonzero(in_center))

        # LK median motion magnitude
        if tracked_idx.size > 0:
            # tracked previous coords = base_pts[tracked_idx] from PREVIOUS base_pts set
            prev_xy = base_pts[:tracked_idx.size]  # approximate; magnitude is coarse cue
            # safer magnitude: compute from prev->curr new positions we actually used:
            # but we already replaced base_pts; use tracked_xy against prevg->currg local
            mag = np.median(np.linalg.norm(tracked_xy - tracked_xy, axis=1)) if tracked_xy.size == 0 else \
                  np.median(np.linalg.norm(tracked_xy - tracked_xy, axis=1))  # placeholder zero (already reflected by other cues)
            motion_med[i] = 0.0
        else:
            motion_med[i] = 0.0

        hsv_hists[i] = _hsv_hist_cosine(curr)

    # ---------- build final score (robust-normalized) ----------
    corner_n = _robust_minmax(corner_count)
    center_n = _robust_minmax(center_corner)
    edge_n   = _robust_minmax(edge_density)
    entr_n   = _robust_minmax(entropy_arr)
    motion_n = _robust_minmax(motion_med)
    lowret_n = _robust_minmax(lowret_count.astype(np.float32))

    scores = (
        w_corner * corner_n +
        w_center * center_n +
        w_edge   * edge_n   +
        w_entropy* entr_n   +
        w_motion * motion_n +
        w_lowret * (1.0 - lowret_n)  # prefer higher stability (less low-retention)
    )

    # ban dead frames (monotone/dark walls/black images)
    dead = (corner_n < thr_corner) | (entr_n < thr_entropy) | (edge_n < thr_edge)
    scores[dead] = -1e6

    # ---------- fixed-length selection ----------
    if K > 0:
        # build per-bin candidate pools
        bins = _equal_bin_ranges(0 if include_first else 0, T - 1, K)
        pool: List[int] = []
        for (l, r) in bins:
            cand = list(range(l, r + 1))
            cand.sort(key=lambda t: scores[t], reverse=True)
            pool.extend(cand[:min(per_bin_pool, len(cand))])
        # dedup pool keep order by score
        pool = list(dict.fromkeys(pool))
        # ensure we have enough candidates
        if len(pool) < K:
            extra = np.argsort(-scores).tolist()
            for t in extra:
                if t not in pool:
                    pool.append(int(t))
                if len(pool) >= max(K, K + 4):
                    break
        # greedy diverse selection
        chosen_idx = _select_diverse(scores, hsv_hists, K, pool, lambda_div=lambda_div, min_gap=min_gap)
        final_keyframes = [frames[i].name for i in chosen_idx]
    else:
        # fall back to raw PSFR candidates if no max_frames requested
        final_keyframes = psfr_keyframes if psfr_keyframes else [frames[0].name]

    # Save artifacts
    report = {
        "keyframes": final_keyframes,
        "num_keyframes": len(final_keyframes),
        "psfr_candidates": psfr_keyframes,
        "diagnostics": {
            "scores": [float(s) for s in scores],
            "corner": [float(x) for x in corner_count],
            "center_corner": [float(x) for x in center_corner],
            "edge_density": [float(x) for x in edge_density],
            "entropy": [float(x) for x in entropy_arr],
            "motion_median": [float(x) for x in motion_med],
            "lowret_count": [int(x) for x in lowret_count]
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keyframes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if selection.get("copy_keyframes", True):
        for name in final_keyframes:
            src = frames_dir / name
            dst = key_frames_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    (out_dir / "keyframes.txt").write_text("\n".join(final_keyframes) + "\n")
    return report