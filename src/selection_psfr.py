from __future__ import annotations

import sys

# numpy & related imports
import numpy as np


def select(
    scores: np.ndarray,
    hsv_hists: np.ndarray,
    allowed_indices: list[int],
    *,
    max_frames: int = 16,
) -> list[int]:
    """
    PSFR-style selector.

    Expected scores shape: (T, 5) with columns:
      0 corner_n
      1 center_n
      2 edge_n
      3 entr_n
      4 lowret_n
    """

    frames_count = int(scores.shape[0])
    if frames_count <= 0 or max_frames <= 0:
        return []

    if len(allowed_indices) <= max_frames:
        return sorted(allowed_indices)

    allowed = sorted(allowed_indices)
    n = len(allowed)
    n_float = float(n)
    if n <= max_frames:
        return allowed

    s5 = scores[allowed]  # (n,5)
    # emulate previous 6D vector by inserting motion=0 at index 4:
    # [corner, center, edge, entr, motion(0), lowret]
    motion_zeros = np.zeros((n, 1), dtype=np.float32)
    s = np.hstack([s5[:, 0:4], motion_zeros, s5[:, 4:5]]).astype(np.float32, copy=False)  # (n,6)

    # old weights: [0.35, 0.25, 0.15, 0.10, 0.10, -0.05]
    # motion term is always 0 => does nothing, but we keep it via emulation.
    std = np.std(s, axis=0) + 1e-9
    signs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float32)
    adapt_weights = signs * std / (np.sum(std) + 1e-9)
    orig_weights = np.array([0.35, 0.25, 0.15, 0.10, 0.10, -0.05], dtype=np.float32)
    weights_all = 0.6 * orig_weights + 0.4 * adapt_weights

    base_quality = s.dot(weights_all)  # (n,)

    # peak detection (quality)
    R_peak = max(1, n // (3 * max_frames))
    peak_bonus = np.zeros(n, dtype=np.float32)
    is_quality_peak = np.zeros(n, dtype=bool)
    for i in range(n):
        left = max(0, i - R_peak)
        right = min(n, i + R_peak + 1)
        if base_quality[i] == np.max(base_quality[left:right]):
            if left < i and right > i + 1:
                neighbor_mean = (np.mean(base_quality[left:i]) + np.mean(base_quality[i + 1 : right])) / 2
                if base_quality[i] > neighbor_mean * 1.05:
                    peak_bonus[i] = 0.20 * base_quality[i]
                    is_quality_peak[i] = True

    # motion peak detection (emulated zeros => no peaks)
    motion_vals = s[:, 4]  # all zeros by construction
    R_motion = max(1, n // (4 * max_frames))
    motion_bonus = np.zeros(n, dtype=np.float32)
    is_motion_peak = np.zeros(n, dtype=bool)
    for i in range(n):
        left = max(0, i - R_motion)
        right = min(n, i + R_motion + 1)
        if motion_vals[i] == np.max(motion_vals[left:right]):
            if left < i and right > i + 1:
                neighbor_mean = (np.mean(motion_vals[left:i]) + np.mean(motion_vals[i + 1 : right])) / 2
                if motion_vals[i] > neighbor_mean * 1.05:
                    motion_bonus[i] = 0.10 * motion_vals[i]
                    is_motion_peak[i] = True

    synergy_bonus = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if is_quality_peak[i] and is_motion_peak[i]:
            synergy_bonus[i] = 0.02 * (base_quality[i] + motion_vals[i]) / 2.0

    current_score = base_quality + peak_bonus + motion_bonus + synergy_bonus

    # cumulative importance
    imp_shift = current_score - current_score.min() + 1e-9
    cum_imp = np.cumsum(imp_shift)

    # normalize hsv histograms for allowed frames
    hsv_allowed = hsv_hists[allowed].astype(np.float32, copy=False)
    norms = np.linalg.norm(hsv_allowed, axis=1, keepdims=True)
    hsv_allowed = hsv_allowed / (norms + 1e-9)

    # cumulative hsv change
    if n > 1:
        dots = np.sum(hsv_allowed[1:] * hsv_allowed[:-1], axis=1)
        dist = 1.0 - dots
        cumdist = np.zeros(n, dtype=np.float32)
        cumdist[1:] = np.cumsum(dist).astype(np.float32)
        diff = np.zeros(n, dtype=np.float32)
        diff[1:] = dist.astype(np.float32)
    else:
        cumdist = np.zeros(n, dtype=np.float32)
        diff = np.zeros(n, dtype=np.float32)

    quality_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        quality_diff[1:] = np.abs(current_score[1:] - current_score[:-1]).astype(np.float32)

    # other diffs (motion/lowret/edge/entropy) – motion is zeros
    motion_diff = np.zeros(n, dtype=np.float32)
    lowret_diff = np.zeros(n, dtype=np.float32)
    edge_diff = np.zeros(n, dtype=np.float32)
    entr_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        lowret_vals = s[:, 5]
        edge_vals = s[:, 2]
        entr_vals = s[:, 3]
        lowret_diff[1:] = np.abs(lowret_vals[1:] - lowret_vals[:-1]).astype(np.float32)
        edge_diff[1:] = np.abs(edge_vals[1:] - edge_vals[:-1]).astype(np.float32)
        entr_diff[1:] = np.abs(entr_vals[1:] - entr_vals[:-1]).astype(np.float32)

    combined_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        combined_diff[1:] = (
            0.5 * diff[1:]
            + 0.2 * quality_diff[1:]
            + 0.1 * motion_diff[1:]   # zeros, preserved
            + 0.05 * lowret_diff[1:]
            + 0.1 * edge_diff[1:]
            + 0.05 * entr_diff[1:]
        ).astype(np.float32)

    strong_peaks: list[int] = []
    if n > 1:
        peaks: list[int] = []
        for i in range(1, n - 1):
            if combined_diff[i] > combined_diff[i - 1] and combined_diff[i] > combined_diff[i + 1]:
                peaks.append(i)
        if peaks:
            median_diff = float(np.median(combined_diff[1:]))
            mean_diff = float(np.mean(combined_diff[1:]))
            threshold = max(0.25 * median_diff, 0.15 * mean_diff)
            strong_peaks = [p for p in peaks if float(combined_diff[p]) > threshold]
            strong_peaks.sort(key=lambda p: float(combined_diff[p]), reverse=True)

    scene_bonus = combined_diff * 0.10
    current_score = current_score + scene_bonus

    if strong_peaks:
        strong_peaks_set = set(strong_peaks)
        for i in range(n):
            if is_quality_peak[i] and i in strong_peaks_set:
                current_score[i] += 0.015 * (combined_diff[i] + base_quality[i]) / 2.0

    imp_shift = current_score - current_score.min() + 1e-9
    cum_imp = np.cumsum(imp_shift)

    # average change
    avg_dist = 0.0 if float(cumdist[-1]) < 1e-12 else float(cumdist[-1]) / n

    # choose slots
    if float(cumdist[-1]) < 1e-12 and float(cum_imp[-1]) < 1e-12:
        targets = np.linspace(0, n - 1, max_frames, endpoint=False) + (n - 1) / (2 * max_frames)
        targets = np.clip(targets, 0, n - 1)
        slot_indices = np.searchsorted(np.arange(n), targets, side="left")
    else:
        ndist = cumdist / float(cumdist[-1]) if float(cumdist[-1]) >= 1e-12 else np.zeros(n, dtype=np.float32)
        nqual = cum_imp / float(cum_imp[-1]) if float(cum_imp[-1]) >= 1e-12 else np.zeros(n, dtype=np.float32)

        comb = 0.7 * ndist + 0.3 * nqual

        t_uniform = (np.arange(n) / (n - 1)).astype(np.float32) if n > 1 else np.zeros(n, dtype=np.float32)

        comb_weight = 0.5 if avg_dist <= 0.0 else 0.2 + 0.4 * min(1.0, avg_dist * 10.0)
        uniform_weight = 1.0 - comb_weight
        comb_mixed = comb_weight * comb + uniform_weight * t_uniform

        segment_length = float(comb_mixed[-1]) / max_frames
        targets = segment_length * (np.arange(max_frames) + 0.5)
        slot_indices = np.searchsorted(comb_mixed, targets, side="left")
        slot_indices = np.clip(slot_indices, 0, n - 1).tolist()

    base_radius = 0.04 * n / max_frames
    factor = 2.0 if avg_dist <= 0.0 else 1.0 + (1.0 - min(1.0, avg_dist * 10.0))
    window_radius = max(1, int(base_radius * factor))

    if strong_peaks:
        adjusted_slots: list[int] = []
        peak_used: set[int] = set()
        for slot in slot_indices:
            window = max(window_radius * 2, n // (max_frames * 2))
            candidates = [p for p in strong_peaks if abs(p - slot) <= window and p not in peak_used]
            if candidates:
                best = max(candidates, key=lambda p: float(combined_diff[p]))
                adjusted_slots.append(best)
                peak_used.add(best)
            else:
                unused = [p for p in strong_peaks if p not in peak_used]
                if unused:
                    strongest_unused = unused[0]
                    if abs(strongest_unused - slot) <= n // max_frames:
                        adjusted_slots.append(strongest_unused)
                        peak_used.add(strongest_unused)
                    else:
                        adjusted_slots.append(int(slot))
                else:
                    adjusted_slots.append(int(slot))
        slot_indices = adjusted_slots

    selected_allowed: list[int] = []
    selected_set: set[int] = set()

    diversity_weight = 1.0 if avg_dist <= 0.0 else 0.3 + 0.7 * (1.0 - min(1.0, avg_dist * 5.0))
    temp_weight = 0.20
    suppress_radius = max(1, n // (2 * max_frames))
    suppress_decay = 0.40

    slot_indices = [int(x) for x in slot_indices]

    for slot in slot_indices:
        left = max(0, slot - window_radius)
        right = min(n, slot + window_radius + 1)
        candidates = [c for c in range(left, right) if c not in selected_set]

        if not candidates:
            for expand in range(1, n):
                left = max(0, slot - expand)
                right = min(n, slot + expand + 1)
                candidates = [c for c in range(left, right) if c not in selected_set]
                if candidates:
                    break

        if not candidates:
            unselected = [c for c in range(n) if c not in selected_set]
            if unselected:
                unselected_arr = np.array(unselected, dtype=np.int32)
                quality_scores = current_score[unselected_arr]
                distance_penalty = np.abs(unselected_arr - slot) / n_float
                combined_scores = 0.7 * quality_scores - 0.3 * distance_penalty
                best = int(unselected_arr[int(np.argmax(combined_scores))])
            else:
                best = int(slot)
        else:
            cand_arr = np.array(candidates, dtype=np.int32)
            if selected_allowed:
                selected_arr = np.array(selected_allowed, dtype=np.int32)
                sims = np.dot(hsv_allowed[cand_arr], hsv_allowed[selected_arr].T)
                max_sim = np.max(sims, axis=1)

                abs_diff = np.abs(cand_arr[:, None] - selected_arr[None, :])
                min_temp = np.min(abs_diff, axis=1) / n_float

                adjusted = (
                    current_score[cand_arr]
                    + diversity_weight * (1.0 - max_sim)
                    + temp_weight * min_temp
                    + 0.25 * (1.0 - np.abs(cand_arr - slot) / n_float)
                )
            else:
                adjusted = current_score[cand_arr] + 0.25 * (1.0 - np.abs(cand_arr - slot) / n_float)

            best = int(cand_arr[int(np.argmax(adjusted))])

        selected_allowed.append(best)
        selected_set.add(best)

        lo = max(0, best - suppress_radius)
        hi = min(n, best + suppress_radius + 1)
        current_score[lo:hi] *= suppress_decay

    selected_frames = [allowed[i] for i in selected_allowed]
    sys.stderr.write(
        f"[select] selected {len(selected_frames)} frames using content-based uniform windows with scene-cut alignment (motion emulated as 0)\n"
    )
    return sorted(selected_frames)