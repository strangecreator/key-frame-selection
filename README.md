![logo](pictures/logo.png)

# Key Frames Selection (adjusted specifically for FindingDory)

A feature-tracking pipeline for **automatic key frame extraction** from long videos.
This branch adds a **content-aware, fixed-length selector** on top of the PSFR tracker so you can output **exactly *K* frames** (e.g., $16$) while avoiding black/monotone walls and favoring object-centric views (the robot gripper + manipulated objects).

> On the FindingDory-style data ($480×640$, ~$640$ frames → $16$), this branch improved our metric from **$0.177$ (uniform-$16$)** to **$0.183$**.


## What’s new in this branch?

1. **Fixed-length output**: `selection.max_frames` → exactly K frames (e.g., $16$).
2. **No forced first frame**: `selection.include_first=false` lets all K slots be useful.
3. **Hard bans on dead frames** (pitch-black / monotone walls): low entropy / low edges / few corners are dropped.
4. **Content score per frame** using only classical CV:

   * corner density, **center-corner density**, edge density (Canny), grayscale entropy,
     mild stability term from PSFR (prefers frames that are not pure turbulence).
5. **Coverage & diversity**:

   * K equal time bins, then takes a small top-pool of candidates per bin by score,
   * greedy selection with **HSV-hist cosine penalty** + a **minimum temporal gap**.
6. **Bug fix**: patches are now built **after resizing** (previously used pre-resize dims).
7. **Diagnostics** saved to `reports/keyframes.json`: per-frame scores and components.


## Why this works on FindingDory?

* **Task need**: identify *what the robot did, which object, and where*. That correlates with **textured, well-lit, object-centric** frames near the gripper (typically near the center), not with extremes of pixel change.
* **Ban monotone**: walls/black frames have low entropy/edges/corners are dropped.
* **Prefer informative content**: more (and centered) corners, more edges, higher entropy mean higher score.
* **Keep coverage**: K bins enforce distribution across the whole episode.
* **Avoid duplicates**: diversity on HSV histograms + min temporal gap prevents repeated near-identical views.


## Method overview

### 1) Tracking backbone: PSFR (unchanged)

* **Shi–Tomasi** corners per patch (with centroidal intersections) and **Lucas–Kanade** sparse optical flow.
* Per-patch retention $r_j^{t+1} = \frac{c_j^{t + 1}}{n_j^\tau}$.
* PSFR keyframe (for logging) if at frame $(t)$: $\mathopen|\{j: r_j^t < \tau_r\}\mathclose| \ge k_{min}$.
* It contributes a small stability preference.

### 2) Frame scoring (new)

For frame $(t)$ we compute (on grayscale unless noted):

* Corner density $(c_t)$ and **center-corner count** $(z_t)$ (within a central window).
* Edge density $(e_t)$ via Canny.
* Entropy $h_t = -\sum_i p_i \log_2 p_i$ from the grayscale histogram.
* Stability proxy $(L_t)$: number of low-retention patches (from PSFR step).

Each feature is **robust-normalized** to $([0, \, 1])$ via $5 – 95\%$ percentiles:
$$
\tilde{x}_t =
\operatorname{clip}\!\Big(
  \frac{x_t - P_5(x)}{P_{95}(x) - P_5(x)},
  \, 0, \, 1
\Big)
$$

Final score:
$$
s_t =
w_{\text{corner}}\,\tilde c_t +
w_{\text{center}}\,\tilde z_t +
w_{\text{edge}}\,\tilde e_t +
w_{\text{entropy}}\,\tilde h_t +
w_{\text{motion}}\,\tilde m_t +
w_{\text{lowret}}\,(1 - \tilde L_t)
$$
(we currently set $\tilde m_t=0$ by default because other cues already capture usefulness and stability).

**Hard filter** (bans dead frames): if any of $\tilde c_t < \theta_c$, $\tilde h_t < \theta_h$, $\tilde e_t < \theta_e$ we set $s_t=-\infty$.

### 3) Fixed-length selection (new)

1. Split timeline into **K equal bins**; from each bin take **top-M candidates** by $(s_t)$ (`per_bin_pool`).
2. Greedy selection of exactly **K frames** maximizing
   $$
   s_t \; - \; \lambda_{\text{div}} \cdot \max_{u \in S}{\cos{\angle(h_t, h_u)}},
   $$
   where $h_t$ is an **L2-normalized HSV histogram** and $S$ is the set of already chosen.
3. Enforce a **minimum temporal gap** between chosen indices.


## `config.json` reference

### `paths`

* `frames_dir`: directory with extracted frames (read by the pipeline).
* `output_dir`: directory for outputs.

### `preprocess`

* `resize.width` (int | null): target width, where `null` keeps original.
* `resize.height` (int | null): target height, where `null` keeps original.
* `keep_aspect` (bool): if true and both sides set — scales to fit (no padding).
* `interpolation` (`nearest`/`linear`/`cubic`/`area`/`lanczos`).

### `patching`

* `nw` (int): patches along width.
* `nh` (int): patches along height.
* `centroidal` (bool): add overlapping “center” patches.

### `shi_tomasi`

* `CORNERS_LIMIT_PER_IMAGE` (int)
* `dedup_radius` (float, px)
* `qualityLevel` (float)
* `minDistance` (float, px)
* `max_corners_patch` (int)
* `useHarrisDetector` (bool)
* `blockSize` (int)
* `gradientSize` (int, odd ≥3)
* `k` (float, Harris)

### `lucas_kanade`

* `winSize` ([int, int])
* `maxLevel` (int)
* `criteria` ([int, int, float])
* `max_error` (float)

### `selection` (new/extended)

* **Output control**

  * `max_frames` (int): exact number of frames to output (e.g., $16$).
  * `include_first` (bool): if `true`, always includes frame $0$, but **we recommend `false`** on FindingDory.
* **Scoring hyperparams**

  * `canny_t1`, `canny_t2` (int): Canny thresholds.
  * `center_frac` (float $\in (0, \, 1]$): central window as a fraction of $W×H$ (e.g., $0.5$).
  * Weights: `w_corner`, `w_center`, `w_edge`, `w_entropy`, `w_motion`, `w_lowret`.
  * Hard-drop thresholds on normalized features: `thr_corner_norm`, `thr_entropy_norm`, `thr_edge_norm`.
* **Coverage & diversity**

  * `min_temporal_gap` (int): minimum frame index gap between picks.
  * `per_bin_pool` (int): top-M candidates kept per bin before greedy selection.
  * `diversity_lambda` (float): cosine penalty weight (HSV histograms).
* **PSFR (still computed; small weight)**

  * `retention_tau` (float)
  * `min_patches_k` (int)
* General

  * `copy_keyframes` (bool)

### `visualization`

* `enabled` (bool), `dir` (str),
* `tracked_color`, `new_color`, `patch_drop_color` (hex),
* `patch_drop_alpha` (float), `point_radius` (int), `point_thickness` (int).


**Tuning tips**

* If you (still) see empty tabletops/walls: try raising `thr_edge_norm` to $0.08$ and/or `w_center` to $0.35$.
* If frames look too similar: try increasing `diversity_lambda` to $0.30–0.40$ or `min_temporal_gap` to $4–6$.
* If content looks under-tracked: try increasing `CORNERS_LIMIT_PER_IMAGE` (e.g., $100$) and reducing `qualityLevel`.


## Usage

Run selection via:

```bash
python src/run.py <absolute-path-to-config-json>
```

Outputs:

* `<output-dir>/reports/keyframes.json`: final frame list, per-frame diagnostics.
* `<output-dir>/reports/keyframes.txt`: plain list of the frames selected.
* `<output-dir>/selected_frames`: copied images (if `copy_keyframes=true`).
* `<output-dir>/visualization`: (if `visualization.enabled=true`).


## Basic commands (utilities)

**Video → Frames**

To convert a video (in `.mp4`, `.mov`, etc.) use the following utility:
```bash
python src/video_to_frames.py <full-path-to-video> <output-frames-directory> --max-fps 4 --ext png --overwrite
```

**Frames → video**

```bash
python src/frames_to_video.py <frames-directory> <full-video-output-path> --fps 24 --ext png --overwrite
```


## Licence

MIT