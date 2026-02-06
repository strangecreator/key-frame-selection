![logo](pictures/logo.png)

# Key Frame Selection (FindingDory-optimized)

This repository provides a lightweight, CPU-friendly key frame selection library optimized for FindingDory-style data. It can select keyframes from:

- a **video file** (`.mp4`, etc.)
- a **directory of pre-extracted frames** (e.g. `frame_000001.jpg`, ...)

The core keyframe selector `select(...)` was **automatically evolved** via **OpenEvolve**
(https://github.com/algorithmicsuperintelligence/openevolve) using the **inclusion** metric as the optimization target.
The results below are reported using the same evaluation setup.


## What the library does:

Given a sequence of frames, the pipeline:

1. Optionally resizes frames.
2. Detects Shi–Tomasi corners (patched).
3. Tracks them with pyramidal Lucas–Kanade optical flow.
4. Computes cheap per-frame cues (corners, center-corners, edge density, grayscale entropy, low-retention count).
5. Feeds normalized cues + HSV histograms into `select(...)` to output a fixed-budget set of frame indices.

Output is a sorted list of **0-based frame indices**.


## Results:

| Method                | Intersection | Inclusion |
|-----------------------|------------:|----------:|
| Uniform               |      0.1106 |    0.7147 |
| MaxInfo               |      0.1277 |    0.7447 |
| **Ours (OpenEvolve-evolved)** |  **0.1678** | **0.8661** |

**Metrics (oracle protocol).** For each QA instance $q$, a selector outputs a set of indices $S_q$ (budget-limited), and the dataset provides multiple valid evidence sets $G_{q,1}, \dots, G_{q,M_q}$ (each is one acceptable grounding). We report:

**Intersection (precision-style, penalizes over-selection):**

$$
\mathrm{Inter}(q) = \min_{m \in \{1,\dots,M_q\}} \frac{|S_q \cap G_{q,m}|}{\max(1, |S_q|)}.
$$

**Inclusion (evidence coverage across all valid ground-truth sets):**

$$
\mathrm{Incl}(q) = \min_{m \in \{1,\dots,M_q\}} \mathbb{I}\big[|S_q \cap G_{q,m}| > 0\big].
$$

We average these metrics over all evaluated instances.


## Installation:

From the repo root:

```bash
uv venv
source .venv/bin/activate

uv pip install -e .
```


## Quickstart

### Select from a video:

```python
from key_frame_selection import select_from_video
from key_frame_selection.types import PipelineConfig

cfg = PipelineConfig()
indices = select_from_video("path/to/video.mp4", cfg=cfg)  # list[int], sorted
print(indices)
```

### Select from a directory of frames:

```python
from key_frame_selection import select_from_frames_dir
from key_frame_selection.types import PipelineConfig

cfg = PipelineConfig()

indices = select_from_frames_dir(
    "path/to/frames_dir",
    cfg=cfg,
    glob_pattern="*.jpg",  # optional
)
print(indices)
```


## Inputs and outputs

### Inputs:

* **Video mode**: `video_path: str | Path`
* **Frames mode**: `frames_dir: str | Path` (+ optional `glob_pattern`)
* `cfg: PipelineConfig` (defaults are embedded in `types.py`)

### Output:

* `list[int]`: sorted **0-based frame indices** into the input sequence.


## Licence:

MIT