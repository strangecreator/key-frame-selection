![logo](pictures/logo.png)

# Key Frames Selection

A feature-tracking-based pipeline for **automatic key frame extraction** from long videos.  
It combines **Shi–Tomasi corner detection**, **spatial patching with intersections**, and **Lucas–Kanade sparse optical flow** tracking to identify frames that best represent temporal changes.

---

## Overview

This project detects **key moments** in a video by analyzing the persistence of visual features through time.

1. **Feature Detection — Shi–Tomasi Algorithm**
   - Detects strong corner features for each patch.
   - Based on the minimum eigenvalue of the gradient covariance matrix:
     $R = \min(\lambda_1, \lambda_2)$ where $(\lambda_1, \lambda_2)$ are eigenvalues of the image gradient matrix.

2. **Patch Division & Intersection**
   - Each frame is divided into spatial patches (e.g. $3×3$ or $4×4$ grid).
   - Features are tracked **independently** for each patch.
   - Patches are **overlapping** to ensure smooth continuity at boundaries.

3. **Feature Tracking — Lucas–Kanade Optical Flow**
   - Tracks feature movement between consecutive frames using:
     $I(x + u, \, y + v, \, t + 1) \approx I(x, \, y, \, t)$, solving for optical flow vector $(u, v)$ via least squares.

4. **Key Frame Selection**
   - As frames progress, some features are lost (occlusion, motion blur, etc.).
   - A frame is marked as **key** when **multiple patches simultaneously** fall below a predefined feature-retention threshold:
     $\frac{N_{\text{tracked}}}{N_{\text{initial}}} < \tau$ across $\ge k$ patches.


## Licence

MIT