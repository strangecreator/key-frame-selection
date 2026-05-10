from __future__ import annotations

import typing as tp
from dataclasses import dataclass


@dataclass(frozen=True)
class ResizeConfig:
    # preprocess.resize
    width: int | None = None
    height: int | None = None

    # preprocess.keep_aspect / preprocess.interpolation
    keep_aspect: bool = False
    interpolation: str = "linear"


@dataclass(frozen=True)
class PatchingConfig:
    # patching
    nw: int = 3
    nh: int = 3
    centroidal: bool = True


@dataclass(frozen=True)
class ShiTomasiConfig:
    # shi_tomasi
    corners_limit_per_image: int = 80
    max_corners_patch: int = 12
    dedup_radius: float = 8.0

    # OpenCV GFTT parameters
    qualityLevel: float = 0.25
    minDistance: float = 12.0
    useHarrisDetector: bool = False
    blockSize: int = 3
    gradientSize: int = 3
    k: float = 0.04


@dataclass(frozen=True)
class LucasKanadeConfig:
    # lucas_kanade
    winSize: tuple[int, int] = (21, 21)
    maxLevel: int = 3
    criteria: tuple[int, int, float] = (3, 30, 0.01)  # (type, maxCount, epsilon)
    max_error: float = 15.0


@dataclass(frozen=True)
class SelectionConfig:
    # selection.max_frames
    max_frames: int = 16

    # used by scoring
    canny_t1: int = 50
    canny_t2: int = 150
    center_frac: float = 0.50

    # LK retention -> lowret_count
    retention_tau: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    resize: ResizeConfig = ResizeConfig()
    patching: PatchingConfig = PatchingConfig()
    shi_tomasi: ShiTomasiConfig = ShiTomasiConfig()
    lucas_kanade: LucasKanadeConfig = LucasKanadeConfig()
    selection: SelectionConfig = SelectionConfig()


ScoresDict = dict[str, tp.Any]