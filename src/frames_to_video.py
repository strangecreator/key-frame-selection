import re
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------- backend detection -----------------------------

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ----------------------------- sequence discovery ----------------------------

_NUMERIC_RE = re.compile(r"^(\d+)$")

def _discover_sequence(frames_dir: str, ext: str) -> Tuple[List[Path], Optional[int], Optional[int]]:
    """
    Returns: (sorted_paths, start_number, pad_width)
      - start_number/pad_width are inferred from numeric stems if strictly numeric (e.g., 000001.png).
      - If filenames are not strictly numeric or have gaps, start_number/pad_width may be None.
    """
    
    d = Path(frames_dir)
    if not d.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    candidates = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == f".{ext.lower()}"])
    if not candidates:
        raise FileNotFoundError(f"No *.{ext} frames in: {frames_dir}")

    # Try to derive numeric sequence info
    indices: List[int] = []
    widths: List[int] = []
    numeric_paths: List[Path] = []

    for p in candidates:
        stem = p.stem
        m = _NUMERIC_RE.match(stem)
        if m:
            s = m.group(1)
            indices.append(int(s))
            widths.append(len(s))
            numeric_paths.append(p)

    if numeric_paths and len(numeric_paths) == len(candidates):
        # All are numeric. Sort by index and check contiguity.
        order = sorted(range(len(indices)), key=lambda i: indices[i])
        sorted_paths = [numeric_paths[i] for i in order]
        sorted_idx = [indices[i] for i in order]
        start = sorted_idx[0]
        contiguous = all(sorted_idx[i] == start + i for i in range(len(sorted_idx)))
        if contiguous:
            # Choose the most common width (defensive vs. weird mixes).
            from collections import Counter
            pad_width = Counter(widths).most_common(1)[0][0]
            return sorted_paths, start, pad_width
        else:
            # Numeric but with gaps -> we cannot use %0Nd pattern reliably; fall back to explicit CV2 path list.
            return sorted_paths, None, None
    else:
        # Non-numeric naming -> fallback path list (CV2)
        return candidates, None, None


# ----------------------------- ffmpeg composer -------------------------------

def compose_with_ffmpeg(
    frames_dir: str,
    out_video: str,
    fps: float = 24.0,
    ext: str = "png",
    overwrite: bool = False,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
    pix_fmt: str = "yuv420p"
) -> None:
    """
    Uses ffmpeg image sequence input. Requires numeric, contiguous names like 000001.png.
    Chooses encoder by `codec`:
      - "h264" -> libx264 + CRF/preset (good default, mp4)
      - "vp9"  -> libvpx-vp9 + CRF (good for webm)
      - "mpeg4"-> mpeg4 (ancient, broadly supported)
    """
    paths, start_number, pad_width = _discover_sequence(frames_dir, ext)
    if start_number is None or pad_width is None:
        # Not suitable for pattern input; tell the caller to fallback.
        raise RuntimeError("Non-contiguous or non-numeric sequence; cannot use ffmpeg pattern input.")

    pattern = (Path(frames_dir) / (f"%0{pad_width}d.{ext}")).as_posix()

    # Encoder mapping
    codec = codec.lower()
    if codec == "h264":
        vcodec = "libx264"
        extra = ["-crf", str(int(crf)), "-preset", str(preset)]
        # Force widely-compatible chroma subsampling/pixel format unless user overrides
        pix_f = pix_fmt
    elif codec == "vp9":
        vcodec = "libvpx-vp9"
        # CRF range is ~15(best)-43(worst) for vp9; use ~30 as typical. We'll use provided crf anyway.
        extra = ["-crf", str(int(crf)), "-b:v", "0"]
        pix_f = pix_fmt
    elif codec == "mpeg4":
        vcodec = "mpeg4"
        # mpeg4 doesn't support CRF; use quality scale. Map CRF-ish 18 -> qscale 2 as a decent default.
        qscale = "2" if crf <= 20 else "5"
        extra = ["-qscale:v", qscale]
        pix_f = pix_fmt
    else:
        raise ValueError(f"Unsupported codec: {codec}. Choose from: h264, vp9, mpeg4")

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-framerate", str(float(fps)),
        "-start_number", str(int(start_number)),
        "-i", pattern,
        "-c:v", vcodec,
        *extra,
        "-pix_fmt", pix_f,
        "-movflags", "+faststart",  # safe for MP4; harmless otherwise
        out_video
    ]

    subprocess.run([c for c in cmd if c], check=True)


# ----------------------------- OpenCV composer -------------------------------

def compose_with_cv2(
    frames_dir: str,
    out_video: str,
    fps: float = 24.0,
    ext: str = "png",
    overwrite: bool = False
) -> None:
    """
    Fallback that iterates explicit frames (handles gaps/non-numeric names).
    Will auto-resize frames to the first frame's size if they differ.
    """

    import cv2  # lazy import
    import numpy as np

    paths, _start, _width = _discover_sequence(frames_dir, ext)
    outp = Path(out_video)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_video}. Use --overwrite to replace.")

    # Read first frame to define size
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {paths[0]}")
    H, W = first.shape[:2]

    # Choose FOURCC by extension (portable choice)
    suffix = outp.suffix.lower()
    if suffix in (".mp4", ".m4v", ".mov"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # broad support, h264 often unavailable in OpenCV builds
    elif suffix in (".avi",):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif suffix in (".webm",):
        fourcc = cv2.VideoWriter_fourcc(*"VP90")  # VP9
    else:
        # default to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(outp), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_video}")

    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        writer.write(im)

    writer.release()


# ----------------------------- unified wrapper -------------------------------

def images_to_video(
    frames_dir: str,
    out_video: str,
    fps: float = 24.0,
    ext: str = "png",
    overwrite: bool = False,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
    pix_fmt: str = "yuv420p"
) -> None:
    """
    Compose a video from frames in `frames_dir` (e.g., 000001.png, 000002.png, ...).

    If ffmpeg is available *and* the sequence is numeric & contiguous:
      -> use ffmpeg for speed/quality.
    Otherwise:
      -> fallback to OpenCV path iteration (handles gaps and non-numeric names).
    """
    if have_ffmpeg():
        try:
            compose_with_ffmpeg(frames_dir, out_video, fps, ext, overwrite, codec, crf, preset, pix_fmt)
            return
        except Exception as e:
            print(f"[ffmpeg path] {e}. Falling back to OpenCV...")

    compose_with_cv2(frames_dir, out_video, fps, ext, overwrite)


# ----------------------------------- CLI ------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compose a video from a folder of frames.")
    p.add_argument("frames_dir", help="Directory with frames (e.g., 000001.png, 000002.png, ...)")
    p.add_argument("out_video", help="Output video file (.mp4, .webm, .avi, ...)")
    p.add_argument("--fps", type=float, default=24.0, help="Target frames per second (default: 24).")
    p.add_argument("--ext", choices=["png", "jpg", "jpeg", "webp"], default="png", help="Frame image extension (default: png).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")

    # ffmpeg-specific knobs (ignored by OpenCV path)
    p.add_argument("--codec", choices=["h264", "vp9", "mpeg4"], default="h264", help="Video codec for ffmpeg path (default: h264/libx264).")
    p.add_argument("--crf", type=int, default=18, help="Constant Rate Factor for ffmpeg (lower = better).")
    p.add_argument("--preset", default="medium", help="ffmpeg x264 preset (ultrafast..placebo).")
    p.add_argument("--pix-fmt", default="yuv420p", help="ffmpeg pixel format (default: yuv420p).")

    args = p.parse_args()

    images_to_video(
        frames_dir=args.frames_dir,
        out_video=args.out_video,
        fps=args.fps,
        ext=args.ext,
        overwrite=args.overwrite,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pix_fmt=args.pix_fmt
    )


if __name__ == "__main__":
    main()