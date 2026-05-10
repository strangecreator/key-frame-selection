#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import pathlib
import sys
import time
import traceback
import typing as tp

import numpy as np
from tqdm import tqdm

from key_frame_selection import compute_scores_from_frames_dir
from key_frame_selection.types import PipelineConfig


DEFAULT_INPUT_DIR = pathlib.Path(
    "/workspace/dataspace/openevolve-project/examples/psfr_gens_intersection_maximization/datasets/gens/videos/no_face"
)
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "/workspace/dataspace/openevolve-project/examples/psfr_gens_intersection_maximization/datasets/gens/precomputed"
)


def _init_worker() -> None:
    try:
        import cv2 as cv

        cv.setNumThreads(1)
        try:
            cv.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass


def _save_npz_atomic(
    output_path: pathlib.Path,
    scores: np.ndarray,
    hsv_hists: np.ndarray,
    compressed: bool,
) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        if compressed:
            np.savez_compressed(tmp_path, scores=scores, hsv_hists=hsv_hists)
        else:
            np.savez(tmp_path, scores=scores, hsv_hists=hsv_hists)

        actual_tmp = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        os.replace(actual_tmp, output_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        alt_tmp = tmp_path.with_suffix(tmp_path.suffix + ".npz")
        if alt_tmp.exists():
            try:
                alt_tmp.unlink()
            except Exception:
                pass


def _process_one(task: tuple[str, str, bool, bool]) -> dict[str, tp.Any]:
    video_dir_str, output_dir_str, overwrite, compressed = task
    video_dir = pathlib.Path(video_dir_str)
    output_dir = pathlib.Path(output_dir_str)
    output_path = output_dir / f"{video_dir.name}.npz"

    try:
        if output_path.exists() and not overwrite:
            return {
                "status": "skip",
                "video": video_dir.name,
                "output": str(output_path),
            }

        cfg = PipelineConfig()
        payload = compute_scores_from_frames_dir(video_dir, cfg=cfg)

        scores = np.asarray(payload["scores"], dtype=np.float32)
        hsv_hists = np.asarray(payload["hsv_hists"], dtype=np.float32)

        output_dir.mkdir(parents=True, exist_ok=True)
        _save_npz_atomic(output_path, scores=scores, hsv_hists=hsv_hists, compressed=compressed)

        return {
            "status": "ok",
            "video": video_dir.name,
            "output": str(output_path),
            "num_frames": int(payload["num_frames"]),
            "scores_shape": tuple(int(x) for x in scores.shape),
            "hsv_hists_shape": tuple(int(x) for x in hsv_hists.shape),
        }
    except Exception as e:
        return {
            "status": "error",
            "video": video_dir.name,
            "output": str(output_path),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _discover_video_dirs(input_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_dir())


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 96))


def _default_chunksize(num_tasks: int, workers: int) -> int:
    return max(1, min(64, num_tasks // max(1, workers * 8)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--failures-log", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    video_dirs = _discover_video_dirs(input_dir)
    if not video_dirs:
        print(f"No video directories found in: {input_dir}", file=sys.stderr)
        return 1

    workers = max(1, int(args.workers))
    chunksize = int(args.chunksize) if args.chunksize is not None else _default_chunksize(len(video_dirs), workers)

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(video_dir), str(output_dir), bool(args.overwrite), bool(args.compressed))
        for video_dir in video_dirs
    ]

    ok = 0
    skipped = 0
    errors = 0
    failures: list[dict[str, tp.Any]] = []

    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Video dirs:  {len(video_dirs)}")
    print(f"Workers:     {workers}")
    print(f"Chunksize:   {chunksize}")
    print(f"Overwrite:   {bool(args.overwrite)}")
    print(f"Compressed:  {bool(args.compressed)}")

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="precomputing",
        ):
            status = result["status"]

            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                failures.append(result)
                tqdm.write(f"[ERROR] {result['video']}: {result['error']}")

    print()
    print(f"Done. ok={ok}, skipped={skipped}, errors={errors}")

    if failures:
        failures_log = args.failures_log or (output_dir / "_failures.txt")
        with failures_log.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(f"{item['video']}\n")
                f.write(f"{item['error']}\n")
                f.write(item["traceback"])
                f.write("\n")
                f.write("=" * 80)
                f.write("\n")
        print(f"Failures log: {failures_log}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())