#!/usr/bin/env python3
"""
Run:
  pip install flask
  python app.py

Endpoint:
  POST /process
  Body (JSON): { "video_path": "/abs/path/to/video.mp4" }

Returns:
  JSON array of strings: ["000123.png", "000456.png", ...]
"""

from __future__ import annotations

import os
import re
import json
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Any

from flask import Flask, request, jsonify, Response

app = Flask(__name__)


# --- repo / script locations ---

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
RUN_PY    = SRC_DIR / "run.py"
V2F_PY    = SRC_DIR / "video_to_frames.py"
BASE_CFG  = REPO_ROOT / "config.json"


# --- utilities ---

def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_abs_file(p: str) -> bool:
    try:
        q = Path(p)
        return q.is_absolute() and q.exists() and q.is_file()
    except Exception:
        return False


def _basename_list(xs: List[str]) -> List[str]:
    return [os.path.basename(x) for x in xs]


def _subproc(cmd: List[str], cwd: Path | None = None, timeout: int | None = None) -> None:
    """
    Run a subprocess with error translation to HTTP 500.
    """
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=timeout
        )
        # Optional: log completed.stdout for debugging
    except subprocess.CalledProcessError as e:
        msg = f"Command failed: {' '.join(cmd)}\nSTDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}"
        raise RuntimeError(msg)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timeout: {' '.join(cmd)} (timeout={timeout}s)")


def _names_to_indices(names: List[str]) -> List[int]:
    idxs: List[int] = []
    for n in names:
        stem = Path(n).stem  # e.g., "000123" from "000123.png"
        m = re.search(r'(\d+)$', stem)
        if not m:
            raise RuntimeError(f"Cannot parse frame index from name: {n}")
        d = m.group(1)
        idxs.append(int(d.lstrip('0') or '0'))  # "000123" -> 123
    return idxs


# --- core pipeline ---

def _process_video(video_path: str) -> List[str]:
    """
    Returns list of selected frame NAMES (no directories).
    """
    if not _is_abs_file(video_path):
        raise ValueError("`video_path` must be an existing absolute file path")

    if not RUN_PY.exists():
        raise RuntimeError(f"run.py not found at {RUN_PY}")
    if not V2F_PY.exists():
        raise RuntimeError(f"video_to_frames.py not found at {V2F_PY}")
    if not BASE_CFG.exists():
        raise RuntimeError(f"Base config.json not found at {BASE_CFG}")

    # Per-request temp working area
    req_id = uuid.uuid4().hex[:8]
    tmp_root = Path(tempfile.mkdtemp(prefix=f"psfr_{req_id}_"))
    frames_dir = tmp_root / "frames"
    output_dir = tmp_root / "output"
    reports_dir = output_dir / "reports"
    tmp_cfg = tmp_root / "config.json"

    try:
        # 1) Extract frames
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Be permissive about arguments supported by your utility.
        # If your script uses flags like `--overwrite`, include it; otherwise harmless.
        v2f_cmd = [
            os.sys.executable, str(V2F_PY),
            str(video_path), str(frames_dir),
            "--max-fps", "4",
            "--ext", "png",
            "--overwrite"
        ]
        _subproc(v2f_cmd, cwd=REPO_ROOT, timeout=60 * 15)  # 15 min safety

        # 2) Build temporary config
        base_cfg = _read_json(BASE_CFG)

        # Make sure the necessary blocks exist
        base_cfg.setdefault("paths", {})
        base_cfg.setdefault("selection", {})
        base_cfg.setdefault("visualization", {})

        base_cfg["paths"]["frames_dir"] = str(frames_dir)
        base_cfg["paths"]["output_dir"] = str(output_dir)

        # Force non-copy and non-viz for API processing (faster/leaner)
        base_cfg["selection"]["copy_keyframes"] = False
        base_cfg["visualization"]["enabled"] = False

        _write_json(tmp_cfg, base_cfg)

        # 3) Run main stage
        # This branch expects: python src/run.py <path-to-config.json>
        run_cmd = [os.sys.executable, str(RUN_PY), str(tmp_cfg)]
        _subproc(run_cmd, cwd=REPO_ROOT, timeout=60 * 30)  # 30 min safety

        # 4) Read results
        keyjson = reports_dir / "keyframes.json"
        keytxt  = reports_dir / "keyframes.txt"

        if keyjson.exists():
            data = _read_json(keyjson)
            frames = data.get("keyframes", [])
            if not isinstance(frames, list):
                raise RuntimeError("Malformed keyframes.json: 'keyframes' is not a list")
            return _names_to_indices(frames)

        if keytxt.exists():
            names = [ln.strip() for ln in keytxt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return _names_to_indices(names)

        raise RuntimeError("No keyframes.json or keyframes.txt produced")

    finally:
        # Cleanup temp data to keep the service lean.
        # Comment out the next line if you want to keep artifacts for debugging.
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass


# --- HTTP endpoints ---

@app.post("/process")
def process() -> Response:
    """
    POST JSON: { "video_path": "/abs/path/to/video.mp4" }
    Returns JSON array of frame names: ["000100.png", ...]
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object expected"}), 400

        video_path = payload.get("video_path")
        if not isinstance(video_path, str):
            return jsonify({"error": "`video_path` (string) is required"}), 400

        frames = _process_video(video_path)
        return jsonify(frames), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        # Return captured stderr/stdout in error message (safe: no secrets here)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"internal error: {e.__class__.__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    # threaded=True allows multiple concurrent requests in the built-in server
    # For production, use a WSGI server (gunicorn, uwsgi, etc.).
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)