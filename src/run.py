# standart imports
import sys
import json
import pathlib

BASE_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(BASE_DIR / "src"))

# other imports
from constants import FRAMES_DIR, OUTPUT_DIR, KEYFRAMES_DIR
from key_frame_selection import select_keyframes_from_frames


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))

    # paths
    p = cfg["paths"]
    frames_dir = pathlib.Path(p.get("frames_dir", FRAMES_DIR))
    out_dir = pathlib.Path(p.get("output_dir", OUTPUT_DIR))
    key_frames_dir = pathlib.Path(p.get("key_frames_dir", KEYFRAMES_DIR))
    preprocess = cfg.get("preprocess", {})

    # keyframe selection on frames
    report = select_keyframes_from_frames(
        frames_dir=frames_dir,
        out_dir=out_dir / "reports",
        key_frames_dir=key_frames_dir,
        patching=cfg["patching"],
        shi=cfg["shi_tomasi"],
        lk=cfg["lucas_kanade"],
        selection=cfg["selection"],
        preprocess=preprocess,
        visualization=cfg.get("visualization", {})
    )

    print(f"Keyframes: {len(report['keyframes'])} saved to {out_dir/'reports'}/keyframes.*")


if __name__ == "__main__":
    main()