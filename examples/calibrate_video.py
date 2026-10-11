"""Per-video offline self-calibration CLI: GPS ego speed + scale TTC -> ranging scale f·H.

Usage:
    python examples/calibrate_video.py --stem 0571873b-faf718b2
    python examples/calibrate_video.py --video /path/to/samples-1k/videos/xxxx.mov --max-frames 200

Runs one pass over the video and inverse-solves f·H = v_ego·τ·(y_bottom − horizon) from
approximately stationary targets, taking the median as this video's ranging scale and
comparing it against the placeholder default 1100 x 1.5 = 1650.

This is a thin wrapper kept for source checkouts: the implementation lives in
`aeb/cli.py` and ships in the wheel, so after `pip install mono-aeb` the same
command is `mono-aeb calib ...` (or `python -m aeb calib ...`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeb.cli import main as cli_main


def main() -> int:
    ap = argparse.ArgumentParser(description="per-video offline self-calibration of f·H")
    ap.add_argument("--video", type=str, default=None, help="full path to the video")
    ap.add_argument("--stem", type=str, default=None, help="clip stem under samples-1k/videos")
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    argv = ["calib", "--weights", args.weights, "--conf", str(args.conf), "--fps", str(args.fps)]
    if args.video:
        argv += ["--video", args.video]
    if args.stem:
        argv += ["--stem", args.stem]
    if args.max_frames is not None:
        argv += ["--max-frames", str(args.max_frames)]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
