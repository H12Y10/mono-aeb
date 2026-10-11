"""Offline AEB chain demo: video / image sequence -> detect+track+range+TTC+decide -> visualize.

Usage:
    python examples/demo_aeb.py --source video.mp4 --weights yolov8n.pt --out out.mp4
    python examples/demo_aeb.py --source ./frames_dir --fps 30 --show

This is a thin wrapper kept for source checkouts: the implementation lives in
`aeb/cli.py` and ships in the wheel, so after `pip install mono-aeb` the same
command is `mono-aeb run --source video.mp4 ...` (or `python -m aeb run ...`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeb.cli import main as cli_main


def main() -> int:
    ap = argparse.ArgumentParser(description="offline AEB chain demo")
    ap.add_argument("--source", required=True, help="video file or image directory")
    ap.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="detection weights; the default yolov8n.pt is COCO-pretrained (use with --coco); "
        "pass AEB-finetuned weights otherwise",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--coco",
        action="store_true",
        help="weights are COCO-pretrained (e.g. yolov8n.pt): map COCO classes to the AEB ones",
    )
    ap.add_argument(
        "--detector",
        choices=["yolo", "dfine"],
        default="yolo",
        help="detector backend: yolo (default) or dfine",
    )
    ap.add_argument(
        "--dfine-weights",
        default=None,
        help="D-FINE weights path; defaults to the MONO_AEB_DFINE_WEIGHTS environment variable",
    )
    ap.add_argument(
        "--dfine-config",
        default=None,
        help="D-FINE config yaml; defaults to <DFINE_ROOT>/configs/dfine/dfine_hgnetv2_m_aeb.yml",
    )
    ap.add_argument("--out", default=None, help="output video path (optional)")
    ap.add_argument("--show", action="store_true", help="show a live window")
    ap.add_argument(
        "--ego-speed",
        type=float,
        default=None,
        help="ego speed in m/s; by default the GPS speed is read from samples-1k/info/*.json",
    )
    args = ap.parse_args()

    argv = [
        "run",
        "--source",
        args.source,
        "--weights",
        args.weights,
        "--conf",
        str(args.conf),
        "--device",
        args.device,
        "--fps",
        str(args.fps),
        "--detector",
        args.detector,
    ]
    if args.coco:
        argv += ["--coco"]
    if args.dfine_weights:
        argv += ["--dfine-weights", args.dfine_weights]
    if args.dfine_config:
        argv += ["--dfine-config", args.dfine_config]
    if args.out:
        argv += ["--out", args.out]
    if args.show:
        argv += ["--show"]
    if args.ego_speed is not None:
        argv += ["--ego-speed", str(args.ego_speed)]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
