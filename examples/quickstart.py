"""mono-aeb quickstart: dependency self-check + synthetic demo.

Usage:
    python examples/quickstart.py --demo              # synthetic demo (default)
    python examples/quickstart.py --check             # dependency self-check only
    python examples/quickstart.py --demo --out a.mp4  # write a visualization video
    python examples/quickstart.py --demo --show       # live window

This is a thin wrapper kept for source checkouts: the implementation lives in
`aeb/cli.py` and ships in the wheel, so after `pip install mono-aeb` the same
commands are `mono-aeb demo` / `mono-aeb check` (or `python -m aeb demo`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeb.cli import main as cli_main


def main() -> int:
    ap = argparse.ArgumentParser(description="mono-aeb quickstart")
    ap.add_argument("--demo", action="store_true", default=True, help="synthetic demo (default)")
    ap.add_argument("--check", action="store_true", help="dependency self-check only, no demo")
    ap.add_argument("--frames", type=int, default=120, help="number of demo frames (default 120)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--v-close", type=float, default=10.0, help="closing speed of the synthetic target (m/s)"
    )
    ap.add_argument("--out", default=None, help="output video path (optional)")
    ap.add_argument("--show", action="store_true", help="show a live window")
    args = ap.parse_args()

    if args.check:
        return cli_main(["check"])
    argv = ["demo", "--frames", str(args.frames), "--fps", str(args.fps)]
    argv += ["--v-close", str(args.v_close)]
    if args.out:
        argv += ["--out", args.out]
    if args.show:
        argv += ["--show"]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
