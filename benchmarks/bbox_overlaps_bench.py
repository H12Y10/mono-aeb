"""bbox_overlaps_bench.py — performance cost of the pure-numpy `bbox_overlaps` shim.

Why it exists: the compiled `cython_bbox` extension needs an MSVC build on Windows and
often cannot be installed, so this repository substitutes a pure-numpy implementation
(`aeb/tracker/byte_shims/cython_bbox.py`) for ByteTrack's `bbox_overlaps`, keeping
"no compiler, plain pip install". This benchmark quantifies that trade-off.

Method (no re-implementation: it imports the exact function the production code uses):
  1) for N in {10, 100, 1000, 5000} (N x N pairwise IoU), take the fastest of several
     runs, measuring the pure-numpy shim;
  2) if the compiled `cython_bbox` is importable in this environment, compare at the
     same sizes and report the speed-up; if not, say so plainly ("it needs an MSVC
     build -- which is exactly why the shim exists") rather than inventing a number;
  3) end-to-end view: the cost of one IoU matching call at a realistic scale
     (detections x tracks ~= 30 x 30), against the 33 ms per-frame budget at 30 FPS.

Scope: synthetic random boxes, not a real detection distribution; the report is the
measured timing at these synthetic sizes only, with no extrapolation.

Usage:
    python benchmarks/bbox_overlaps_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeb.tracker.byte_shims.cython_bbox import bbox_overlaps


def _rand_boxes(n, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1000, n)
    y1 = rng.uniform(0, 700, n)
    w = rng.uniform(20, 200, n)
    h = rng.uniform(20, 200, n)
    return np.column_stack([x1, y1, x1 + w, y1 + h])


def _timeit(fn, boxes, query, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(boxes, query)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    sizes = [10, 100, 1000, 5000]
    print("pure-numpy `bbox_overlaps` shim timing (N x N pairwise IoU, fastest of 5 runs)")
    print(f"{'N':>6} {'shim (ms)':>14}")

    for n in sizes:
        boxes = _rand_boxes(n, seed=0)
        query = _rand_boxes(n, seed=1)
        t = _timeit(bbox_overlaps, boxes, query) * 1000
        print(f"{n:>6} {t:>14.3f}")

    # 编译版 cython_bbox 若可装则同规模对比
    try:
        import cython_bbox as _cb

        cb = _cb.bbox_overlaps
        print("\ncompiled cython_bbox is importable here; comparison:")
        print(f"{'N':>6} {'numpy shim (ms)':>16} {'cython (ms)':>12} {'speedup':>8}")
        for n in sizes:
            boxes = _rand_boxes(n, seed=0)
            query = _rand_boxes(n, seed=1)
            t_shim = _timeit(bbox_overlaps, boxes, query) * 1000
            t_cb = _timeit(cb, boxes, query) * 1000
            print(f"{n:>6} {t_shim:>14.3f} {t_cb:>12.3f} {t_shim / t_cb:>7.1f}x")
    except ImportError:
        print("\ncompiled cython_bbox is not installed here (it needs an MSVC build) --")
        print("which is exactly why the pure-numpy shim exists. No direct comparison;")
        print("only the end-to-end share below.")

    # 端到端视角：真实规模（检测框 × 跟踪框 ≈ 30×30）
    n_real = 30
    boxes = _rand_boxes(n_real, seed=0)
    query = _rand_boxes(n_real, seed=1)
    t_real = _timeit(bbox_overlaps, boxes, query) * 1000
    frame_budget = 1000 / 30.0  # 33.3 ms @ 30 FPS
    share = t_real / frame_budget * 100
    print(f"\nend-to-end: one IoU matching call at {n_real}x{n_real} (realistic scale)")
    print(
        f"  {t_real:.3f} ms = {share:.2f}% of the {frame_budget:.1f} ms per-frame budget at 30 FPS"
    )


if __name__ == "__main__":
    main()
