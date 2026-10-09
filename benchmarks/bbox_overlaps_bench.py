"""bbox_overlaps_bench.py — 纯 numpy `bbox_overlaps` 替身的性能代价基准（A3）。

为什么存在：cython_bbox 在 Windows 上需 MSVC 编译、常装不上，故仓库用纯 numpy
替身（aeb/tracker/byte_shims/cython_bbox.py）顶替 ByteTrack 的 `bbox_overlaps`，
换来「免编译、pip 直装」。但「免编译」的代价是性能——本基准把这个代价量化出来，
回答评审「纯 numpy 替身慢多少」的问题。

测法（零重写数值逻辑，直接 import 生产代码用的那个函数）：
  1) 对 N ∈ {10, 100, 1000, 5000}（框数，N×N 两两 IoU），各跑多轮取最快，测纯 numpy shim；
  2) 若本机可 import 真 `cython_bbox`（编译版），同规模对比并报加速比；不可装则如实说明
     「需 MSVC 编译，这正是用 shim 的原因」，不臆造对比；
  3) 端到端视角：报「真实规模」（检测框 × 跟踪框 ≈ 30×30）下单次 IoU 匹配的耗时，
     对照 30 FPS 每帧 33 ms 预算，给出诚实结论（shim 的代价占帧预算的比例）。

口径：合成随机框、无真实检测分布，结论只写「合成规模下的计时」，不做外推。

用法：
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
    print("纯 numpy `bbox_overlaps` 替身 计时（N×N 两两 IoU，取 5 轮最快）")
    print(f"{'N':>6} {'shim 耗时(ms)':>14}")

    for n in sizes:
        boxes = _rand_boxes(n, seed=0)
        query = _rand_boxes(n, seed=1)
        t = _timeit(bbox_overlaps, boxes, query) * 1000
        print(f"{n:>6} {t:>14.3f}")

    # 编译版 cython_bbox 若可装则同规模对比
    try:
        import cython_bbox as _cb

        cb = _cb.bbox_overlaps
        print("\n本机已装编译版 cython_bbox，对比：")
        print(f"{'N':>6} {'numpy shim(ms)':>14} {'cython(ms)':>12} {'加速比':>8}")
        for n in sizes:
            boxes = _rand_boxes(n, seed=0)
            query = _rand_boxes(n, seed=1)
            t_shim = _timeit(bbox_overlaps, boxes, query) * 1000
            t_cb = _timeit(cb, boxes, query) * 1000
            print(f"{n:>6} {t_shim:>14.3f} {t_cb:>12.3f} {t_shim / t_cb:>7.1f}x")
    except ImportError:
        print("\n本机未装编译版 cython_bbox（需 MSVC 编译）——这正是仓库用纯 numpy 替身的原因。")
        print("故无直接对比；下面只报端到端占比。")

    # 端到端视角：真实规模（检测框 × 跟踪框 ≈ 30×30）
    n_real = 30
    boxes = _rand_boxes(n_real, seed=0)
    query = _rand_boxes(n_real, seed=1)
    t_real = _timeit(bbox_overlaps, boxes, query) * 1000
    frame_budget = 1000 / 30.0  # 33.3 ms @ 30 FPS
    print(f"\n端到端：真实规模 {n_real}×{n_real} 单次 IoU 匹配 = {t_real:.3f} ms")
    print(f"占 30 FPS 每帧预算（{frame_budget:.1f} ms）的 {t_real / frame_budget * 100:.2f}%")


if __name__ == "__main__":
    main()
