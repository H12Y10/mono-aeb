"""零数据合成跟踪基准：已知真值轨迹喂 ByteTrack，量化 ID 稳定性与召回。

为什么存在（诚实补缺口）：真实视频的跟踪评测（HOTA/MOTA/IDF1）需要 BDD100K MOT
跟踪真值（box_track），受「仅学术/非商业、禁止再分发」许可限制、且本机不一定持有，
无法随仓库做零数据回归。本基准用「已知 GT 轨迹的合成多目标」验证跟踪器接线正确、
可回归，覆盖三件事：
  - 类别分库是否生效（car / person / rider 不会互相串 id）
  - 干净检测下 ID 是否稳定（ID switch = 0）
  - 目标短暂丢失后能否被 track_buffer 接回（可选遮挡场景）

⚠️ 口径：合成场景理想、无真实遮挡噪声，结论只能写「合成场景下的可回归基准」，
不得外推为真实跟踪精度。真实指标见 benchmarks/tracking_eval.py（需 BDD100K MOT GT）。

用法：
    python benchmarks/synth_tracking.py                 # 干净场景
    python benchmarks/synth_tracking.py --occlude 10    # 目标 0 消失 10 帧再出现
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 允许 `python benchmarks/synth_tracking.py` 直接从仓库根目录外运行（独立脚本）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeb.tracker.bytetrack import ByteTrackTracker
from aeb.types import Detection


@dataclass
class _Target:
    """一条合成目标：gt_id、类别、初始框、每帧速度、(消失起止帧 or None)。"""

    gt_id: int
    class_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    velocity: tuple  # (dx, dy) 每帧位移
    occlude: tuple | None  # (start, end) 左闭右开区间内消失


def make_targets() -> list[_Target]:
    """3 个互不重叠、匀速直线运动的目标（car / person / rider，覆盖三种类别分库）。"""
    return [
        _Target(0, 2, (300.0, 360.0, 380.0, 420.0), (2.0, 0.5), None),  # car
        _Target(1, 0, (880.0, 260.0, 920.0, 340.0), (-1.5, 0.3), None),  # person
        _Target(2, 1, (580.0, 140.0, 630.0, 230.0), (1.8, 0.2), None),  # rider
    ]


def gen_frames(targets, n_frames, noise=0.0, seed=0):
    """逐帧产出 (list[Detection], list[(gt_id, class_id, bbox)])，噪声为框坐标高斯抖动。"""
    rng = np.random.default_rng(seed)
    for f in range(n_frames):
        dets: list[Detection] = []
        gts: list[tuple] = []
        for t in targets:
            if t.occlude and t.occlude[0] <= f < t.occlude[1]:
                continue  # 该帧目标消失
            x1, y1, x2, y2 = t.bbox
            x1 += t.velocity[0] * f
            x2 += t.velocity[0] * f
            y1 += t.velocity[1] * f
            y2 += t.velocity[1] * f
            if noise > 0:
                x1 += rng.normal(0, noise)
                y1 += rng.normal(0, noise)
                x2 += rng.normal(0, noise)
                y2 += rng.normal(0, noise)
            bbox = (float(x1), float(y1), float(x2), float(y2))
            dets.append(
                Detection(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], score=0.9, class_id=t.class_id
                )
            )
            gts.append((t.gt_id, t.class_id, bbox))
        yield dets, gts


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def match_tracks(tracks, gts, max_dist=40.0):
    """把 tracked 输出按中心距离就近匹配到 GT（同类别），返回 {gt_id: track_id}。

    类别不一致视为不匹配——若跟踪器发生跨类串 id（分库失效），此处在 recall 上显形。
    """
    matched: dict[int, int] = {}
    used: set[int] = set()
    for gt_id, cls, bbox in gts:
        gc = _center(bbox)
        best, best_d = None, max_dist
        for tr in tracks:
            if tr.track_id in used or tr.class_id != cls:
                continue
            c = _center(tr.bbox)
            d = ((c[0] - gc[0]) ** 2 + (c[1] - gc[1]) ** 2) ** 0.5
            if d < best_d:
                best, best_d = tr.track_id, d
        if best is not None:
            matched[gt_id] = best
            used.add(best)
    return matched


def run_benchmark(tracker, frames_list, warmup=10):
    """跑完所有帧，统计每个 GT 目标的 track_id 序列 → ID switch / 召回。"""
    id_seq: dict[int, list] = {}
    for dets, gts in frames_list:
        frame = np.zeros((720, 1280, 3), np.uint8)
        tracks = tracker.update(dets, frame)
        m = match_tracks(tracks, gts)
        for gt_id, _, _ in gts:
            id_seq.setdefault(gt_id, []).append(m.get(gt_id))

    n_switch = 0
    n_miss = 0
    n_pairs = 0
    for seq in id_seq.values():
        seq = seq[warmup:]  # 跳过 warmup（ByteTrack 确认轨迹前的帧不计）
        if not seq:
            continue
        n_pairs += len(seq)
        n_miss += sum(1 for s in seq if s is None)
        prev = None
        for s in seq:
            if s is None:
                continue
            if prev is not None and s != prev:
                n_switch += 1
            prev = s
    recall = 1.0 - n_miss / n_pairs if n_pairs else float("nan")
    return {
        "id_switches": n_switch,
        "recall": recall,
        "gt_targets": len(id_seq),
        "eval_pairs": n_pairs,
    }


def main():
    ap = argparse.ArgumentParser(description="零数据合成跟踪基准")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--occlude", type=int, default=0, help="目标 0 消失帧数（测 track_buffer）")
    args = ap.parse_args()

    targets = make_targets()
    if args.occlude > 0:
        targets[0].occlude = (30, 30 + args.occlude)

    frames = list(gen_frames(targets, args.frames, noise=args.noise))
    tracker = ByteTrackTracker(fps=30.0)
    m = run_benchmark(tracker, frames)
    print("合成跟踪基准（零数据，不可外推为真实跟踪精度）")
    print(f"  目标数={m['gt_targets']}  帧={args.frames}  噪声={args.noise}  遮挡={args.occlude}")
    print(
        f"  ID switch = {m['id_switches']}   召回 = {m['recall']:.3f}  （评估对 {m['eval_pairs']}）"
    )


if __name__ == "__main__":
    main()
