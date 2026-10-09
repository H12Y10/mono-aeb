"""零数据合成跟踪基准回归：干净场景 ID 稳定、召回接近 1；短暂遮挡后能接回。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aeb.tracker.bytetrack import ByteTrackTracker
from benchmarks import synth_tracking as st


def test_clean_tracking_id_stable_and_recall():
    """干净检测（无噪声、无遮挡）下：3 目标各得稳定 id、无 ID 切换、召回≥0.95。"""
    targets = st.make_targets()
    frames = list(st.gen_frames(targets, n_frames=90, noise=0.0, seed=0))
    tracker = ByteTrackTracker(fps=30.0)
    m = st.run_benchmark(tracker, frames)
    assert m["gt_targets"] == 3
    assert m["id_switches"] == 0, f"干净场景不该有 ID 切换，实际 {m['id_switches']}"
    assert m["recall"] >= 0.95, f"召回应≥0.95，实际 {m['recall']:.3f}"


def test_occlusion_recovers_after_gap():
    """目标 0 消失 10 帧（< track_buffer=30）后应接回同一 id，无 ID 切换。"""
    targets = st.make_targets()
    targets[0].occlude = (30, 40)
    frames = list(st.gen_frames(targets, n_frames=90, noise=0.0, seed=0))
    tracker = ByteTrackTracker(fps=30.0)
    m = st.run_benchmark(tracker, frames)
    assert m["id_switches"] == 0, f"短暂遮挡后应接回同一 id，实际 {m['id_switches']}"
    assert m["recall"] >= 0.85, f"短暂遮挡后召回应较高，实际 {m['recall']:.3f}"
