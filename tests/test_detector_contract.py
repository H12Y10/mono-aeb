"""检测器契约：任何符合 `BaseDetector.detect()` 的实现都能驱动整条链。

为什么要有这组测试：文档里写「检测器无关」，但此前只有两个内置后端，
无法证明"换一个第三方检测器也能跑"。这里用 examples 里的零依赖实现做正向验证，
再用一个恒返回空列表的检测器验证"无目标帧"不会炸。
"""

import importlib.util
from pathlib import Path

import numpy as np

from aeb.config import AEBConfig
from aeb.detectors import BaseDetector
from aeb.pipeline import AEBPipeline
from aeb.types import RiskLevel

ROOT = Path(__file__).resolve().parents[1]


def _load_example():
    path = ROOT / "examples" / "custom_detector.py"
    spec = importlib.util.spec_from_file_location("_example_custom_detector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _EmptyDetector(BaseDetector):
    def detect(self, frame):
        return []


def test_example_detector_satisfies_the_interface():
    mod = _load_example()
    det = mod.BrightBlobDetector()
    assert isinstance(det, BaseDetector)
    frame = np.full((240, 320, 3), 40, np.uint8)
    out = det.detect(frame)
    assert isinstance(out, list)  # 允许空列表


def test_custom_detector_climbs_all_four_levels():
    """真实链路（跟踪/测距/TTC/状态机）跑在一个非 YOLO、非 D-FINE 的检测器上。"""
    mod = _load_example()
    cfg = AEBConfig(fps=30.0)
    cfg.ego_speed_mps = 12.0
    pipe = AEBPipeline(cfg, mod.BrightBlobDetector())

    levels = []
    for i, frame in enumerate(mod.synthetic_frames(150, 30.0)):
        levels.append(pipe.process(frame, i).global_level)

    distinct = [lv for i, lv in enumerate(levels) if i == 0 or lv != levels[i - 1]]
    assert distinct[:4] == [
        RiskLevel.NORMAL,
        RiskLevel.ATTENTION,
        RiskLevel.FCW,
        RiskLevel.AEB_WARNING,
    ], [lv.name for lv in distinct]


def test_empty_detections_keep_the_chain_normal():
    cfg = AEBConfig(fps=30.0, ego_speed_mps=12.0)
    pipe = AEBPipeline(cfg, _EmptyDetector())
    frame = np.zeros((240, 320, 3), np.uint8)
    for i in range(30):
        assert pipe.process(frame, i).global_level is RiskLevel.NORMAL
