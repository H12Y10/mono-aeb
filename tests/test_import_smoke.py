"""导入链 / ROI 派生 / ByteTrack 封装冒烟测试（不跑检测，无需权重与视频）。
"""

import numpy as np

from aeb.config import AEBConfig
from aeb.in_path import ego_path_polygon
from aeb.tracker import ByteTrackTracker


def test_roi_polygon_derives_from_config():
    """固定梯形 ROI 应由 CameraConfig + EgoPathConfig 推导出 4 个顶点。"""
    cfg = AEBConfig()
    poly = ego_path_polygon(cfg.camera, cfg.ego_path)
    assert poly.shape == (4, 2), f"ROI 应为 4 个顶点 (4,2)，实际 {poly.shape}"


def test_bytetrack_tracker_instantiates_and_updates():
    """分库跟踪器应为 7 类各建一个，且空检测也能正常推进（轨迹老化）。"""
    trk = ByteTrackTracker()
    assert len(trk._trackers) == 7, "7 类目标应各有一个 tracker（类别一致性靠分库保证）"

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert trk.update([], frame) == [], "无检测时不应产出轨迹"
