"""地面平面测距。

平坦地面假设：目标底部落在自车前方地面上，纵向距离
    D = f_px * H / (y_bottom - y_horizon)
其中 f_px 为焦距(像素)、H 为相机高度(m)、y_horizon 为地平线像素行。
"""

import numpy as np

from .base import DistanceEstimator
from ..config import CameraConfig
from ..types import Track


class GroundPlaneDistance(DistanceEstimator):
    def __init__(self, camera: CameraConfig,
                 max_distance: float = 150.0, min_distance: float = 1.0):
        self.fh = camera.focal_height  # 测距尺度 f·H（f、H 尺度简并，只取乘积）
        self.hy = camera.horizon_y
        self.d_max = max_distance
        self.d_min = min_distance

    def estimate(self, track: Track, frame: np.ndarray) -> float:
        y_bottom = track.bbox[3]  # y2
        if y_bottom <= self.hy:
            # 底部在地平线及其上方 → 投影无解，视为不可测
            return float("inf")
        d = self.fh / (y_bottom - self.hy)
        if d > self.d_max:
            # 太远 / 底部贴地平线：分母趋零，数值不可靠，视为不可测
            return float("inf")
        return max(d, self.d_min)
