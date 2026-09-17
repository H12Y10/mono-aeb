"""TTC 源 2（免标定主干）：bbox 高度膨胀率（looming / 尺度 TTC）。

TTC_scale = h / ḣ 完全不需要相机内参/外参——BDD100K 众包采集、相机逐视频不同，
尺度 TTC 天然免疫，故作为碰撞时间的主干来源；TTC_distance 仅在标定可靠时作交叉校验。

实现要点（稳健 + 无偏）：
  - 匀速接近时 h(t) = C/(D0−v·t) 是双曲，直接对 h 做直线拟合会系统高估 TTC；
    但 1/h(t) = (D0−v·t)/C 是 t 的线性函数，故对 (t, 1/h) 做线性拟合无偏。
    取斜率 b（接近时 b<0），TTC = −1/(b·h_now)。
  - 斜率用 Theil-Sen 估计，对单帧 bbox 抖动/离群不敏感；
  - h_now 取高度序列中位数，避免末帧离群值直接污染结果。
"""

import numpy as np

from .base import TTCEstimator
from .history import TrackHistory

# bbox 高度抖动会让 h_dot 噪声很大，进而 TTC_scale = h/h_dot 出现
# 0.01s 这种物理上不可能的极小值（AEB 早已触发）。限幅到 [TTC_MIN, TTC_MAX]。
TTC_MIN = 0.1
TTC_MAX = 100.0


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen 斜率估计（对 ~50% 离群不敏感）。"""
    n = len(x)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n)
        for j in range(i + 1, n)
        if x[j] != x[i]
    ]
    if not slopes:
        return 0.0
    return float(np.median(slopes))


class ScaleTTC(TTCEstimator):
    def estimate(self, track_id: int, history: TrackHistory) -> float:
        seq = history.get(track_id)
        if len(seq) < 2:
            return float("inf")
        ts = np.array([f / self.fps for f, _, _ in seq], dtype=np.float64)
        hs = np.array([h for _, _, h in seq], dtype=np.float64)
        valid = hs > 0
        if valid.sum() < 2:
            return float("inf")
        ts, hs = ts[valid], hs[valid]

        b = theil_sen_slope(ts, 1.0 / hs)  # 1/h 对 t 的斜率，接近时 b < 0
        h_now = float(np.median(hs))        # 中位数，抗末帧离群
        if b >= 0 or h_now <= 0:
            return float("inf")  # 未接近 / 静止 / 远离
        return float(np.clip(-1.0 / (b * h_now), TTC_MIN, TTC_MAX))
