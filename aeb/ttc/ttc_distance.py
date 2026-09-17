"""TTC 源 1：纵向距离微分。

对最近 N 帧的距离序列做稳健一阶拟合（Theil-Sen，与尺度源一致），
接近速度 v_close = -斜率，TTC = D_now / v_close。
"""

import numpy as np

from .base import TTCEstimator
from .history import TrackHistory
from .ttc_scale import theil_sen_slope

# 接近速度物理上限（m/s）。即便稳健拟合，距离序列也可能被极端离群拉出
# 几百 m/s 的假斜率（远超真实道路接近速度）。超过该值视为不可信，直接判
# 不可测（返回 inf / 0），让融合只用尺度源，而不是拿截断值硬算。
MAX_CLOSING_SPEED = 40.0


class DistanceTTC(TTCEstimator):
    def _fit(self, track_id: int, history: TrackHistory):
        seq = history.get(track_id)
        if len(seq) < 2:
            return None
        ts = np.array([f / self.fps for f, _, _ in seq], dtype=np.float64)
        ds = np.array([d for _, d, _ in seq], dtype=np.float64)
        valid = np.isfinite(ds)
        if valid.sum() < 2:
            return None
        ts, ds = ts[valid], ds[valid]

        a = theil_sen_slope(ts, ds)   # D = a·t + b，接近时 a < 0
        v_close = float(-a)
        if v_close > MAX_CLOSING_SPEED:
            return None               # 超出物理上限，判不可信
        d_now = float(np.median(ds))  # 中位数，抗末帧离群
        return v_close, d_now

    def estimate(self, track_id: int, history: TrackHistory) -> float:
        fit = self._fit(track_id, history)
        if fit is None:
            return float("inf")
        v_close, d_now = fit
        if v_close <= 1e-3 or d_now <= 0:
            return float("inf")  # 未接近 / 距离无效
        return d_now / v_close

    def closing_speed(self, track_id: int, history: TrackHistory) -> float:
        """供 a_req 使用的接近速度（m/s），不可测返回 0。"""
        fit = self._fit(track_id, history)
        if fit is None:
            return 0.0
        v_close, _ = fit
        return float(v_close) if v_close > 0 else 0.0
