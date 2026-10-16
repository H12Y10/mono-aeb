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
        (y[j] - y[i]) / (x[j] - x[i]) for i in range(n) for j in range(i + 1, n) if x[j] != x[i]
    ]
    if not slopes:
        return 0.0
    return float(np.median(slopes))


# 可靠性门限：双源共用的"信噪比"判据——序列的**总跨度**（信号）必须显著超过
# **逐帧差的中位绝对差**（噪声）。目标远/小/静止时，真实运动淹没在测量抖动里，
# 此时任何斜率估计都是噪声（实测：~20px 远框把 23s 真值算成 2.3s；远处单目距离
# 把 17s 算成 1.7s）。用**相对**判据（span vs 逐帧噪声）而非绝对阈值，才不会误杀
# 近距真险情（真接近时 span≫噪声，信噪比很高）。
MIN_SAMPLES = 4  # 少于 4 帧的斜率是纯噪声
SNR_K = 4.0  # 信号须 ≥ 4× 逐帧噪声


def motion_snr_ok(series: np.ndarray, k: float = SNR_K) -> bool:
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    if series.size < MIN_SAMPLES:
        return False
    span = float(series.max() - series.min())
    d = np.diff(series)
    noise = float(np.median(np.abs(d - np.median(d)))) if d.size else 0.0
    return span >= k * noise if noise > 0 else True


class ScaleTTC(TTCEstimator):
    def estimate(self, track_id: int, history: TrackHistory) -> float:
        seq = history.get(track_id)
        if len(seq) < MIN_SAMPLES:
            return float("inf")
        ts = np.array([f / self.fps for f, _, _ in seq], dtype=np.float64)
        hs = np.array([h for _, _, h in seq], dtype=np.float64)
        valid = hs > 0
        if valid.sum() < MIN_SAMPLES:
            return float("inf")
        ts, hs = ts[valid], hs[valid]
        if not motion_snr_ok(1.0 / hs):  # 生长信号未超测量噪声 → 斜率不可信
            return float("inf")

        b = theil_sen_slope(ts, 1.0 / hs)  # 1/h 对 t 的斜率，接近时 b < 0
        h_now = float(np.median(hs))  # 中位数，抗末帧离群
        if b >= 0 or h_now <= 0:
            return float("inf")  # 未接近 / 静止 / 远离
        return float(np.clip(-1.0 / (b * h_now), TTC_MIN, TTC_MAX))
