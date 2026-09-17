"""4 状态风险状态机，带迟滞避免抖动。

    NORMAL(0) → ATTENTION(1) → FCW(2) → AEB_WARNING(3)

升级（危险度上升）：越过阈值立即进入。
降级（危险度下降）：需 TTC 超过阈值 + 迟滞带才降，防边缘抖动。
"""

import numpy as np

from ..config import RiskThresholds
from ..types import RiskFeature, RiskLevel


class RiskStateMachine:
    def __init__(self, thresholds: RiskThresholds, hysteresis: float = 0.3,
                 ego_speed_mps: float = 12.0):
        self.thr = thresholds
        self.hys = hysteresis
        self.ego_speed = ego_speed_mps
        self._level = {}  # track_id -> RiskLevel

    def set_ego_speed(self, v_mps: float):
        """更新自车速度，阈值梯随之平移（demo 逐帧/逐视频更新）。"""
        self.ego_speed = v_mps

    def _ladder(self) -> tuple[float, float, float]:
        # (ttc_attention, ttc_fcw, ttc_aeb)
        return self.thr.ladder(self.ego_speed)

    def _raw_level(self, ttc: float) -> RiskLevel:
        if not np.isfinite(ttc):
            return RiskLevel.NORMAL
        att, fcw, aeb = self._ladder()
        if ttc < aeb:
            return RiskLevel.AEB_WARNING
        if ttc < fcw:
            return RiskLevel.FCW
        if ttc < att:
            return RiskLevel.ATTENTION
        return RiskLevel.NORMAL

    def _down_threshold(self, level: RiskLevel) -> float:
        # 从 level 降到下一级，TTC 需大于该阈值（含迟滞）
        att, fcw, aeb = self._ladder()
        if level == RiskLevel.AEB_WARNING:
            return aeb + self.hys
        if level == RiskLevel.FCW:
            return fcw + self.hys
        if level == RiskLevel.ATTENTION:
            return att + self.hys
        return 0.0

    def decide(self, risk: RiskFeature) -> RiskLevel:
        ttc = risk.ttc_fused
        raw = self._raw_level(ttc)
        prev = self._level.get(risk.track_id, RiskLevel.NORMAL)

        if raw >= prev:
            new = raw  # 上升或持平 → 立即响应
        else:
            # 下降 → 需明显越过阈值才降级（迟滞）
            new = raw if ttc > self._down_threshold(prev) else prev

        self._level[risk.track_id] = new
        return new
