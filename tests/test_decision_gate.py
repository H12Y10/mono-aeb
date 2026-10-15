"""物理先验闸回归测试（`RiskStateMachine.decide` 的 TTC 自洽性门限）。

闸规则：同向 in-path 目标，`distance / ttc_fused` 隐含的接近速度不得显著超过
自车速度（+CLOSING_MARGIN=5）。超过即判为远处弱信号被噪声压出的假值 → 不升级。
"""

from aeb.config import RiskThresholds
from aeb.decision import RiskStateMachine
from aeb.types import RiskFeature, RiskLevel


def test_gate_rejects_inconsistent_far_ttc():
    """远处弱信号假值：D=25m、ttc=0.5s → 隐含接近 50 m/s，远超自车速 12+5 → 判 NORMAL。"""
    sm = RiskStateMachine(RiskThresholds(), ego_speed_mps=12.0)
    r = RiskFeature(track_id=1, in_path=True, distance=25.0, closing_speed=50.0,
                    ttc_fused=0.5)
    assert sm.decide(r) == RiskLevel.NORMAL


def test_gate_keeps_consistent_near_ttc():
    """自洽真险情：D=8m、ttc=0.9s → 隐含 8.9 m/s ≤ 12+5 → 照常升级到 AEB_WARNING。"""
    sm = RiskStateMachine(RiskThresholds(), ego_speed_mps=12.0)
    r = RiskFeature(track_id=2, in_path=True, distance=8.0, closing_speed=8.9,
                    ttc_fused=0.9)
    assert sm.decide(r) == RiskLevel.AEB_WARNING


def test_gate_inactive_when_distance_unknown():
    """距离未知（inf）时闸不生效，行为与原来一致（按 TTC 梯升级）。"""
    sm = RiskStateMachine(RiskThresholds(), ego_speed_mps=12.0)
    r = RiskFeature(track_id=3, in_path=True, distance=float("inf"), ttc_fused=0.9)
    assert sm.decide(r) == RiskLevel.AEB_WARNING
