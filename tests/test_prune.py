"""过期状态清理：保持项与轨迹等级不能无限增长，也不能删早了改变决策。

`TTCFusion._last` 与 `RiskStateMachine._level` 都是按 track_id 的字典。
长视频里 id 会不断新增，不清理就是内存泄漏；清理过头又会把"短暂丢失后恢复"
的迟滞状态删掉，改变行为。两条边界各测一次。
"""

from aeb.config import RiskThresholds
from aeb.decision.state_machine import RiskStateMachine
from aeb.ttc.fusion import TTCFusion
from aeb.types import RiskFeature, RiskLevel


def test_fusion_drops_entries_outside_the_persist_window():
    fus = TTCFusion(persist_k=3)
    fus.fuse(1, frame_idx=0, ttc_distance=2.0, ttc_scale=float("inf"))
    assert 1 in fus._last

    fus.prune(3)  # 0 + 3 仍在窗口内
    assert 1 in fus._last

    fus.prune(4)  # 超过 persist_k
    assert 1 not in fus._last


def test_fusion_kept_value_is_still_used_before_prune():
    fus = TTCFusion(persist_k=3)
    first = fus.fuse(1, 0, 2.0, float("inf"))
    held = fus.fuse(1, 2, float("inf"), float("inf"))  # 双源失效，窗口内回填
    assert held == first


def test_state_machine_keeps_level_inside_ttl_and_drops_after():
    sm = RiskStateMachine(RiskThresholds(), ego_speed_mps=12.0)
    aeb = RiskFeature(track_id=7, in_path=True, ttc_fused=1.0, distance=8.0)
    assert sm.decide(aeb, frame_idx=0) is RiskLevel.AEB_WARNING

    sm.prune(20)  # ttl 默认 30，仍在窗口内
    assert sm._level[7] is RiskLevel.AEB_WARNING

    sm.prune(31)
    assert 7 not in sm._level
    assert 7 not in sm._last_seen


def test_reappearing_track_after_ttl_starts_from_normal():
    """id 过期后被复用：不能继承上一段的危险等级。"""
    sm = RiskStateMachine(RiskThresholds(), ego_speed_mps=12.0)
    hot = RiskFeature(track_id=9, in_path=True, ttc_fused=1.0, distance=8.0)
    sm.decide(hot, frame_idx=0)
    sm.prune(40)  # 超过 ttl → 状态清空

    calm = RiskFeature(track_id=9, in_path=True, ttc_fused=float("inf"), distance=50.0)
    assert sm.decide(calm, frame_idx=41) is RiskLevel.NORMAL
