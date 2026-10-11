"""摩擦自适应阈值（零数据）：实测表插值/端点钳制/NaN + ladder 与固定 a_max 逐位一致。"""

import pytest

from aeb.config import RiskThresholds
from aeb.friction import A_EFF_TABLE, a_eff_for_mu


def test_table_is_ascending_and_positive():
    """表按 μ 升序、无重复节点、数值为正。"""
    mus = [m for m, _ in A_EFF_TABLE]
    assert mus == sorted(mus)
    assert len(set(mus)) == len(mus)
    assert all(a > 0.0 for _, a in A_EFF_TABLE)


@pytest.mark.parametrize("mu", [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 3.5])
def test_table_nodes_are_exact(mu):
    """节点上返回表值本身（a_max 放宽到不钳制，逐位相等）。"""
    expected = dict(A_EFF_TABLE)[mu]
    assert a_eff_for_mu(mu, 100.0) == expected


def test_interpolation_between_nodes():
    """节点之间线性插值。"""
    # 0.6→0.8 中点
    assert a_eff_for_mu(0.7, 100.0) == pytest.approx((3.566 + 4.751) / 2.0)
    # 1.2→1.5 上 1/6 处
    assert a_eff_for_mu(1.25, 100.0) == pytest.approx(5.639 + (0.05 / 0.3) * (6.376 - 5.639))


def test_clamps_outside_range_without_extrapolation():
    """μ 超出 [0.4, 3.5] 取端点值，不外推。"""
    assert a_eff_for_mu(0.1, 100.0) == a_eff_for_mu(0.4, 100.0) == 2.511
    assert a_eff_for_mu(3.5, 100.0) == 7.118
    assert a_eff_for_mu(99.0, 100.0) == 7.118


def test_capped_at_a_max():
    """插值结果钳制到 ≤ a_max。"""
    assert a_eff_for_mu(3.5, 6.0) == 6.0
    assert a_eff_for_mu(1.0, 4.0) == 4.0
    assert a_eff_for_mu(0.4, 2.0) == 2.0


def test_nan_raises():
    """NaN 显式报错，不得静默落到高附着端。"""
    with pytest.raises(ValueError):
        a_eff_for_mu(float("nan"), 6.86)


def test_monotonic_non_decreasing():
    """扫一遍 μ：a_eff 单调不减。"""
    vals = [a_eff_for_mu(0.3 + 0.05 * i, 100.0) for i in range(70)]
    assert vals == sorted(vals)


def test_ladder_without_mu_is_bit_identical():
    """friction_mu=None：ladder 与 t_react + v/(2·a_max) 逐位一致（含默认构造）。"""
    th = RiskThresholds()
    assert th.friction_mu is None
    for v in (0.0, 5.0, 12.0, 20.0, 33.3):
        aeb = th.t_react + v / (2.0 * th.a_max)
        assert th.ladder(v) == (
            aeb + th.ttc_attention_margin,
            aeb + th.ttc_fcw_margin,
            aeb,
        )


def test_ladder_with_mu_uses_table_and_is_more_conservative():
    """friction_mu=0.4（冰面附近）：阈值整体前移，且等于查表值算出的公式。"""
    v = 20.0
    off = RiskThresholds().ladder(v)[2]
    th = RiskThresholds(friction_mu=0.4)
    on = th.ladder(v)[2]
    assert on == th.t_react + v / (2.0 * 2.511)
    assert on > off


def test_ladder_high_mu_matches_off_behaviour():
    """μ=3.5 查表后仍被钳到 a_max：与 None 行为一致。"""
    v = 20.0
    assert RiskThresholds(friction_mu=3.5).ladder(v) == RiskThresholds().ladder(v)
