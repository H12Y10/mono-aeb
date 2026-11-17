"""自车速度：插值语义与「合法零」不被当成缺失。

这两个行为都直接影响阈值梯：把 0 当缺失回退成 12，会把"车停着"算成"12 m/s"，
阈值整条上移；插值端点处理错，则首尾帧的速度是错的。
"""

import numpy as np

from aeb.ego_speed import ego_speed_at, load_ego_speed_series


def test_interpolates_linearly_between_samples():
    ts = [0.0, 1.0, 2.0]
    speeds = [10.0, 20.0, 30.0]
    assert ego_speed_at(ts, speeds, 0.5) == 15.0
    assert ego_speed_at(ts, speeds, 1.5) == 25.0


def test_clamps_at_both_ends():
    ts = [0.0, 1.0, 2.0]
    speeds = [10.0, 20.0, 30.0]
    assert ego_speed_at(ts, speeds, -5.0) == 10.0
    assert ego_speed_at(ts, speeds, 99.0) == 30.0


def test_missing_series_returns_default():
    assert ego_speed_at(None, None, 1.0, default=7.5) == 7.5
    assert ego_speed_at([], [], 1.0, default=7.5) == 7.5


def test_legal_zero_is_kept_not_treated_as_missing():
    ts = [0.0, 1.0, 2.0]
    speeds = [0.0, 0.0, 0.0]
    assert ego_speed_at(ts, speeds, 1.0) == 0.0


def test_series_loader_returns_none_when_no_info(tmp_path):
    """没有伴随 GPS 数据的视频：明确返回 None，而不是伪造一串速度。"""
    assert load_ego_speed_series(tmp_path / "nope.mov") == (None, None)


def test_numpy_inputs_are_accepted():
    ts = np.array([0.0, 2.0])
    speeds = np.array([4.0, 8.0])
    assert ego_speed_at(ts, speeds, 1.0) == 6.0
