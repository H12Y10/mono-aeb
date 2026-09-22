"""合成静止目标：验证联合标定能恢复 horizon_y（非默认 360）。

horizon_y 是决策相关量（在 ttc_distance 里不约掉），故严格断言；
f·H 是辅助量（TTC 决策里约掉、只影响显示，且受 ByteTrack Kalman 平滑偏差），
只做宽松 ballpark 校验（f·H 对远目标小 denom 的地平线误差极敏感）。
"""

import numpy as np
import pytest

from aeb.calibration import estimate_ground_plane
from aeb.config import CameraConfig
from aeb.types import Detection

FPS = 30.0
D0, V, TRUE_FH = 100.0, 20.0, 1650.0


class FakeStaticDetector:
    """正前方一个静止目标，自车以 v 接近，bbox 按地面模型增长（horizon 可自定义）。"""

    def __init__(self, horizon: float, d0: float = D0, v: float = V,
                 fps: float = FPS, focal_height: float = TRUE_FH):
        self.horizon = horizon
        self.d0, self.v, self.fps = d0, v, fps
        self.fh = focal_height
        self.i = 0

    def detect(self, frame):
        d = self.d0 - self.v * (self.i / self.fps)
        self.i += 1
        if d <= 3.0:
            return []
        y_bottom = self.horizon + self.fh / d
        h = self.fh / d          # 目标高设为与相机高同尺度，仅为 bbox 合理
        w = 1.2 * h
        cx = 640.0
        return [Detection(cx - w / 2, y_bottom - h, cx + w / 2, y_bottom, 0.9, 2)]


@pytest.mark.parametrize("horizon", [400.0, 320.0, 360.0])
def test_estimate_ground_plane_recovers_horizon(horizon):
    det = FakeStaticDetector(horizon)

    def frames():
        for i in range(150):
            yield i, np.zeros((720, 1280, 3), np.uint8)

    camera = CameraConfig()  # 默认 horizon=360
    fh, hy, n_samples, n_tracks = estimate_ground_plane(
        det, frames(), camera, lambda i: 20.0, fps=FPS,
    )
    print(f"真值 f·H={TRUE_FH} horizon={horizon}  →  "
          f"恢复 f·H={fh if fh is None else round(fh, 1)} "
          f"horizon={hy if hy is None else round(hy, 1)}  "
          f"采样={n_samples} 轨迹={n_tracks}")

    assert fh is not None and hy is not None, "应能联合标定出 f·H 与 horizon"
    # horizon：决策相关量，严格断言（旧默认 360 在真值 400/320 时差 40px）
    assert abs(hy - horizon) < 8.0, f"horizon 偏差过大: {hy:.1f} vs {horizon}"
    # f·H：辅助量，只做 ballpark（Kalman 平滑 + 小 denom 放大）
    assert 0.5 * TRUE_FH < fh < 1.6 * TRUE_FH, f"f·H 明显异常: {fh:.1f}"
