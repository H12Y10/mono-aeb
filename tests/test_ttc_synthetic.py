"""合成接近场景：验证测距恢复 + TTC 双源 + 状态机升级链（无需外部数据/权重）。

移植自主项目 `02-代码/test_ttc_synthetic.py`，断言与阈值保持一致。

模拟正前方一辆车以 v=20 m/s 匀速接近，bbox 按地面平面模型增长，喂入完整
pipeline，检查：
  1) 测距能恢复真实距离 D；
  2) TTC 与状态机随距离缩短单调升级；
  3) 风险等级覆盖 NORMAL → ATTENTION → FCW → AEB_WARNING 四级。
"""

import numpy as np

from aeb.config import AEBConfig
from aeb.pipeline import AEBPipeline
from aeb.types import Detection, RiskLevel


class FakeApproachDetector:
    """返回正前方一个匀速接近的 car 检测框，box 随距离按地面模型变化。"""

    def __init__(self, d0: float = 100.0, v: float = 20.0, fps: float = 30.0):
        self.d0, self.v, self.fps = d0, v, fps
        self.i = 0

    def detect(self, frame):
        d = self.d0 - self.v * (self.i / self.fps)
        self.i += 1
        if d <= 1.0:
            return []
        # 地面模型：y_bottom = horizon + f·H/D，框高 h = f·H/D（目标与相机同高）
        f, cam_h, horizon, cx = 1100.0, 1.5, 360.0, 640.0
        y_bottom = horizon + f * cam_h / d
        h = f * cam_h / d
        w = 1.2 * h
        return [Detection(cx - w / 2, y_bottom - h, cx + w / 2, y_bottom, 0.9, 2)]


def test_distance_ttc_and_state_machine_ladder():
    cfg = AEBConfig()
    d0, v, fps = 100.0, 20.0, cfg.fps
    det = FakeApproachDetector(d0=d0, v=v, fps=fps)
    pipe = AEBPipeline(cfg, det)
    frame = np.zeros((720, 1280, 3), np.uint8)

    seen_levels = []
    dist_rel_err = []
    for i in range(150):
        res = pipe.process(frame, i)
        d_true = max(d0 - v * (i / fps), 0.0)
        if not res.risks:
            continue
        r = res.risks[0]
        seen_levels.append(r.level)
        # 测距恢复精度：只在 5~80 m 有效区间内统计
        if 5.0 < d_true < 80.0 and np.isfinite(r.distance):
            dist_rel_err.append(abs(r.distance - d_true) / d_true)
        if i % 15 == 0:
            ttc_true = d_true / v if d_true > 0 else 0.0
            print(f"t={i / fps:.1f}s D_true={d_true:5.1f}  d_est={r.distance:5.1f}  "
                  f"ttc={r.ttc_fused:4.2f}(真{ttc_true:4.2f})  "
                  f"vc={r.closing_speed:4.1f}  {r.level.name}")

    # 等级变化顺序（去重）
    order = []
    for lvl in seen_levels:
        if not order or lvl != order[-1]:
            order.append(lvl)
    print("等级升级序列:", " -> ".join(l.name for l in order))

    assert order[0] == RiskLevel.NORMAL, "应从 NORMAL 起步"
    assert RiskLevel.AEB_WARNING in seen_levels, "应最终进入 AEB_WARNING"

    # 升级段（到第一次 AEB 为止）应单调上升、且覆盖全部四级；
    # 末尾回落到 NORMAL 是目标距离 <1 m 检测消失导致的，属正常
    up = order[: order.index(RiskLevel.AEB_WARNING) + 1]
    assert up == sorted(up, key=int), f"升级段应单调，实际 {[l.name for l in up]}"
    assert set(up) == {RiskLevel.NORMAL, RiskLevel.ATTENTION,
                       RiskLevel.FCW, RiskLevel.AEB_WARNING}, "升级段应覆盖四级"

    # 可核验的效果数据：测距中位相对误差
    assert dist_rel_err, "应采到有效测距样本"
    median_err = float(np.median(dist_rel_err))
    print(f"测距中位相对误差 = {median_err:.1%}（n={len(dist_rel_err)}）")
    assert median_err < 0.10, f"测距中位相对误差过大: {median_err:.1%}"
