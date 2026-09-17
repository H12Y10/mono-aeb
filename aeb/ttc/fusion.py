"""TTC 融合：取双源更危险者 + K 帧持续（时序滤波）。

- 融合值 = min(ttc_distance, ttc_scale)：哪个源判断更危险用哪个。
- 当融合值突然跳回 inf（测距/尺度短时失败）时，用上一有效值保持 K 帧，
  避免目标仍在却被瞬时判成 NORMAL。
"""

import numpy as np

# 融合值物理限幅：低于 TTC_MIN 的异常极小值或高于 TTC_MAX 的异常极大值都不可信，
# 在融合出口统一截断，防止任一源离群值直接灌进状态机。
TTC_MIN = 0.1
TTC_MAX = 100.0


class TTCFusion:
    def __init__(self, persist_k: int = 3):
        self.persist_k = persist_k
        self._last = {}  # track_id -> (frame_idx, fused_ttc)

    def fuse(self, track_id: int, frame_idx: int,
             ttc_distance: float, ttc_scale: float) -> float:
        ttc = min(ttc_distance, ttc_scale)

        if np.isfinite(ttc):
            # 双源给出真实有限值：限幅后作为新基准，刷新时间戳
            ttc = float(np.clip(ttc, TTC_MIN, TTC_MAX))
            self._last[track_id] = (frame_idx, ttc)
            return ttc

        # 双源都失效（目标远离 / 自车制动 / 短时漏检）：用上一有效值保持，
        # 最多 persist_k 帧。时间戳必须沿用原始 pframe、不得随回填刷新，
        # 否则 frame_idx-pframe 恒为 1、窗口永不关闭，危险 TTC 会被无限期保持
        # （→ 目标远离后持续假 AEB/FCW）。
        prev = self._last.get(track_id)
        if prev is not None:
            pframe, pttc = prev
            if frame_idx - pframe <= self.persist_k:
                return pttc
        return ttc
