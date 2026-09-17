"""AEB 前向碰撞预警链路（检测器无关）。

整条链只消费 `[x1, y1, x2, y2, score, class]`，检测器可自由替换
（YOLOv8n 默认 / D-FINE 可选）。
"""

from .types import (
    AEB_CLASS_NAMES,
    Detection,
    Track,
    RiskFeature,
    RiskLevel,
    FrameResult,
)

__all__ = [
    "AEB_CLASS_NAMES",
    "Detection",
    "Track",
    "RiskFeature",
    "RiskLevel",
    "FrameResult",
]
