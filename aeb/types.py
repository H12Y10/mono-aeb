"""核心数据结构：全链通用契约。"""

from dataclasses import dataclass
from enum import IntEnum

# 7 类 AEB 目标，顺序即 class_id（与 data.yaml / data_common.py 对齐）
AEB_CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "bike", "motor"]


class RiskLevel(IntEnum):
    NORMAL = 0
    ATTENTION = 1
    FCW = 2
    AEB_WARNING = 3


@dataclass
class Detection:
    """检测器无关契约：`[x1, y1, x2, y2, score, class]`。"""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def bottom_center(self) -> tuple:
        """目标底边中点（测距 / In-Path 都用它）。"""
        return (self.cx, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Track:
    """跟踪输出（含历史轨迹供 TTC_scale / 时序滤波）。"""
    track_id: int
    class_id: int
    bbox: tuple          # (x1, y1, x2, y2)
    score: float
    state: str           # tracked | new | lost
    age: int = 0

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def bottom_center(self) -> tuple:
        return (self.cx, self.bbox[3])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class RiskFeature:
    """决策输入（风险特征）+ 决策输出。"""
    track_id: int
    in_path: bool
    distance: float = float("inf")
    closing_speed: float = 0.0
    ttc_distance: float = float("inf")
    ttc_scale: float = float("inf")
    ttc_fused: float = float("inf")
    required_deceleration: float = 0.0
    detection_confidence: float = 0.0
    track_age: int = 0
    level: RiskLevel = RiskLevel.NORMAL  # 决策输出（状态机回填）


@dataclass
class FrameResult:
    """单帧完整结果，供可视化 / 评测。"""
    frame_idx: int
    detections: list
    tracks: list
    risks: list
    global_level: RiskLevel
