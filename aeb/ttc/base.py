"""TTC 估计抽象接口。"""

from abc import ABC, abstractmethod

from .history import TrackHistory


class TTCEstimator(ABC):
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    @abstractmethod
    def estimate(self, track_id: int, history: TrackHistory) -> float:
        """返回碰撞时间 TTC（秒）；无法估计返回 +inf。"""
        ...
