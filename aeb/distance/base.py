"""距离估计抽象接口。"""

from abc import ABC, abstractmethod

import numpy as np

from ..types import Track


class DistanceEstimator(ABC):
    @abstractmethod
    def estimate(self, track: Track, frame: np.ndarray) -> float:
        """返回目标纵向距离（米）；不可测时返回 +inf。"""
        ...
