"""跟踪器抽象接口。"""

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection, Track


class BaseTracker(ABC):
    @abstractmethod
    def update(self, dets: list[Detection], frame: np.ndarray) -> list[Track]:
        """输入当前帧检测，输出跟踪轨迹（含 track_id / age）。"""
        ...
