"""检测器抽象接口。"""

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """输入 BGR 帧，输出检测列表（xyxy + score + class）。"""
        ...
