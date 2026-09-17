"""自车路径筛选抽象接口。"""

from abc import ABC, abstractmethod

from ..types import Track


class InPathFilter(ABC):
    @abstractmethod
    def in_path(self, track: Track) -> bool:
        """判断目标是否在自车前方走廊内（排除相邻车道误报）。"""
        ...
