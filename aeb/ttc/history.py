"""按 track_id 维护的时序缓冲，供 TTC 双源估计共用。"""

from collections import deque


class TrackHistory:
    """每条轨迹保存 (frame_idx, distance, bbox_height) 的最近 N 个观测。"""

    def __init__(self, max_len: int = 10):
        self.max_len = max_len
        self._bufs = {}  # track_id -> deque[(frame_idx, distance, height)]

    def push(self, track_id: int, frame_idx: int, distance: float, height: float) -> None:
        buf = self._bufs.setdefault(track_id, deque(maxlen=self.max_len))
        buf.append((frame_idx, distance, height))

    def get(self, track_id: int) -> list:
        return list(self._bufs.get(track_id, []))

    def prune(self, active_ids: set) -> None:
        """清除不再出现的轨迹（长时间丢失后不参与决策）。"""
        for tid in list(self._bufs.keys()):
            if tid not in active_ids:
                del self._bufs[tid]
