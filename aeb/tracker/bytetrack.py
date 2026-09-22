"""ByteTrack 封装：按类别各建一个 tracker。

依赖 ByteTrack（MIT License，只 import 其 byte_tracker/basetrack；默认用包内
vendor 的 `aeb/tracker/vendor/bytetrack`，随 wheel 一起分发，editable 与常规安装
行为一致；可用 BYTETRACK_ROOT 环境变量覆盖为外部 ByteTrack 仓库），
并用本地 numpy shim 顶替 cython_bbox。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .base import BaseTracker
from ..types import Detection, Track

# ByteTrack 根目录：优先取 BYTETRACK_ROOT 环境变量，否则回退到包内 vendor 的最小子集
# （aeb/tracker/vendor/bytetrack，MIT License）。cython_bbox 由本地 numpy shim 顶替。
#   BYTETRACK_ROOT  ByteTrack 仓库根目录（内含 yolox/tracker/），可覆盖包内 vendor 默认
# 注意：默认路径基于 __file__ 同级的 vendor/，因此 editable 安装（指向源码树）与常规
# 安装（指向 site-packages/aeb/tracker/）都能解析到，不再依赖仓库根目录层级。
_BYTETRACK_ROOT = os.environ.get("BYTETRACK_ROOT") or (
    Path(__file__).resolve().parent / "vendor" / "bytetrack")
_SHIM_DIR = Path(__file__).resolve().parent / "byte_shims"


def _patch_numpy():
    """numpy>=1.24 删除了 np.float/np.int/np.bool 等别名，旧版 ByteTrack 依赖它们。

    在 import yolox 之前把这些别名补回内置类型（dtype=float/int/bool 均合法）。
    """
    import numpy as np
    # 仅补 ByteTrack 实际用到的三个已删别名（np.float/int/bool），
    # 不碰 np.object/np.str 等仍存在但已弃用的，避免 FutureWarning
    aliases = {"float": float, "int": int, "bool": bool}
    for name, val in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, val)


def _ensure_importable():
    """把 shim + ByteTrack 路径注入 sys.path，再返回所需符号。"""
    _patch_numpy()
    root = Path(_BYTETRACK_ROOT)
    if not root.exists():
        raise RuntimeError(
            f"ByteTrack 源码不可用（{root} 不存在）：包内 vendor 副本缺失时请重装本包；"
            "若设置了 BYTETRACK_ROOT，请确认它指向 ByteTrack 仓库根目录（内含 yolox/tracker/）")
    for p in (_SHIM_DIR, root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import TrackState
    return BYTETracker, TrackState


class ByteTrackTracker(BaseTracker):
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, fps: float = 30.0):
        BYTETracker, _ = _ensure_importable()
        args = argparse.Namespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=False,
        )
        # 7 类各一个 tracker：类别一致性靠「分库」保证，car 不会关联成 person
        self._trackers = {c: BYTETracker(args, frame_rate=fps) for c in range(7)}

    def update(self, dets: list[Detection], frame: np.ndarray) -> list[Track]:
        _, TrackState = _ensure_importable()
        img_h, img_w = frame.shape[:2]
        # 检测坐标已在原图空间 → img_size 与 img_info 一致，scale=1 不做重缩放
        img_info = (img_h, img_w)
        img_size = (img_h, img_w)

        tracks = []
        for c in range(7):
            group = [d for d in dets if d.class_id == c]
            if group:
                arr = np.array([[d.x1, d.y1, d.x2, d.y2, d.score] for d in group],
                               dtype=np.float32)
            else:
                # 无该类检测也要推进 update，让丢失轨迹正确老化
                arr = np.zeros((0, 5), dtype=np.float32)

            online = self._trackers[c].update(arr, img_info, img_size)
            for st in online:
                if st.state != TrackState.Tracked:
                    continue
                x1, y1, x2, y2 = st.tlbr.tolist()
                tracks.append(Track(
                    track_id=int(st.track_id),
                    class_id=c,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(st.score),
                    state="tracked",
                    age=int(st.frame_id - st.start_frame),
                ))
        return tracks
