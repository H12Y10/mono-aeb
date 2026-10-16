"""读写图像：绕开 OpenCV 在非 ASCII（中文）路径下的静默失败。

实测（Windows + OpenCV）：`cv2.imwrite` 收到含非 ASCII 字符的路径时**静默失败**——
返回 False、不落盘、也不抛异常；`cv2.imread` 在同一条件下同样打不开（返回 None）。
而 ASCII 路径（含相对路径）一切正常。项目/数据目录带中文是很常见的部署形态，且这种
失败**没有任何报错**，是最难排查的一类。

因此统一改走：写 → `cv2.imencode` + `Path.write_bytes`；读 → `np.fromfile` +
`cv2.imdecode`——编码器/解码器不变，**路径交给 Python 处理**。

注意：`cv2.VideoCapture` / `cv2.VideoWriter` 走 FFmpeg 自己的路径处理，**不受**此问题
影响，无需替换（实测亦然）。
"""

from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread_unicode", "imwrite_unicode"]


def imread_unicode(path):
    """读图像，支持任意（含中文）路径。打不开返回 None（与 `cv2.imread` 行为一致）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, ext: str | None = None) -> bool:
    """写图像到任意（含中文）路径，返回是否**真的落盘**。

    扩展名默认取路径后缀（无后缀则用 `.png`）。**调用方应检查返回值**——本函数存在的
    前提正是"失败是静默的"，不能写完就当成功。
    """
    path = Path(path)
    suffix = ext or path.suffix or ".png"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        return False
    try:
        path.write_bytes(buf.tobytes())
    except OSError:
        return False
    return True
