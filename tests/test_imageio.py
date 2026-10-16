"""unicode 图像读/写封装：中文路径往返（零数据、无需外部文件）。

存在前提：`cv2.imread` / `cv2.imwrite` 在非 ASCII 路径下会**静默失败**（分别返回
None / False，都不抛异常），而项目目录带中文是常见部署形态。这里断言封装在中文
路径下能真正落盘并读回，且失败时按约定返回 None/False。
"""

import numpy as np

from aeb.imageio import imread_unicode, imwrite_unicode


def test_roundtrip_in_non_ascii_path(tmp_path):
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # 绿色通道给个可识别值

    d = tmp_path / "中文目录"
    d.mkdir()
    p = d / "图_带中文.png"

    assert imwrite_unicode(p, img) is True
    assert p.exists() and p.stat().st_size > 0

    back = imread_unicode(p)
    assert back is not None
    assert back.shape == img.shape
    assert int(back[0, 0, 1]) == 200


def test_imread_unicode_missing_returns_none(tmp_path):
    assert imread_unicode(tmp_path / "不存在.png") is None


def test_imwrite_unicode_defaults_to_png(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "无后缀"
    assert imwrite_unicode(p, img) is True
    assert imread_unicode(p) is not None
