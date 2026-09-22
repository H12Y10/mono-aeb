"""vendored ByteTrack 可用性：确认**不依赖 Cython 扩展**即可运行。

这是本仓库相对上游 ByteTrack 的关键适配（上游 `setup.py` 走
`torch.utils.cpp_extension`，安装需要 C++ 编译器）：

  1. `cython_bbox` C 扩展 → 纯 numpy shim（`aeb/tracker/byte_shims/cython_bbox.py`）；
  2. numpy>=1.24 删除的 `np.float` / `np.int` / `np.bool` 别名 → 运行时补回
     （`bytetrack._patch_numpy`）；
  3. `import lap` 失败时回退 `lapx`（vendored `matching.py` 内）；
  4. 上游 `yolox/__init__.py` 会连带导入 torch → 替换为空实现。

本测试在 CI（无编译器、新 numpy）下真实驱动跟踪器，保证上述适配持续生效。
"""

from pathlib import Path

import numpy as np

from aeb.tracker import ByteTrackTracker
from aeb.tracker import bytetrack as bt
from aeb.types import Detection


def test_vendored_root_resolves_inside_package():
    """vendored 根目录必须落在包内 —— 这样 editable 与常规安装都能解析到。"""
    root = Path(bt._BYTETRACK_ROOT)
    assert root.is_dir(), f"vendored ByteTrack 根目录不存在: {root}"
    assert "vendor" in root.parts and "bytetrack" in root.parts, \
        f"vendored 根目录位置异常: {root}"


def test_cython_bbox_uses_numpy_shim_not_extension():
    """应加载纯 numpy shim，而不是需要编译的 Cython 扩展。"""
    ByteTrackTracker()  # 触发 _ensure_importable()，把 byte_shims 注入 sys.path

    import cython_bbox

    impl = Path(cython_bbox.__file__).as_posix()
    assert "byte_shims" in impl, f"应使用纯 numpy shim，实际加载的是 {impl}"


def test_numpy_aliases_are_restored():
    """numpy>=1.24 移除了 np.float/np.int/np.bool，上游 ByteTrack 仍在使用它们。"""
    bt._patch_numpy()
    for name in ("float", "int", "bool"):
        assert hasattr(np, name), f"np.{name} 应被补回内置类型"


def test_tracks_are_produced_across_frames():
    """连续帧驱动：匹配（lap/lapx）+ 卡尔曼滤波 + 轨迹生命周期全部真实跑通。"""
    trk = ByteTrackTracker(track_thresh=0.5, fps=30.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = []
    for i in range(10):
        x = 600.0 + 2.0 * i
        tracks = trk.update([Detection(x, 400.0, x + 60.0, 460.0, 0.9, 2)], frame)

    assert tracks, "连续 10 帧同类别检测应产出被跟踪目标"
    assert tracks[0].class_id == 2, f"类别应为 2(car)，实际 {tracks[0].class_id}"
