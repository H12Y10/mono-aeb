"""检测器后端：BaseDetector + YOLOv8 + D-FINE。

默认只导出 BaseDetector（无重依赖）。YOLOv8Detector / DFineDetector / COCO_TO_AEB
按需懒加载：`from aeb.detectors import YOLOv8Detector` 仍可用，但只有真正访问时才
import ultralytics / torch，保证 `--demo`（只用 BaseDetector）不拉入重依赖。
"""

from .base import BaseDetector

__all__ = ["BaseDetector", "YOLOv8Detector", "DFineDetector", "COCO_TO_AEB"]

_LAZY = {
    "YOLOv8Detector": (".yolov8_detector", "YOLOv8Detector"),
    "COCO_TO_AEB": (".yolov8_detector", "COCO_TO_AEB"),
    "DFineDetector": (".dfine_detector", "DFineDetector"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod_name, __name__), attr)
        globals()[name] = value  # 缓存，避免重复 import
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
