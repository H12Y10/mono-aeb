"""Per-video calibration config: save what ``calib`` estimated, load it in ``run``.

`mono-aeb calib` 只在终端打印 f·H 与地平线，并把结果写进**当前进程**的
``CameraConfig``；进程结束就没了，``run`` 也没有读它的入口。本模块把同一份估计
落成带版本与来源的 JSON，供 ``mono-aeb run --calib`` 读回。

原则：这里只搬运 ``CameraConfig`` 已有的字段，**不重算任何数值**；写回时走既有的
``calibrate_scale()`` / ``calibrate_horizon()``，避免出现第二套标定口径。

注意 ``CameraConfig.focal_height`` 是属性（``focal_px × cam_height_m``），所以文件里
存的是分解后的 ``focal_px`` 与 ``cam_height_m``，读回时用乘积还原 f·H。
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

SCHEMA = "mono-aeb/calibration/1"

_REQUIRED_CAMERA = ("focal_px", "cam_height_m", "horizon_y")


def save_calibration(
    path: str | Path,
    camera,
    *,
    source_path: str | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    fps: float | None = None,
    focal_height: float | None = None,
    samples: int | None = None,
    tracks: int | None = None,
    version: str | None = None,
) -> Path:
    """把 ``camera`` 的标定结果写成 JSON，返回落盘路径。

    ``focal_height`` 允许显式传入估计值；省略时取 ``camera.focal_height``。
    ``source_*`` 只作溯源记录，不参与计算。
    """
    out = Path(path)
    fh = float(focal_height) if focal_height is not None else float(camera.focal_height)
    data = {
        "schema": SCHEMA,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "mono_aeb_version": version or "unknown",
        "source": {
            "path": str(source_path) if source_path else None,
            "width": int(source_width) if source_width else int(camera.img_w),
            "height": int(source_height) if source_height else int(camera.img_h),
            "fps": float(fps) if fps else None,
        },
        "camera": {
            "focal_px": float(camera.focal_px),
            "cam_height_m": float(camera.cam_height_m),
            "horizon_y": float(camera.horizon_y),
            "principal_x": float(camera.principal_x),
            "img_w": int(camera.img_w),
            "img_h": int(camera.img_h),
        },
        "estimate": {
            "focal_height": fh,
            "samples": int(samples) if samples is not None else None,
            "tracks": int(tracks) if tracks is not None else None,
        },
    }
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def load_calibration(path: str | Path) -> dict:
    """读回并校验标定 JSON；结构不对时抛 ``ValueError``（含可操作提示）。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"calibration file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration file is not valid JSON ({p}): {exc}") from exc

    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported calibration schema {schema!r} in {p}; expected {SCHEMA!r}")
    cam = data.get("camera") or {}
    missing = [k for k in _REQUIRED_CAMERA if k not in cam]
    if missing:
        raise ValueError(f"calibration file {p} is missing camera fields: {', '.join(missing)}")
    return data


def apply_calibration(camera, data: dict, *, frame_size: tuple[int, int] | None = None) -> None:
    """把标定写回 ``camera``（走既有 setter）。

    ``frame_size=(w, h)`` 给定时校验输入尺寸：f·H 以像素为单位标定，
    换分辨率后直接套用会算错距离，因此尺寸不符时抛 ``ValueError``，
    要求显式重新标定或自行换算。
    """
    cam = data["camera"]
    if frame_size is not None:
        w, h = int(frame_size[0]), int(frame_size[1])
        cw, ch = int(cam.get("img_w", w)), int(cam.get("img_h", h))
        if (w, h) != (cw, ch):
            raise ValueError(
                f"calibration was estimated at {cw}x{ch} but the input is {w}x{h}; "
                "f·H is in pixels, so re-run `mono-aeb calib` on this resolution "
                "instead of reusing the file"
            )
        camera.img_w, camera.img_h = w, h

    fh = data.get("estimate", {}).get("focal_height")
    if fh is None:
        fh = float(cam["focal_px"]) * float(cam["cam_height_m"])
    camera.calibrate_scale(float(fh))
    camera.calibrate_horizon(float(cam["horizon_y"]))
    if "principal_x" in cam:
        camera.principal_x = float(cam["principal_x"])
