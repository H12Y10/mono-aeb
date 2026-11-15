"""标定配置往返 / 帧率来源 / 入口失败反馈。

对应 0.3.1 的四个使用断点：
  1. `calib` 只在终端打印，`run` 拿不到 —— 现在可 `--out-config` 落盘、`--calib` 读回；
  2. `run` 的帧率写死 30 —— 现在按「显式 --fps > 源视频 FPS > 30」解析并打印来源；
  3. 图片目录没有固有帧率 —— 必须显式给，否则退出；
  4. 标定失败返回 0、缺依赖时 `check` 直接崩 —— 现在都给出退出码。

本文件不加载模型、不读真实视频、不依赖网络。
"""

import json

import pytest

from aeb.calib_io import SCHEMA, apply_calibration, load_calibration, save_calibration
from aeb.cli import _pick_fps
from aeb.config import CameraConfig

# --------------------------------------------------------------- 配置往返


def test_calibration_roundtrip_restores_focal_height_and_horizon(tmp_path):
    cam = CameraConfig()
    cam.calibrate_scale(960.0)  # f·H = 960
    cam.calibrate_horizon(372.5)
    cam.img_w, cam.img_h = 1280, 720

    out = save_calibration(
        tmp_path / "calib.json",
        cam,
        source_path="clip.mp4",
        source_width=1280,
        source_height=720,
        fps=30.0,
        focal_height=960.0,
        samples=42,
        tracks=7,
        version="0.3.1",
    )
    data = load_calibration(out)

    fresh = CameraConfig()
    apply_calibration(fresh, data, frame_size=(1280, 720))

    assert data["schema"] == SCHEMA
    assert data["estimate"]["samples"] == 42
    assert fresh.focal_height == pytest.approx(960.0)
    assert fresh.horizon_y == pytest.approx(372.5)
    assert (fresh.img_w, fresh.img_h) == (1280, 720)


def test_save_creates_parent_directory(tmp_path):
    out = save_calibration(tmp_path / "nested" / "dir" / "calib.json", CameraConfig())
    assert out.exists()


def test_load_rejects_unknown_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "something/else", "camera": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_calibration(p)


def test_load_rejects_missing_camera_fields(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": SCHEMA, "camera": {"focal_px": 640.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing camera fields"):
        load_calibration(p)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_calibration(tmp_path / "nope.json")


def test_apply_rejects_resolution_mismatch(tmp_path):
    """f·H 以像素为单位；换分辨率直接套用会算错距离，必须明确失败。"""
    cam = CameraConfig()
    cam.img_w, cam.img_h = 1280, 720
    out = save_calibration(tmp_path / "calib.json", cam, focal_height=960.0)
    data = load_calibration(out)

    with pytest.raises(ValueError, match="1280x720"):
        apply_calibration(CameraConfig(), data, frame_size=(1920, 1080))


# --------------------------------------------------------------- 帧率来源


def test_pick_fps_prefers_explicit_argument():
    fps, src = _pick_fps(25.0, 30.0, is_dir=False)
    assert fps == 25.0
    assert "command line" in src


def test_pick_fps_uses_source_video_rate():
    fps, src = _pick_fps(None, 24.0, is_dir=False)
    assert fps == 24.0
    assert "CAP_PROP_FPS" in src


def test_pick_fps_falls_back_to_30_when_source_reports_nothing():
    fps, src = _pick_fps(None, 0.0, is_dir=False)
    assert fps == 30.0
    assert "default" in src


def test_pick_fps_rejects_image_directory_without_explicit_rate():
    with pytest.raises(SystemExit, match="image directory"):
        _pick_fps(None, None, is_dir=True)


def test_pick_fps_rejects_non_positive_argument():
    with pytest.raises(SystemExit, match="positive"):
        _pick_fps(0.0, 30.0, is_dir=False)


def test_probe_source_reads_image_directory_resolution(tmp_path):
    """图片目录没有帧率但有分辨率；标定校验依赖它，不能返回 None。"""
    import cv2
    import numpy as np

    from aeb.cli import _probe_source

    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((240, 320, 3), np.uint8))
    is_dir, fps, size = _probe_source(str(tmp_path))
    assert is_dir is True
    assert fps is None
    assert size == (320, 240)


# --------------------------------------------------------------- 入口反馈


def test_check_reports_import_failure_instead_of_crashing(monkeypatch):
    """缺依赖时 check 要给人话 + 退出码 1，而不是 import 阶段 traceback。"""
    import aeb.cli as cli

    hint = "numpy is missing (pip install numpy)"
    monkeypatch.setattr(cli, "core_import_error", lambda: hint)
    monkeypatch.setattr(cli, "missing_deps", lambda need_detector=False: [hint])
    assert cli.run_check() == 1


def test_version_flag_does_not_require_core_dependencies():
    from aeb.cli import main

    assert main(["--version"]) == 0
