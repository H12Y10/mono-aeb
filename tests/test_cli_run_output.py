"""`run` 的结构化输出、摘要、帧数上限与异常收尾。

对应 0.3.2 的四项使用改进：
  1. `--jsonl`：逐帧一行 JSON，字段名与智能驾驶演示平台 CSV 的前 11 列一致；
  2. 结束摘要：帧数、耗时、处理帧率、四档等级计数；
  3. `--max-frames`：只处理前 N 帧；
  4. 异常时仍关闭 JSONL / 视频写入器（不留半截产物）。

不加载模型：检测器被替换为固定返回一个框的假实现；输入是 3 张合成 PNG。
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import aeb.cli as cli
from aeb.types import Detection, RiskLevel

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KEYS = [
    "frame",
    "global_level",
    "n_tracks",
    "n_in_path",
    "min_ttc_fused",
    "min_ttc_distance",
    "min_ttc_scale",
    "min_distance_m",
    "min_closing_speed",
    "n_fcw",
    "n_aeb",
]


class _FakeDetector:
    """固定返回一个居中偏下的车框；真的走 pipeline 的跟踪/测距/TTC/决策。"""

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        h, w = frame.shape[:2]
        return [Detection(w * 0.42, h * 0.55, w * 0.58, h * 0.90, 0.9, 2)]


class _BoomDetector(_FakeDetector):
    def detect(self, frame):
        raise RuntimeError("detector exploded")


def _write_images(directory: Path, n: int = 3, size=(240, 320)) -> None:
    import cv2

    h, w = size
    for i in range(n):
        cv2.imwrite(str(directory / f"f{i:02d}.png"), np.full((h, w, 3), 70, np.uint8))


@pytest.fixture()
def image_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    _write_images(d)
    return d


def _run(monkeypatch, argv):
    monkeypatch.setattr("aeb.detectors.YOLOv8Detector", _FakeDetector)
    return cli.main(argv)


def test_frame_record_has_platform_column_names():
    class _R:
        track_id = 1
        in_path = True
        distance = float("inf")
        closing_speed = 3.0
        ttc_distance = float("inf")
        ttc_scale = float("inf")
        ttc_fused = float("inf")
        level = RiskLevel.NORMAL

    class _Result:
        frame_idx = 7
        tracks = (object(),)
        risks = (_R(),)
        global_level = RiskLevel.NORMAL

    rec = cli.frame_record(_Result())
    assert list(rec) == EXPECTED_KEYS
    assert rec["frame"] == 7
    assert rec["global_level"] == "NORMAL"
    assert rec["n_in_path"] == 1
    # inf 必须变成 null，JSON 里不能出现 Infinity
    assert rec["min_ttc_fused"] is None
    assert rec["min_distance_m"] is None
    assert json.dumps(rec, allow_nan=False)  # 序列化不抛 = 没有 nan/inf
    assert rec["min_closing_speed"] == 3.0


def test_jsonl_written_one_object_per_frame(monkeypatch, image_dir, tmp_path, capsys):
    out = tmp_path / "rows.jsonl"
    rc = _run(
        monkeypatch,
        ["run", "--source", str(image_dir), "--fps", "30", "--jsonl", str(out)],
    )
    assert rc == 0

    with open(out, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    assert [r["frame"] for r in rows] == [0, 1, 2]
    assert all(list(r) == EXPECTED_KEYS for r in rows)

    stdout = capsys.readouterr().out
    assert "summary:" in stdout
    assert "processing" in stdout
    assert "NORMAL" in stdout


def test_max_frames_limits_processing(monkeypatch, image_dir, tmp_path):
    out = tmp_path / "rows.jsonl"
    rc = _run(
        monkeypatch,
        [
            "run",
            "--source",
            str(image_dir),
            "--fps",
            "30",
            "--jsonl",
            str(out),
            "--max-frames",
            "2",
        ],
    )
    assert rc == 0
    with open(out, encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == 2


def test_jsonl_is_flushed_when_detector_raises(monkeypatch, image_dir, tmp_path):
    """异常传播出去，但 JSONL 必须已关闭——否则半截文件读不出来。"""
    out = tmp_path / "rows.jsonl"
    monkeypatch.setattr("aeb.detectors.YOLOv8Detector", _BoomDetector)
    with pytest.raises(RuntimeError, match="detector exploded"):
        cli.main(["run", "--source", str(image_dir), "--fps", "30", "--jsonl", str(out)])
    # 已关闭：能重新打开并逐行读取（且第一帧在处理前就抛了，行数为 0 或 1 都合法）
    assert out.exists()
    with open(out, encoding="utf-8") as fh:
        fh.read()


def test_run_still_writes_video(tmp_path, monkeypatch, image_dir):
    out = tmp_path / "out.mp4"
    rc = _run(
        monkeypatch,
        ["run", "--source", str(image_dir), "--fps", "30", "--out", str(out)],
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------------------ examples 薄壳


def _load_example(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["quickstart", "demo_aeb", "calibrate_video"])
def test_example_help_works(name, monkeypatch):
    mod = _load_example(name)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0


def test_demo_aeb_forwards_new_flags(monkeypatch):
    mod = _load_example("demo_aeb")
    seen = {}
    monkeypatch.setattr(mod, "cli_main", lambda argv: (seen.__setitem__("argv", argv), 0)[1])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_aeb.py",
            "--source",
            "clip.mp4",
            "--jsonl",
            "rows.jsonl",
            "--max-frames",
            "5",
            "--calib",
            "calib.json",
        ],
    )
    assert mod.main() == 0
    argv = seen["argv"]
    assert argv[:2] == ["run", "--source"]
    assert argv[argv.index("--jsonl") + 1] == "rows.jsonl"
    assert argv[argv.index("--max-frames") + 1] == "5"
    assert argv[argv.index("--calib") + 1] == "calib.json"


def test_jsonl_columns_match_platform_prefix(tmp_path):
    """字段名必须与智能驾驶 CSV 前 11 列同名同序——两套口径不能分叉。"""
    golden = (
        ROOT.parent.parent
        / "AIC智能驾驶参赛项目"
        / "05-演示平台"
        / "_stage"
        / "p3"
        / "rows_20261026_clear.csv"
    )
    if not golden.exists():  # 跨项目路径不在时跳过，不伪造断言
        pytest.skip("智驾平台黄金 CSV 不在本机，跳过跨项目列名比对")
    with open(golden, encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh))
    assert header[:11] == EXPECTED_KEYS
