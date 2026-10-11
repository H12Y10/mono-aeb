"""打包后的 CLI：`python -m aeb ...` 与进程内 `aeb.cli.main` 两套驱动。

tests/test_quickstart_demo.py 跑的是 examples/quickstart.py；这里跑的是随 wheel
分发的 `aeb/cli.py` 本体（模块形式，不经 examples/），确保“pip 安装后立刻可跑”
不依赖源码 checkout 里的示例脚本。

两种驱动各司其职：
- subprocess（`python -m aeb ...`，本文件上半）：真端到端，证明模块形式可执行；
- 进程内（直接调 `aeb.cli.main([...])`，本文件下半）：快速失败定位，且让覆盖率
  能统计到 cli.py——在 subprocess 里执行的代码，父进程的覆盖率统计看不见。
"""

import subprocess
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aeb", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_module_check_reports_dependencies():
    r = _run("check")
    assert r.returncode == 0, (
        f"python -m aeb check exited with {r.returncode}:\n{r.stdout}\n{r.stderr}"
    )
    assert "all set" in r.stdout, f"check did not report core dependencies as ready:\n{r.stdout}"


def test_module_demo_walks_the_full_risk_ladder():
    r = _run("demo")
    out = r.stdout
    assert r.returncode == 0, f"python -m aeb demo exited with {r.returncode}:\n{out}\n{r.stderr}"
    for token in ("NORMAL", "ATTENTION", "FCW", "AEB"):
        assert token in out, f"demo output is missing level {token}:\n{out}"
    assert "[quickstart] done" in out, f"demo did not finish normally:\n{out}"


def test_module_version_exits_zero():
    r = _run("--version")
    assert r.returncode == 0, (
        f"python -m aeb --version exited with {r.returncode}:\n{r.stdout}\n{r.stderr}"
    )
    assert r.stdout.strip(), "python -m aeb --version printed nothing"


# ---------------------------------------------------------------------------
# 进程内驱动：同一批入口直接调 aeb.cli.main，让覆盖率统计得到 cli.py。
# subprocess 版仍然是真端到端证据；这里保住的是“覆盖率口径 + 快速回归定位”。
# ---------------------------------------------------------------------------


def test_inproc_version_returns_zero(capsys):
    from aeb.cli import main

    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    # 纯源码 checkout（未安装）时版本退化为 unknown，故只断言前缀。
    assert out.startswith("mono-aeb "), f"unexpected --version output: {out!r}"


def test_inproc_check_reports_dependencies(capsys):
    from aeb.cli import main

    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "core dependencies" in out and "all set" in out, out
    assert "detector dependencies" in out, out


def test_inproc_demo_walks_the_full_risk_ladder(capsys):
    from aeb.cli import main

    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    for token in ("NORMAL", "ATTENTION", "FCW", "AEB"):
        assert token in out, f"demo output is missing level {token}:\n{out}"
    assert "[quickstart] done, 120 frames" in out, out


def test_inproc_no_command_prints_help(capsys):
    from aeb.cli import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out, out
    for token in ("demo", "check", "run", "calib"):
        assert token in out, f"help is missing subcommand {token}:\n{out}"


def test_inproc_demo_writes_video(tmp_path, capsys):
    from aeb.cli import main

    out_path = tmp_path / "demo.mp4"
    probe = cv2.VideoWriter(
        str(tmp_path / "probe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64)
    )
    if not probe.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v video")
    probe.release()

    assert main(["demo", "--frames", "3", "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert f"video written to {out_path}" in out, out
    assert out_path.exists() and out_path.stat().st_size > 0, "demo --out wrote no video"
