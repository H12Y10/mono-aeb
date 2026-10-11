"""打包后的 CLI：`python -m aeb ...` 冒烟测试（checkout 环境，cwd=仓库根）。

tests/test_quickstart_demo.py 跑的是 examples/quickstart.py；这里跑的是随 wheel
分发的 `aeb/cli.py` 本体（模块形式，不经 examples/），确保“pip 安装后立刻可跑”
不依赖源码 checkout 里的示例脚本。
"""

import subprocess
import sys
from pathlib import Path

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
