"""端到端零数据 demo：把 README 的承诺固化为断言。

README「快速开始」承诺 120 帧内出现 NORMAL → ATTENTION → FCW → AEB 的等级
演进，并给出了逐帧样例输出。本测试直接跑该 CLI，避免文档与实现悄悄脱节。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(QUICKSTART), *args],
        capture_output=True, text=True, cwd=str(ROOT),
    )


def test_dependency_check_passes():
    r = _run("--check")
    assert r.returncode == 0, f"--check exited with {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "all set" in r.stdout, f"--check did not report core dependencies as ready:\n{r.stdout}"


def test_demo_walks_the_full_risk_ladder():
    r = _run("--demo")
    out = r.stdout
    assert r.returncode == 0, f"--demo exited with {r.returncode}:\n{out}\n{r.stderr}"
    for token in ("NORMAL", "ATTENTION", "FCW", "AEB"):
        assert token in out, f"demo output is missing level {token}:\n{out}"
    assert "[quickstart] done" in out, f"demo did not finish normally:\n{out}"
