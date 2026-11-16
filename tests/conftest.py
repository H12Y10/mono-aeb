"""pytest 引导：未安装本包时也能直接 `pytest`（把仓库根加入 sys.path）。

CI 与 README 均按 `pip install -e .` 安装；保留此引导是为了让
“clone 后只装核心依赖、不安装本包”也能跑通测试。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import re  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def parse_levels():
    """从 CLI 输出里抽出逐帧等级序列。

    为什么要这样断言：`"AEB" in stdout` 会被头部那行
    `AEB threshold = ...` 命中，链路只升到 FCW 也能通过——测试形同虚设。
    这里只认 `frame <n>: <LEVEL>` 这种逐帧行。
    """

    def _parse(out: str) -> list[str]:
        return re.findall(r"^\s*frame\s+\d+:\s+([A-Z_]+)", out, re.M)

    return _parse
