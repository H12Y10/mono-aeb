"""pytest 引导：未安装本包时也能直接 `pytest`（把仓库根加入 sys.path）。

CI 与 README 均按 `pip install -e .` 安装；保留此引导是为了让
「clone 后只装核心依赖、不安装本包」也能跑通测试。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
