#!/usr/bin/env python3
"""隔离模式入口：允许用 `python3 -I run_evaluate.py ...` 运行离线评估。

python3 -I 不会把当前目录加入 sys.path，因此这里按脚本位置显式加入项目根；
输入路径仍由参数传入，解释器不会从数据目录加载模块（防止植入的 json.py 等）。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_replay_book.evaluate import _module_main  # noqa: E402

if __name__ == '__main__':
    sys.exit(_module_main())
