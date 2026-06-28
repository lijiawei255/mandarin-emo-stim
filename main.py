"""程序入口。

仅做环境初始化与 app 启动。用法::

    python main.py                 # 启动 GUI
    python main.py --headless x.wav  # 无头分析单个音频

等价于直接运行 ``python app.py``。
"""

from __future__ import annotations

import sys

from app import main

if __name__ == "__main__":
    sys.exit(main())
