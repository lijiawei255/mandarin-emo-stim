"""命令行历史记录导出工具。

用法::

    python scripts/export_history.py --out history.json
    python scripts/export_history.py --out history.csv --format csv
    python scripts/export_history.py --clear   # 导出后清空
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import portable  # noqa: F401
from src.storage.history import HistoryManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 Mandarin-EmoStim 历史记录")
    parser.add_argument("--out", default="history_export.json",
                        help="输出文件路径（默认 history_export.json）")
    parser.add_argument("--format", default=None, choices=["json", "csv"],
                        help="导出格式（默认按文件扩展名推断）")
    parser.add_argument("--clear", action="store_true", help="导出后清空全部记录")
    args = parser.parse_args(argv)

    mgr = HistoryManager()
    count = mgr.db.get_count()
    if count == 0:
        print("历史记录为空，无内容可导出。", flush=True)
        return 0

    fmt = args.format
    if fmt is None:
        fmt = "csv" if args.out.lower().endswith(".csv") else "json"

    out = mgr.export(Path(args.out), fmt=fmt)
    print(f"已导出 {count} 条记录到：{out}（格式：{fmt}）", flush=True)

    if args.clear:
        deleted = mgr.clear()
        print(f"已清空 {deleted} 条记录。", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
