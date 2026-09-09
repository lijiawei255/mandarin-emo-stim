"""真实平台渲染 GUI 并截图，用于视觉验证与 README 配图。

输出 3 分辨率 × 3 状态共 9 张 PNG 到 ``docs/images/ui/``（或 ``--out``）::

    python scripts/gui_screenshot.py                 # 全部 9 张
    python scripts/gui_screenshot.py --size 1440x900 --state analyzed

**必须在真实桌面平台运行**（Windows / X11 / Wayland），不要设 offscreen：
offscreen 平台无 CJK 字体回退，中文会渲染成方块，无法用于目视核验。
几何检查（无需字体）见 ``scripts/ui_geometry_check.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import portable  # noqa: E402,F401  触发便携环境
from scripts.ui_geometry_check import SIZES, STATES, apply_state  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "docs" / "images" / "ui"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GUI 截图")
    parser.add_argument("--size", default=None, help="WxH，默认三档全截")
    parser.add_argument("--state", default=None, choices=STATES)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    from src.gui.theme import build_qss

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(x) for x in args.size.lower().split("x"))] if args.size else list(SIZES)
    states = [args.state] if args.state else list(STATES)
    jobs = [(w, h, s) for (w, h) in sizes for s in states]

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # 逐窗截图，关窗不退出
    app.setStyleSheet(build_qss())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def run_next() -> None:
        if not jobs:
            app.quit()
            return
        w, h, state = jobs.pop(0)
        win = MainWindow(auto_load_models=False)
        win.resize(w, h)
        win.show()
        apply_state(win, state)

        def grab() -> None:
            pix = win.grab()
            out = out_dir / f"{w}x{h}_{state}.png"
            pix.save(str(out))
            print(f"截图 {out.name} ({pix.width()}x{pix.height()})", flush=True)
            win.close()
            QTimer.singleShot(200, run_next)

        QTimer.singleShot(700, grab)  # 等待样式/字体/布局稳定

    QTimer.singleShot(100, run_next)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
