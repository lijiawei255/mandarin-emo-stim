"""应用主类：生命周期管理、模块协调。

负责便携环境初始化、日志、硬件检测、应用启动与退出清理。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src import portable

logger = portable.setup_logging()


class Application:
    """Mandarin-EmoStim 应用主类。"""

    def __init__(self):
        self.config = None
        self._qt_app = None
        self._window = None

    def _detect_hardware(self) -> str:
        """硬件检测，返回推理模式描述。"""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                mode = f"GPU(CUDA): {name} ({vram:.1f}GB)"
            else:
                mode = "CPU"
            logger.info("硬件检测: %s", mode)
            return mode
        except Exception as e:
            logger.warning("硬件检测失败: %s", e)
            return "未知"

    def run(self, headless_audio: str | None = None) -> int:
        """启动应用。

        Args:
            headless_audio: 若提供，则无头分析该音频文件（不启动 GUI）。
        """
        from src.config_loader import load_settings
        self.config = load_settings()
        logger.info("=== Mandarin-EmoStim 启动 ===")
        self._detect_hardware()

        if headless_audio:
            return self._run_headless(headless_audio)
        return self._run_gui()

    def _run_headless(self, audio_path: str) -> int:
        from src.cli import main as cli_main
        return cli_main(["--audio", audio_path])

    def _run_gui(self) -> int:
        from PySide6.QtWidgets import QApplication
        self._qt_app = QApplication.instance() or QApplication(sys.argv)

        # 加载样式表
        qss_path = Path(__file__).parent / "src" / "gui" / "styles.qss"
        if not qss_path.exists():
            qss_path = Path(__file__).parent / "gui" / "styles.qss"
        if qss_path.exists():
            self._qt_app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

        from src.gui.main_window import MainWindow
        self._window = MainWindow(config=self.config)
        self._window.show()
        return self._qt_app.exec()


def main() -> int:
    app = Application()
    # 支持 `python app.py --headless path.wav` 无头运行
    headless = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--headless":
        headless = sys.argv[2]
    return app.run(headless_audio=headless)


if __name__ == "__main__":
    sys.exit(main())
