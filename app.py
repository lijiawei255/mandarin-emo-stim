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
        from PySide6.QtCore import QTimer, Signal
        from PySide6.QtWidgets import QApplication
        self._qt_app = QApplication.instance() or QApplication(sys.argv)

        self._install_crash_handlers(self._qt_app)

        # 加载样式表
        qss_path = Path(__file__).parent / "src" / "gui" / "styles.qss"
        if not qss_path.exists():
            qss_path = Path(__file__).parent / "gui" / "styles.qss"
        if qss_path.exists():
            self._qt_app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

        from src.gui.main_window import MainWindow
        self._window = MainWindow(config=self.config)
        self._window.show()

        # 关键：Qt 事件循环默认不让 Python 检查 SIGINT，导致 Ctrl+C 在模型加载等
        # 长耗时操作期间无响应（看似卡死）。用一个周期定时器强制 Python 运行信号
        # 处理，使 Ctrl+C 能及时中断 exec()。
        self._sigint_timer = QTimer(self._qt_app)
        self._sigint_timer.start(200)  # 200ms 唤醒一次

        return self._qt_app.exec()

    @staticmethod
    def _install_crash_handlers(qt_app: "QApplication") -> None:
        """安装崩溃/中断处理器。

        - SIGINT(Ctrl+C)/SIGTERM：优雅退出（触发窗口 closeEvent 清理资源）。
        - sys.excepthook：未捕获异常兜底，记录日志并提示，避免静默崩溃。
        """
        import signal

        def _on_signal(signum, _frame):
            logger.warning("收到中断信号 %s，正在退出…", signum)
            # 优雅退出：让 QApplication 退出，触发 closeEvent 清理
            if qt_app:
                qt_app.quit()

        # SIGINT(Ctrl+C) 与 SIGTERM(kill)
        try:
            signal.signal(signal.SIGINT, _on_signal)
            # Windows 上 SIGTERM 等同于 SIGINT；非 Windows 注册 SIGTERM
            if hasattr(signal, "SIGTERM") and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, _on_signal)
        except (ValueError, OSError) as e:
            # 非主线程无法注册信号（如某些测试环境），忽略
            logger.warning("无法注册信号处理器（%s）", e)

        # 未捕获异常兜底
        def _on_uncaught(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                # Ctrl+C 在某些路径下走 excepthook
                qt_app.quit()
                return
            logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    None, "程序异常",
                    f"发生未预期的错误：\n{exc_value}\n\n"
                    "详细日志已记录到 portable_data/logs/。\n建议重启程序。",
                )
            except Exception:
                pass  # GUI 不可用时不再二次崩溃

        sys.excepthook = _on_uncaught


def main() -> int:
    app = Application()
    # 支持 `python app.py --headless path.wav` 无头运行
    headless = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--headless":
        headless = sys.argv[2]
    return app.run(headless_audio=headless)


if __name__ == "__main__":
    sys.exit(main())
