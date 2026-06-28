"""启动 GUI 并延时截图,用于视觉验证布局。

非 offscreen 模式启动真实窗口,用桩数据填充后截图保存(大屏 1440x900 +
最小支持分辨率 1280x720 两张)。用 widget.grab() 直接渲染,不依赖屏幕坐标。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 真实窗口(不设 offscreen)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import portable  # noqa: F401


def main() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)

    # 加载样式表
    qss = PROJECT_ROOT / "src" / "gui" / "styles.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))

    # 不自动加载模型(避免21s等待与显存占用),用桩数据填充界面以验证布局
    win = MainWindow(auto_load_models=False)
    # 用固定较大尺寸(非 maximized,避免多屏坐标问题)
    win.resize(1440, 900)
    win.show()
    win.raise_()
    win.activateWindow()

    # 用桩数据填充,模拟「分析完成」后的界面状态
    def populate():
        from src.fusion.quadrant import QUADRANT_NAMES
        win.metric_negative.set_value(0.593)
        win.metric_valence.set_value(0.407)
        win.metric_arousal.set_value(0.579)
        win.quadrant_label.setText("情绪象限：Q2 — 焦虑/紧张")
        win.asr_text.setPlainText(
            "我最近总是没有办法静下心，我感觉我的心里一直悬着一块石头，"
            "我在夜里感觉我会翻来覆去睡不着，一点小事儿就会让我胡思乱想。"
        )
        win.modal_bars.update_scores({
            "acoustic": {"negative": 0.292, "arousal": 0.599},
            "prosody": {"negative": 0.697, "arousal": 0.605},
            "paralang": {"negative": 0.500, "arousal": 0.500},
            "physical": {"negative": 0.662, "arousal": 0.391},
            "text_llm": {"negative": 0.900, "arousal": 0.950},
            "text_stat": {"negative": 0.480, "arousal": 0.184},
        }, axis="negative")
        win.modal_bars.update_events([
            {"name_zh": "叹息", "confidence": 0.72, "label": "Sigh"},
        ])
        win.status_block.set_status("分析完成")
        win.status_block.set_mode("CUDA")
        win.status_block.set_model_progress(4, 4)

        # 生成一段桩波形
        import numpy as np
        sr = 44100
        t = np.linspace(0, 10, int(sr * 10), endpoint=False)
        wave = (0.1 * np.sin(2 * np.pi * 300 * t) *
                (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)))
        stereo = np.column_stack([wave, wave]).astype(np.float32)
        win.waveform.set_waveform(stereo, sr)
        win.param_label.setText("♩=48BPM  f=280Hz  粉噪=20%  谐和=natural_harmonics  时长=10s")

    QTimer.singleShot(800, populate)

    # 截图:先大屏(1440x900),再切小屏(1280x720)各截一次
    def _grab(tag: str, then=None):
        import time as _t
        ts = _t.strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "portable_data" / "temp" / f"gui_{tag}_{ts}.png"
        pixmap = win.grab()
        pixmap.save(str(out))
        print(f"截图[{tag}]: {out} ({pixmap.width()}x{pixmap.height()})", flush=True)
        if then:
            QTimer.singleShot(800, then)
        else:
            app.quit()

    def _resize_small():
        win.resize(1280, 720)

    QTimer.singleShot(2000, lambda: _grab("large", then=_resize_small))
    QTimer.singleShot(3200, lambda: _grab("small"))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
