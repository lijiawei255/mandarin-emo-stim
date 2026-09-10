"""GUI 布局几何检查：自动发现重叠、挤压、截断、溢出。

作为 ``tests/test_gui_layout.py`` 的核心，也可独立运行输出报告::

    python scripts/ui_geometry_check.py            # 三分辨率 × 三状态
    python scripts/ui_geometry_check.py --size 1280x720

检查项（对所有可见控件）：
    overlap   同一父控件下两个可见兄弟控件的几何矩形相交（忽略「容器套内容」
              的父子关系与浮层）；
    squeeze   非自动换行的 QLabel / QPushButton / QComboBox 的 sizeHint 宽度
              超过实际分配宽度（文字会被截断或省略）；
    overflow  子控件超出父控件的 rect（被裁切）；
    too_short 控件高度小于其 minimumSizeHint 高度（内容被压扁）。

设计取舍：只报告「可见 + 有尺寸」的控件；进度条、滑块等无文字控件不做
squeeze 检查；LoadingOverlay 是有意覆盖全窗口的浮层，跳过 overlap 检查。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SIZES = ((1280, 720), (1440, 900), (1920, 1080))
STATES = ("idle", "analyzed", "loading")


@dataclass(frozen=True)
class Issue:
    kind: str          # overlap / squeeze / overflow / too_short
    widget: str        # 控件描述（类名#objectName 或文本）
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.widget}: {self.detail}"


def _describe(w) -> str:
    name = w.objectName() or ""
    text = ""
    if hasattr(w, "text") and callable(w.text):
        try:
            text = str(w.text())[:24]
        except Exception:  # noqa: BLE001
            text = ""
    label = type(w).__name__
    if name:
        label += f"#{name}"
    if text:
        label += f"({text!r})"
    return label


def _is_overlay(w) -> bool:
    return w.objectName() == "LoadingOverlay" or type(w).__name__ == "LoadingOverlay"


def _visible_children(parent):
    from PySide6.QtWidgets import QWidget
    out = []
    for c in parent.children():
        if not isinstance(c, QWidget):
            continue
        if not c.isVisible() or c.isWindow():
            continue
        g = c.geometry()
        if g.width() <= 0 or g.height() <= 0:
            continue
        out.append(c)
    return out


_TEXT_TYPES = ("QLabel", "QPushButton", "QComboBox")


def collect_issues(window, *, tolerance: int = 1) -> list[Issue]:
    """遍历窗口所有可见控件，返回布局问题列表（空列表 = 通过）。"""
    from PySide6.QtWidgets import QLabel, QWidget

    issues: list[Issue] = []
    stack: list[QWidget] = [window]
    while stack:
        parent = stack.pop()
        kids = _visible_children(parent)
        prect = parent.rect()

        for c in kids:
            g = c.geometry()
            # overflow
            if not _is_overlay(c) and (
                g.left() < prect.left() - tolerance or g.top() < prect.top() - tolerance
                or g.right() > prect.right() + tolerance or g.bottom() > prect.bottom() + tolerance
            ):
                issues.append(Issue("overflow", _describe(c),
                                    f"geometry={g.getRect()} parent={prect.getRect()}"))
            # squeeze（仅文字控件，且非自动换行）
            if type(c).__name__ in _TEXT_TYPES:
                wrap = isinstance(c, QLabel) and c.wordWrap()
                if not wrap:
                    need = c.sizeHint().width()
                    if need > g.width() + tolerance and need > 0:
                        issues.append(Issue("squeeze", _describe(c),
                                            f"needs {need}px, has {g.width()}px"))
            # too_short（文字隐藏的进度条是有意的细线，minimumSizeHint 含文字高，跳过）
            min_h = c.minimumSizeHint().height()
            thin_bar = type(c).__name__ == "QProgressBar" and not c.isTextVisible()
            if not thin_bar and 0 < min_h and g.height() + tolerance < min_h:
                issues.append(Issue("too_short", _describe(c),
                                    f"needs {min_h}px, has {g.height()}px"))
            stack.append(c)

        # overlap：两两检查兄弟
        for i in range(len(kids)):
            a = kids[i]
            if _is_overlay(a):
                continue
            for j in range(i + 1, len(kids)):
                b = kids[j]
                if _is_overlay(b):
                    continue
                inter = a.geometry().intersected(b.geometry())
                if inter.width() > tolerance and inter.height() > tolerance:
                    issues.append(Issue("overlap", _describe(a),
                                        f"with {_describe(b)}: {inter.getRect()}"))
    return issues


# ---------------------------------------------------------------------- #
# 状态填充（与 gui_screenshot.py 共用的桩数据）
# ---------------------------------------------------------------------- #
def populate_analyzed(win) -> None:
    import numpy as np
    win._on_analysis_done({
        "negative": 0.593, "valence": 0.407, "arousal": 0.579,
        "dominant_quadrant": "Q2",
        "modal_scores": {
            "acoustic": {"negative": 0.292, "arousal": 0.599},
            "prosody": {"negative": 0.697, "arousal": 0.605},
            "paralang": {"negative": 0.500, "arousal": 0.500},
            "physical": {"negative": 0.662, "arousal": 0.391},
            "text_llm": {"negative": 0.900, "arousal": 0.950},
            "text_stat": {"negative": 0.480, "arousal": 0.184},
        },
        "asr_text": ("我最近总是没有办法静下心，我感觉我的心里一直悬着一块石头，"
                     "我在夜里感觉我会翻来覆去睡不着，一点小事儿就会让我胡思乱想。"),
        "audio_quality": {"snr_db": 8.0},          # 触发 SNR 警告
        "paralang_events": [{"name_zh": "叹息", "confidence": 0.72, "label": "Sigh"}],
        "memberships": {"Q1": 0.05, "Q2": 0.80, "Q3": 0.10, "Q4": 0.05},
        "duration": 12.3, "asr_confidence": 0.9,
        "degraded_modalities": ["paralang"],       # 触发降级提示（两行警告）
        "uncertainty": {"negative_sd": 0.21, "arousal_sd": 0.08, "n_active": 5},   # v0.3 象限标题加长
        "calibration_source": "personal",
    })
    win.status_block.set_mode("CUDA")
    win.status_block.set_model_progress(4, 4)
    sr = 44100
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    wave = (0.1 * np.sin(2 * np.pi * 300 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)))
    win.waveform.set_waveform(np.column_stack([wave, wave]).astype(np.float32), sr)
    win.param_label.setText("♩=48BPM  f=280Hz  粉噪=20%  谐和=natural_harmonics  时长=10s")
    for b in (win.btn_play, win.btn_pause, win.btn_stop, win.btn_save):
        b.setEnabled(True)


def apply_state(win, state: str) -> None:
    if state == "analyzed":
        populate_analyzed(win)
    elif state == "loading":
        win.loading_overlay.show_loading()
        win.loading_overlay.update_progress("emotion2vec", 25)
        win.status_block.set_status("加载：emotion2vec 25%")
    elif state != "idle":
        raise ValueError(state)


def make_window(width: int, height: int, state: str):
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import MainWindow
    from src.gui.theme import build_qss
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(build_qss())
    win = MainWindow(auto_load_models=False)
    win.resize(width, height)
    win.show()
    apply_state(win, state)
    app.processEvents()
    app.processEvents()
    return app, win


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GUI 布局几何检查")
    parser.add_argument("--size", default=None, help="WxH，默认三档全跑")
    parser.add_argument("--state", default=None, choices=STATES)
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    sizes = [tuple(int(x) for x in args.size.lower().split("x"))] if args.size else SIZES
    states = [args.state] if args.state else STATES
    total = 0
    for w, h in sizes:
        for state in states:
            app, win = make_window(w, h, state)
            issues = collect_issues(win)
            tag = f"{w}x{h}/{state}"
            if issues:
                total += len(issues)
                print(f"== {tag}: {len(issues)} issue(s)")
                for it in issues:
                    print("   ", it)
            else:
                print(f"== {tag}: OK")
            win.close()
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
