"""多模态分解详情控件（6 个模态的水平条形图）。

构成主义风格：白底蓝色块面板，白色实心矩形条 + 白色数值标注。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QVBoxLayout, QWidget)

# 6 模态显示名（中文）
MODAL_LABELS = {
    "acoustic": "声学情感",
    "prosody": "韵律特征",
    "paralang": "副语言",
    "physical": "物理特征",
    "text_llm": "文本语义",
    "text_stat": "文本统计",
}


class ModalBars(QWidget):
    """6 模态分项条形图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars: dict[str, QProgressBar] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("多模态分解")
        title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        for key, label in MODAL_LABELS.items():
            row = QHBoxLayout()
            name = QLabel(label)
            name.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            name.setFixedWidth(80)
            row.addWidget(name)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(16)
            bar.setStyleSheet(
                "QProgressBar { border: 2px solid #FFF; background: transparent; }"
                "QProgressBar::chunk { background: #FFFFFF; }"
            )
            row.addWidget(bar, 1)

            val = QLabel("0.50")
            val.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            val.setFixedWidth(40)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(val)

            layout.addLayout(row)
            self._bars[key] = bar
            self._value_labels[key] = val

        # 副语言事件标签区
        self.events_label = QLabel("")
        self.events_label.setStyleSheet(
            "color: #000; background: #FFF; font-weight: bold; padding: 4px;"
            "border: 2px solid #FFF;"
        )
        self.events_label.setWordWrap(True)
        layout.addWidget(self.events_label)
        layout.addStretch()

    def update_scores(self, modal_scores: dict, axis: str = "negative") -> None:
        """更新各模态分值。

        Args:
            modal_scores: {模态名: {"negative": x, "arousal": y}}。
            axis: 展示哪个轴（"negative" / "arousal"）。
        """
        for key, bar in self._bars.items():
            sc = modal_scores.get(key, {})
            val = sc.get(axis, 0.5)
            val = max(0.0, min(1.0, float(val)))
            bar.setValue(int(val * 1000))
            self._value_labels[key].setText(f"{val:.2f}")

    def update_events(self, events: list[dict]) -> None:
        """更新副语言事件标签。"""
        if not events:
            self.events_label.setText("（无副语言事件）")
            return
        names = [f"[{ev.get('name_zh', ev.get('label', '?'))} "
                 f"{ev.get('confidence', 0):.2f}]" for ev in events]
        self.events_label.setText(" ".join(names))
