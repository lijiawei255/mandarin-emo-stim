"""核心指标进度条控件（NEGATIVE / VALENCE / AROUSAL）。

构成主义风格：大号数值 + 横向实心矩形进度条（无圆角），按指标配色。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QVBoxLayout, QWidget)


class MetricBar(QWidget):
    """单条指标展示（标签 + 数值 + 进度条）。"""

    # 各指标对应的进度条 chunk 颜色
    COLORS = {
        "negative": "#E30613",
        "valence": "#0057B8",
        "arousal": "#FFD600",
    }

    def __init__(self, title: str, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        self._build(title)

    def _build(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        head = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.value_label = QLabel("0.50")
        self.value_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #000000;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.title_label)
        head.addStretch()
        head.addWidget(self.value_label)
        layout.addLayout(head)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(500)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(24)
        color = self.COLORS.get(self.key, "#000000")
        # 指标条配色（chunk 颜色）
        self.bar.setStyleSheet(
            f"QProgressBar {{ border: 3px solid #000; background: #FFF; }}"
            f"QProgressBar::chunk {{ background: {color}; }}"
        )
        layout.addWidget(self.bar)

    def set_value(self, value: float) -> None:
        """设置指标值 [0,1]。"""
        value = max(0.0, min(1.0, float(value)))
        self.value_label.setText(f"{value:.2f}")
        self.bar.setValue(int(value * 1000))
