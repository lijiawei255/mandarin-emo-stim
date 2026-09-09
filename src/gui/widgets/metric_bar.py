"""核心指标进度条控件（NEGATIVE / VALENCE / AROUSAL）。

标题行（小号大写标签 + 大号数值）+ 横向进度条。配色全部来自 theme.PALETTE。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from src.gui.theme import inline


class MetricBar(QWidget):
    """单条指标展示（标签 + 数值 + 进度条）。"""

    def __init__(self, title: str, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        self._build(title)

    def _build(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 8)
        layout.setSpacing(4)

        # 标题行：小号大写标签（左）+ 大号数值（右）
        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.value_label = QLabel("0.50")
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet(
            inline(font_size="24px", font_weight="700", color="text",
                   border="none", padding="0")
        )
        head.addWidget(self.title_label)
        head.addStretch()
        head.addWidget(self.value_label)
        layout.addLayout(head)

        # 进度条：统一主色填充，细线边框
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(500)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)  # 样式由全局 QSS（theme）统一
        layout.addWidget(self.bar)

    def set_value(self, value: float) -> None:
        """设置指标值 [0,1]。"""
        value = max(0.0, min(1.0, float(value)))
        self.value_label.setText(f"{value:.2f}")
        self.bar.setValue(int(value * 1000))
