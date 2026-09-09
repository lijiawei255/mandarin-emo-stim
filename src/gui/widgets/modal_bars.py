"""多模态分解详情控件（6 个模态的水平条形图）。

配色全部来自 theme.PALETTE；检测到副语言事件时用 warning 色强调。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar,
                               QVBoxLayout, QWidget)

from src.gui.theme import inline

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("多模态分解")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        for key, label in MODAL_LABELS.items():
            row = QHBoxLayout()
            row.setSpacing(8)
            name = QLabel(label)
            name.setStyleSheet(inline(color="text", font_weight="600", border="none"))
            name.setMinimumWidth(name.fontMetrics().horizontalAdvance("四个汉字") + 8)
            row.addWidget(name)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            row.addWidget(bar, 1)

            val = QLabel("0.50")
            val.setStyleSheet(inline(color="text", font_weight="600", border="none"))
            # 宽度按字体度量而非写死像素：不同平台/字体下 "0.00" 宽度差异可达 30%
            val.setMinimumWidth(val.fontMetrics().horizontalAdvance("0.00") + 10)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)

            layout.addLayout(row)
            self._bars[key] = bar
            self._value_labels[key] = val

        # 副语言事件标签区（状态色：默认中性，检测到时主色）
        self.events_label = QLabel("（无副语言事件）")
        self.events_label.setStyleSheet(self._events_style(active=False))
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

    @staticmethod
    def _events_style(active: bool) -> str:
        if active:
            return inline(color="warning", font_weight="600", font_size="12px",
                          border="none", padding_top="4px")
        return inline(color="text_muted", font_size="12px", border="none",
                      padding_top="4px")

    def update_events(self, events: list[dict]) -> None:
        """更新副语言事件标签（检测到时用警告色强调）。"""
        if not events:
            self.events_label.setText("（无副语言事件）")
            self.events_label.setStyleSheet(self._events_style(active=False))
            return
        names = [f"[{ev.get('name_zh', ev.get('label', '?'))} "
                 f"{ev.get('confidence', 0):.2f}]" for ev in events]
        self.events_label.setText(" ".join(names))
        self.events_label.setStyleSheet(self._events_style(active=True))
