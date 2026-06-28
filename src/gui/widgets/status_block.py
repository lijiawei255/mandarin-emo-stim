"""状态指示块控件（构成主义黑色块）。

显示运行状态、推理模式、模型加载进度。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class StatusBlock(QWidget):
    """黑色状态块。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusPanel")
        self.setFixedHeight(56)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.status_label = QLabel("就绪")
        self.mode_label = QLabel("推理模式：—")
        self.model_label = QLabel("模型：0/4")
        for lbl in (self.status_label, self.mode_label, self.model_label):
            lbl.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        layout.addWidget(self.status_label)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.model_label)

    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    def set_mode(self, mode: str) -> None:
        self.mode_label.setText(f"推理模式：{mode}")

    def set_model_progress(self, loaded: int, total: int = 4) -> None:
        self.model_label.setText(f"模型：{loaded}/{total}")
