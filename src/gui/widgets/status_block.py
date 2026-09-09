"""状态指示块控件。

次级面板底色，水平排列运行状态、推理模式、模型加载进度。
状态色语义化（theme.PALETTE）：就绪=success、运行中=accent、错误=error。
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from src.gui.theme import inline


class StatusBlock(QFrame):
    """浅色状态条（水平布局）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusPanel")
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(24)

        self.status_label = QLabel("就绪")
        self.mode_label = QLabel("推理模式：—")
        self.model_label = QLabel("模型：0/4")
        for lbl in (self.status_label, self.mode_label, self.model_label):
            lbl.setStyleSheet(inline(color="text", font_size="12px", border="none"))
        layout.addWidget(self.status_label)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.model_label)
        layout.addStretch()

    def set_status(self, status: str) -> None:
        """设置运行状态（按语义着色）。"""
        self.status_label.setText(status)
        if any(k in status for k in ("就绪", "完成")):
            color = "success"
        elif any(k in status for k in ("失败", "错误", "中断")):
            color = "error"
        elif any(k in status for k in ("中", "加载", "分析", "生成", "录音")):
            color = "accent"    # 运行中
        else:
            color = "text"
        self.status_label.setStyleSheet(
            inline(color=color, font_size="12px", font_weight="600", border="none")
        )

    def set_mode(self, mode: str) -> None:
        self.mode_label.setText(f"推理模式：{mode}")

    def set_model_progress(self, loaded: int, total: int = 4) -> None:
        self.model_label.setText(f"模型：{loaded}/{total}")
