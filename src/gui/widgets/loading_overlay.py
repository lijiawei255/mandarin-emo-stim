"""模型加载进度浮层控件。

在模型加载期间覆盖在主窗口之上的半透明面板，明确显示：
    - 当前正在加载哪个模型（阶段名）
    - 总体进度百分比（0-100%，含 4 个模型的阶段标记）
    - 已完成 / 进行中 / 待加载 的模型列表
让用户清楚知道加载进展与预计，避免误以为程序卡死。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QLabel, QProgressBar, QVBoxLayout,
                               QWidget)

from src.gui.theme import inline


# 4 个加载阶段的中文展示名（与 ModelManager.load_all 顺序一致）
STAGE_LABELS = [
    ("ASR (Paraformer)", "ASR 语音识别 · Paraformer-large"),
    ("emotion2vec", "声学情感 · emotion2vec_plus_large"),
    ("PANNs (CNN10)", "副语言事件 · PANNs CNN10"),
    ("LLM (Qwen3)", "文本语义 · Qwen3-1.7B"),
]


class LoadingOverlay(QWidget):
    """模型加载进度浮层。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")
        self._build()
        self._reset()

        # 默认隐藏，仅在加载时显示
        self.hide()

    def _build(self) -> None:
        # 半透明遮罩 + 居中卡片
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch()

        card = QFrame()
        card.setObjectName("LoadingCard")
        card.setFixedSize(520, 320)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(12)

        title = QLabel("模型加载中")
        title.setStyleSheet(inline(font_size="20px", font_weight="700", color="text", border="none"))
        card_layout.addWidget(title)

        self.hint = QLabel("首次启动需加载 4 个模型（约需 20-40 秒），请稍候…")
        self.hint.setStyleSheet(inline(font_size="12px", color="text_muted", border="none"))
        self.hint.setWordWrap(True)
        card_layout.addWidget(self.hint)

        # 总进度条（主色填充）
        self.total_bar = QProgressBar()
        self.total_bar.setRange(0, 100)
        self.total_bar.setValue(0)
        self.total_bar.setFixedHeight(20)
        self.total_bar.setTextVisible(True)
        self.total_bar.setFormat("总进度 %p%")
        card_layout.addWidget(self.total_bar)

        # 当前阶段（主色）
        self.current_label = QLabel("准备中…")
        self.current_label.setStyleSheet(self._current_style("accent"))
        card_layout.addWidget(self.current_label)

        # 4 个阶段的状态列表
        self.stage_labels: list[QLabel] = []
        for _, display in STAGE_LABELS:
            lbl = QLabel(f"○  {display}")
            lbl.setStyleSheet(self._stage_style("pending"))
            self.stage_labels.append(lbl)
            card_layout.addWidget(lbl)

        card_layout.addStretch()

        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)
        outer.addStretch()

    @staticmethod
    def _current_style(color_key: str) -> str:
        return inline(font_size="13px", font_weight="700", color=color_key, border="none")

    @staticmethod
    def _stage_style(state: str) -> str:
        """阶段行样式：pending（待加载）/ active（进行中）/ done（完成）。"""
        if state == "done":
            return inline(font_size="12px", color="success", font_weight="600", border="none")
        if state == "active":
            return inline(font_size="12px", color="accent", font_weight="700", border="none")
        return inline(font_size="12px", color="text_faint", border="none")

    def _reset(self) -> None:
        """重置为初始状态。"""
        self.total_bar.setValue(0)
        self.current_label.setText("准备中…")
        self.current_label.setStyleSheet(self._current_style("accent"))
        for i, lbl in enumerate(self.stage_labels):
            _, display = STAGE_LABELS[i]
            lbl.setText(f"○  {display}")
            lbl.setStyleSheet(self._stage_style("pending"))

    # ------------------------------------------------------------------ #
    def show_loading(self) -> None:
        """显示加载浮层（重置后显示）。"""
        self._reset()
        self._resize_to_parent()
        self.show()
        self.raise_()

    def _resize_to_parent(self) -> None:
        """铺满父窗口。"""
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())

    def showEvent(self, event) -> None:
        """显示时确保铺满父窗口。"""
        self._resize_to_parent()
        super().showEvent(event)

    def update_progress(self, stage: str, pct: int) -> None:
        """更新进度。

        Args:
            stage: 阶段名（与 STAGE_LABELS 第一列匹配，或任意描述）。
            pct: 总进度百分比 0-100。
        """
        self.total_bar.setValue(int(pct))

        # 标记已完成 / 进行中的阶段
        stage_idx = None
        for i, (key, _) in enumerate(STAGE_LABELS):
            if key in stage or stage in key:
                stage_idx = i
                break

        for i, lbl in enumerate(self.stage_labels):
            _, display = STAGE_LABELS[i]
            if stage_idx is not None and i < stage_idx:
                lbl.setText(f"✓  {display}")
                lbl.setStyleSheet(self._stage_style("done"))
            elif stage_idx is not None and i == stage_idx:
                lbl.setText(f"●  {display}")
                lbl.setStyleSheet(self._stage_style("active"))
            else:
                lbl.setText(f"○  {display}")
                lbl.setStyleSheet(self._stage_style("pending"))

        self.current_label.setText(f"正在加载：{stage}")

    def show_done(self) -> None:
        """加载完成：全部标记为完成后淡出。"""
        for i, lbl in enumerate(self.stage_labels):
            _, display = STAGE_LABELS[i]
            lbl.setText(f"✓  {display}")
            lbl.setStyleSheet(self._stage_style("done"))
        self.total_bar.setValue(100)
        self.current_label.setText("加载完成 ✓")
        self.current_label.setStyleSheet(self._current_style("success"))
        # 短暂展示后隐藏（由调用方控制，或直接隐藏）
        self.hide()

    def show_failed(self, msg: str) -> None:
        """加载失败：显示错误信息。"""
        self.current_label.setText(f"加载失败：{msg}")
        self.current_label.setStyleSheet(self._current_style("error"))
        self.current_label.setWordWrap(True)
        self.hint.setText("模型加载失败，请查看错误信息。可关闭后重试或检查网络/显存。")
