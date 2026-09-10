"""会话面板（v0.5）：开始/结束会话、试次与阶段控制、平滑状态显示。

放在输入区下方。逻辑不在这里，只发信号；MainWindow 负责持久化与状态跟踪。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.gui.theme import inline

PHASE_ZH = {"pre": "前测", "post": "后测"}


class SessionPanel(QWidget):
    start_requested = Signal()
    end_requested = Signal()
    new_trial_requested = Signal()
    phase_toggle_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("会话（前测 → 刺激 → 后测）")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_start = QPushButton("开始会话…")
        self.btn_start.setObjectName("btn_session_start")
        self.btn_end = QPushButton("结束并导出")
        self.btn_end.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_end)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.btn_trial = QPushButton("新试次")
        self.btn_trial.setEnabled(False)
        self.btn_phase = QPushButton("切到后测")
        self.btn_phase.setEnabled(False)
        row2.addWidget(self.btn_trial)
        row2.addWidget(self.btn_phase)
        layout.addLayout(row2)

        self.status = QLabel("未开始会话：每段录音独立判定")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(inline(color="text_muted", font_size="12px", border="none"))
        layout.addWidget(self.status)

        self.smoothed = QLabel("")
        self.smoothed.setWordWrap(True)
        self.smoothed.setStyleSheet(inline(color="accent_text", font_size="12px", font_weight="600", border="none"))
        layout.addWidget(self.smoothed)

        self.btn_start.clicked.connect(self.start_requested)
        self.btn_end.clicked.connect(self.end_requested)
        self.btn_trial.clicked.connect(self.new_trial_requested)
        self.btn_phase.clicked.connect(self.phase_toggle_requested)

    # ------------------------------------------------------------------ #
    def set_inactive(self) -> None:
        self.btn_start.setEnabled(True)
        for b in (self.btn_end, self.btn_trial, self.btn_phase):
            b.setEnabled(False)
        self.status.setText("未开始会话：每段录音独立判定")
        self.smoothed.setText("")

    def set_active(self, session_id: int, participant: str | None, target: str,
                   trial_index: int, phase: str, n_pre: int, n_post: int) -> None:
        self.btn_start.setEnabled(False)
        for b in (self.btn_end, self.btn_trial, self.btn_phase):
            b.setEnabled(True)
        self.btn_phase.setText("切到后测" if phase == "pre" else "切到前测")
        who = f"受试者 {participant}" if participant else "无档案"
        self.status.setText(f"会话 #{session_id} · {who} · 诱发目标：{target}\n"
                            f"试次 {trial_index} · 当前阶段：{PHASE_ZH[phase]}（前测 {n_pre} 段 / 后测 {n_post} 段）")

    def set_smoothed(self, state: dict | None) -> None:
        if not state or not state.get("n"):
            self.smoothed.setText("")
            return
        q = state.get("quadrant") or "—"
        stable = "稳定" if state.get("stable") else "未稳定"
        self.smoothed.setText(f"会话平滑：V {state['valence']:.2f} · A {state['arousal']:.2f} · "
                              f"象限 {q}（{stable}，{state['n']} 段，切换 {state.get('flips', 0)} 次）")
