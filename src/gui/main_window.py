"""构成主义风格主窗口。

五大功能块面：
    左红   : 音频输入控制
    中白   : 核心情感量化展示
    右蓝   : 多模态分解详情
    底黄   : 声刺激生成与波形
    右黑   : 状态指示

多线程：模型加载 / 分析 / 刺激生成均在工作线程，通过 Signal 更新 UI。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QTextEdit,
                               QVBoxLayout, QWidget)

from src import portable
from src.config_loader import load_settings
from src.gui.threads import (AnalysisWorker, ModelLoadWorker, StimulusWorker)
from src.gui.widgets.metric_bar import MetricBar
from src.gui.widgets.modal_bars import ModalBars
from src.gui.widgets.status_block import StatusBlock
from src.gui.widgets.waveform_view import WaveformView
from src.stimulus.player import AudioPlayer
from src.storage.history import HistoryFull, HistoryManager

logger = logging.getLogger("mandarin_emo_stim.gui")


class MainWindow(QMainWindow):
    """构成主义风格主窗口。"""

    def __init__(self, config: dict | None = None, auto_load_models: bool = True):
        super().__init__()
        self.config = config if config is not None else load_settings()
        self.setObjectName("MainWindow")
        self.setWindowTitle("Mandarin-EmoStim · 中文语音情感分析与声刺激生成")

        self.manager = None
        self.last_result: dict | None = None
        self.last_stimulus = None
        self.history = HistoryManager()

        self._build_ui()
        self._connect_workers_slots()

        if auto_load_models:
            QTimer.singleShot(100, self.start_model_loading)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 上半部：红 | 白 | 蓝 三栏
        top = QHBoxLayout()
        top.setSpacing(0)
        top.addWidget(self._build_red_panel(), 1)
        top.addWidget(self._build_main_panel(), 3)
        top.addWidget(self._build_blue_panel(), 2)
        root.addLayout(top, 3)

        # 下半部：黄色刺激区
        root.addWidget(self._build_yellow_panel(), 2)

        # 状态块
        self.status_block = StatusBlock()
        root.addWidget(self.status_block)

        self._build_menu()

        self.resize(1280, 800)

    def _panel_frame(self, name: str) -> QFrame:
        f = QFrame()
        f.setObjectName(name)
        return f

    # ----- 红色块：音频输入控制 -----
    def _build_red_panel(self) -> QWidget:
        panel = self._panel_frame("RedPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._h1("音频输入"))

        self.btn_record = QPushButton("开始录音")
        self.btn_record.setCheckable(True)
        self.btn_upload = QPushButton("上传音频")
        self.btn_reset = QPushButton("停止 / 重置")
        for b in (self.btn_record, self.btn_upload, self.btn_reset):
            layout.addWidget(b)

        self.device_combo = QComboBox()
        self.device_combo.addItems(self._list_input_devices())
        layout.addWidget(self.device_combo)

        self.snr_warning = QLabel("")
        self.snr_warning.setStyleSheet("color: #FFD600; font-weight: bold;")
        self.snr_warning.setWordWrap(True)
        layout.addWidget(self.snr_warning)
        layout.addStretch()

        self.btn_record.clicked.connect(self.on_record_clicked)
        self.btn_upload.clicked.connect(self.on_upload_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        return panel

    def _list_input_devices(self) -> list[str]:
        try:
            from src.audio.recorder import AudioRecorder
            devs = AudioRecorder.list_devices()
            return [f"{d['id']}: {d['name']}" for d in devs] or ["（未检测到麦克风）"]
        except Exception:
            return ["（无法枚举设备）"]

    # ----- 白色主区：核心指标 -----
    def _build_main_panel(self) -> QWidget:
        panel = self._panel_frame("MainPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.metric_negative = MetricBar("NEGATIVE SCORE", "negative")
        self.metric_valence = MetricBar("VALENCE", "valence")
        self.metric_arousal = MetricBar("AROUSAL", "arousal")
        layout.addWidget(self.metric_negative)
        layout.addWidget(self.metric_valence)
        layout.addWidget(self.metric_arousal)

        self.quadrant_label = QLabel("情绪象限：—")
        self.quadrant_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.quadrant_label)

        layout.addWidget(self._h1("ASR 转写"))
        self.asr_text = QTextEdit()
        self.asr_text.setReadOnly(True)
        self.asr_text.setFixedHeight(80)
        self.asr_text.setPlaceholderText("（转写结果将显示于此）")
        layout.addWidget(self.asr_text)
        layout.addStretch()
        return panel

    # ----- 蓝色块：多模态分解 -----
    def _build_blue_panel(self) -> QWidget:
        panel = self._panel_frame("BluePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        self.modal_bars = ModalBars()
        layout.addWidget(self.modal_bars)
        return panel

    # ----- 黄色块：声刺激 -----
    def _build_yellow_panel(self) -> QWidget:
        panel = self._panel_frame("YellowPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        head = QHBoxLayout()
        self.btn_generate = QPushButton("生成刺激波形")
        self.btn_generate.setEnabled(False)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_stop = QPushButton("■ 停止")
        self.btn_save = QPushButton("💾 保存")
        for b in (self.btn_play, self.btn_pause, self.btn_stop, self.btn_save):
            b.setEnabled(False)
        head.addWidget(self.btn_generate)
        head.addWidget(self.btn_play)
        head.addWidget(self.btn_pause)
        head.addWidget(self.btn_stop)
        head.addWidget(self.btn_save)
        head.addStretch()
        self.param_label = QLabel("♩=—  f=—  时长=—")
        self.param_label.setStyleSheet("font-weight: bold;")
        head.addWidget(self.param_label)
        layout.addLayout(head)

        self.waveform = WaveformView()
        layout.addWidget(self.waveform)

        vol_row = QHBoxLayout()
        vol_row.addWidget(self._h1("音量"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        vol_row.addWidget(self.volume_slider)
        layout.addLayout(vol_row)

        self.player = AudioPlayer(sr=int(self.config["audio"]["stimulus_sample_rate"]))
        self.player.position_updated.connect(self.waveform.set_position)
        self.player.playback_finished.connect(self.on_playback_finished)

        self.btn_generate.clicked.connect(self.on_generate_clicked)
        self.btn_play.clicked.connect(self.on_play_clicked)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.volume_slider.valueChanged.connect(
            lambda v: self.player.set_volume(v / 100.0)
        )
        return panel

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        act_export = QAction("导出历史记录", self)
        act_export.triggered.connect(self.on_export_history)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_export)
        file_menu.addAction(act_quit)

        help_menu = menubar.addMenu("帮助")
        act_about = QAction("关于 / 安全声明", self)
        act_about.triggered.connect(self.on_about)
        help_menu.addAction(act_about)

    def _h1(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        return lbl

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #
    def _connect_workers_slots(self) -> None:
        # 占位：worker 信号在创建时连接
        pass

    def start_model_loading(self) -> None:
        self.status_block.set_status("模型加载中…")
        self.model_worker = ModelLoadWorker(config=self.config)
        self.model_worker.progress.connect(self._on_model_progress)
        self.model_worker.finished_ok.connect(self._on_models_loaded)
        self.model_worker.failed.connect(self._on_model_failed)
        self.model_worker.start()

    def _on_model_progress(self, stage: str, pct: int) -> None:
        self.status_block.set_status(f"加载：{stage} {pct}%")

    def _on_models_loaded(self, manager) -> None:
        self.manager = manager
        self.status_block.set_status("就绪")
        self.status_block.set_mode(manager.get_device().upper())
        self.status_block.set_model_progress(manager.loaded_count, 4)
        self.btn_generate.setEnabled(False)  # 待分析完成

    def _on_model_failed(self, msg: str) -> None:
        self.status_block.set_status("模型加载失败")
        QMessageBox.critical(self, "模型加载失败", msg)

    def on_record_clicked(self) -> None:
        # 录音功能在阶段 7 简化：提示用户改用上传（录音需更复杂的线程管理）
        if self.btn_record.isChecked():
            QMessageBox.information(
                self, "录音",
                "录音功能请使用命令行或后续版本。当前请使用「上传音频」分析已有文件。",
            )
            self.btn_record.setChecked(False)

    def on_upload_clicked(self) -> None:
        if self.manager is None:
            QMessageBox.warning(self, "未就绪", "模型尚未加载完成，请稍候。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a)",
        )
        if path:
            self._start_analysis(path)

    def on_reset_clicked(self) -> None:
        self.player.stop()
        self.last_result = None
        self.last_stimulus = None
        self.asr_text.clear()
        for m in (self.metric_negative, self.metric_valence, self.metric_arousal):
            m.set_value(0.5)
        self.quadrant_label.setText("情绪象限：—")
        self.btn_generate.setEnabled(False)
        for b in (self.btn_play, self.btn_pause, self.btn_stop, self.btn_save):
            b.setEnabled(False)

    def _start_analysis(self, audio_path: str) -> None:
        self.status_block.set_status("分析中…")
        self.btn_generate.setEnabled(False)
        self.analysis_worker = AnalysisWorker(self.manager, audio_path)
        self.analysis_worker.progress.connect(
            lambda s, p: self.status_block.set_status(f"分析：{s} {p}%")
        )
        self.analysis_worker.finished_ok.connect(self._on_analysis_done)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

    def _on_analysis_done(self, result: dict) -> None:
        self.last_result = result
        self.status_block.set_status("分析完成")
        self.metric_negative.set_value(result["negative"])
        self.metric_valence.set_value(result["valence"])
        self.metric_arousal.set_value(result["arousal"])
        q = result["dominant_quadrant"]
        from src.fusion.quadrant import QUADRANT_NAMES
        self.quadrant_label.setText(f"情绪象限：{q} — {QUADRANT_NAMES.get(q, '')}")
        self.asr_text.setPlainText(result.get("asr_text", ""))

        # SNR 警告
        snr = result["audio_quality"]["snr_db"]
        thr = self.config["thresholds"]["snr_warning_db"]
        if snr < thr:
            self.snr_warning.setText(f"⚠ 录音环境嘈杂（SNR {snr:.1f}dB），结果可能不准确")
        else:
            self.snr_warning.setText("")

        # 多模态分解
        self.modal_bars.update_scores(result["modal_scores"], axis="negative")
        self.modal_bars.update_events(result.get("paralang_events", []))

        self.btn_generate.setEnabled(True)

        # 存入历史
        try:
            self._save_to_history(result, audio_path=None)
        except HistoryFull:
            QMessageBox.warning(self, "历史记录已满",
                                f"已达上限（{self.history.max_records} 条），请先导出并清空。")

    def _on_analysis_failed(self, msg: str) -> None:
        self.status_block.set_status("分析失败")
        QMessageBox.critical(self, "分析失败", msg)

    def on_generate_clicked(self) -> None:
        if self.last_result is None:
            return
        self.status_block.set_status("生成刺激中…")
        self.stim_worker = StimulusWorker(self.last_result)
        self.stim_worker.finished_ok.connect(self._on_stimulus_done)
        self.stim_worker.failed.connect(self._on_stimulus_failed)
        self.stim_worker.start()

    def _on_stimulus_done(self, waveform, params) -> None:
        self.last_stimulus = waveform
        self.status_block.set_status("刺激已生成")
        self.waveform.set_waveform(waveform, self.player.sr)
        bpm = int(params.pr * 60)
        self.param_label.setText(
            f"♩={bpm}BPM  f={int(params.f0)}Hz  粉噪={params.noise_ratio*100:.0f}%  "
            f"谐和={params.harmony}  时长={self.waveform.duration:.0f}s"
        )
        for b in (self.btn_play, self.btn_pause, self.btn_stop, self.btn_save):
            b.setEnabled(True)

    def _on_stimulus_failed(self, msg: str) -> None:
        self.status_block.set_status("生成失败")
        QMessageBox.critical(self, "刺激生成失败", msg)

    def on_play_clicked(self) -> None:
        if self.last_stimulus is not None:
            self.player.play(self.last_stimulus, loop=False)

    def on_stop_clicked(self) -> None:
        self.player.stop()
        self.waveform.clear_position()

    def on_playback_finished(self) -> None:
        self.waveform.clear_position()

    def on_save_clicked(self) -> None:
        if self.last_stimulus is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存刺激音频", "stimulus.wav",
                                              "WAV (*.wav)")
        if path:
            import soundfile as sf
            sf.write(path, self.last_stimulus, self.player.sr, subtype="PCM_16")

    def on_export_history(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出历史记录", "history.json",
                                              "JSON (*.json);;CSV (*.csv)")
        if path:
            fmt = "csv" if path.lower().endswith(".csv") else "json"
            self.history.export(Path(path), fmt=fmt)
            QMessageBox.information(self, "导出完成", f"已导出到：{path}")

    def on_about(self) -> None:
        QMessageBox.about(
            self, "关于 Mandarin-EmoStim",
            "<h3>Mandarin-EmoStim</h3>"
            "<p>全离线中文语音情感分析与个性化声刺激生成科研工具。</p>"
            "<p><b>声音安全声明</b>：生成刺激峰值限幅 -10dBFS"
            "（约 70-75dB SPL，正常交谈音量），无听力损伤风险。"
            "特殊人群（癫痫史、严重心脏病、重度抑郁症正在治疗者）"
            "建议在专业人员指导下使用。</p>"
            "<p>本工具为科研探索用途，不构成医疗建议或治疗手段。</p>"
            "<p>协议：Apache License 2.0</p>",
        )

    def _save_to_history(self, result: dict, audio_path) -> None:
        record = {
            "source": "upload" if audio_path else "analysis",
            "audio_path": str(audio_path) if audio_path else None,
            "duration": result.get("duration"),
            "negative": result["negative"],
            "valence": result["valence"],
            "arousal": result["arousal"],
            "quadrant": result["dominant_quadrant"],
            "asr_text": result.get("asr_text", ""),
            "asr_confidence": result.get("asr_confidence"),
            "snr_db": result["audio_quality"]["snr_db"],
            "modal_scores": result["modal_scores"],
            "memberships": result["memberships"],
            "paralang_events": result.get("paralang_events", []),
        }
        self.history.add(record)

    def closeEvent(self, event) -> None:
        try:
            self.player.stop()
            if self.manager is not None:
                self.manager.unload_all()
        except Exception:
            pass
        super().closeEvent(event)
