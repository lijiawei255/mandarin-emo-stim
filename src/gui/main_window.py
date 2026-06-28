"""浅色包豪斯（Light Bauhaus）风格主窗口。

设计原则：功能优先（去除冗余装饰）、几何网格（8px 对齐）、高对比度
（浅色背景 + 深色文字）、有限色彩（主色 Bauhaus 蓝 #1F5FA8 + 中性灰阶；
状态色语义化：成功绿 / 警告琥珀 / 错误红）。

五大功能分区（卡片式，1px 细线分隔，无厚色块）：
    左   : 音频输入控制（开始/上传/设备选择/录制计时）
    中   : 核心情感量化展示（NEGATIVE/VALENCE/AROUSAL + 象限 + ASR）
    右   : 多模态分解详情（6 模态分项条形图 + 副语言事件）
    底   : 声刺激生成与波形（按钮 + 波形 + 音量）
    底栏 : 状态指示（运行状态 / 推理模式 / 模型进度）

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
from src.audio.loader import save_wav
from src.audio.recorder import AudioRecorder
from src.config_loader import load_settings
from src.gui.threads import (AnalysisWorker, ModelLoadWorker, StimulusWorker)
from src.gui.widgets.loading_overlay import LoadingOverlay
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

        # 录音状态
        self.recorder: AudioRecorder | None = None
        self.is_recording = False
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self._update_record_elapsed)

        # 工作线程强引用（避免被 GC 导致 QThread destroyed 崩溃）
        self.model_worker: ModelLoadWorker | None = None
        self.analysis_worker = None
        self.stim_worker = None

        # 模型加载进度浮层
        self.loading_overlay = LoadingOverlay(self)

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

    # ----- 音频输入控制区 -----
    def _build_red_panel(self) -> QWidget:
        panel = self._panel_frame("RedPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(self._h1("音频输入"))

        self.btn_record = QPushButton("开始录音")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setCheckable(True)
        self.btn_upload = QPushButton("上传音频")
        self.btn_reset = QPushButton("停止 / 重置")
        for b in (self.btn_record, self.btn_upload, self.btn_reset):
            layout.addWidget(b)

        self.device_combo = QComboBox()
        self.device_combo.addItems(self._list_input_devices())
        layout.addWidget(self.device_combo)

        # 录制时长显示
        self.record_elapsed_label = QLabel("已录制：0.0 秒")
        self.record_elapsed_label.setProperty("role", "success")
        layout.addWidget(self.record_elapsed_label)

        self.snr_warning = QLabel("")
        self.snr_warning.setProperty("role", "warning")  # 警告色
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

    # ----- 核心指标区 -----
    def _build_main_panel(self) -> QWidget:
        panel = self._panel_frame("MainPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        self.metric_negative = MetricBar("NEGATIVE SCORE", "negative")
        self.metric_valence = MetricBar("VALENCE", "valence")
        self.metric_arousal = MetricBar("AROUSAL", "arousal")
        layout.addWidget(self.metric_negative)
        layout.addWidget(self.metric_valence)
        layout.addWidget(self.metric_arousal)

        self.quadrant_label = QLabel("情绪象限：—")
        self.quadrant_label.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #1F5FA8; "
            "padding: 8px 0; border: none;"
        )
        layout.addWidget(self.quadrant_label)

        layout.addWidget(self._h1("ASR 转写"))
        self.asr_text = QTextEdit()
        self.asr_text.setReadOnly(True)
        self.asr_text.setFixedHeight(72)
        self.asr_text.setPlaceholderText("（转写结果将显示于此）")
        layout.addWidget(self.asr_text)
        layout.addStretch()
        return panel

    # ----- 多模态分解区 -----
    def _build_blue_panel(self) -> QWidget:
        panel = self._panel_frame("BluePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        self.modal_bars = ModalBars()
        layout.addWidget(self.modal_bars)
        return panel

    # ----- 声刺激区 -----
    def _build_yellow_panel(self) -> QWidget:
        panel = self._panel_frame("YellowPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.btn_generate = QPushButton("生成刺激波形")
        self.btn_generate.setObjectName("btn_generate")  # 主色强调
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
        self.param_label.setStyleSheet(
            "color: #6A6A6A; font-size: 12px; border: none;"
        )
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
        """节区小标题（主色大写，下划线分隔，包豪斯风格）。"""
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #
    def _connect_workers_slots(self) -> None:
        # 占位：worker 信号在创建时连接
        pass

    def start_model_loading(self) -> None:
        self.status_block.set_status("模型加载中…")
        # 显示加载进度浮层
        self.loading_overlay.show_loading()
        # 强引用 worker，避免被 GC 导致 QThread destroyed
        self.model_worker = ModelLoadWorker(config=self.config)
        self.model_worker.progress.connect(self._on_model_progress)
        self.model_worker.finished_ok.connect(self._on_models_loaded)
        self.model_worker.failed.connect(self._on_model_failed)
        self.model_worker.interrupted.connect(self._on_model_interrupted)
        self.model_worker.start()

    def _on_model_progress(self, stage: str, pct: int) -> None:
        self.status_block.set_status(f"加载：{stage} {pct}%")
        self.status_block.set_model_progress(min(3, pct // 25), 4)
        # 更新浮层
        self.loading_overlay.update_progress(stage, pct)

    def _on_models_loaded(self, manager) -> None:
        self.manager = manager
        self.status_block.set_status("就绪")
        self.status_block.set_mode(manager.get_device().upper())
        self.status_block.set_model_progress(manager.loaded_count, 4)
        # 隐藏浮层
        self.loading_overlay.show_done()
        self.btn_record.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_generate.setEnabled(False)  # 待分析完成

    def _on_model_interrupted(self) -> None:
        self.status_block.set_status("模型加载已中断")
        self.loading_overlay.show_failed("用户中断了模型加载")
        self.btn_record.setEnabled(False)
        self.btn_upload.setEnabled(False)

    def _on_model_failed(self, msg: str) -> None:
        self.status_block.set_status("模型加载失败")
        self.loading_overlay.show_failed(msg)
        QMessageBox.critical(self, "模型加载失败", msg)

    def on_record_clicked(self) -> None:
        """开始 / 停止 录音（切换式）。"""
        if self.btn_record.isChecked():
            self._start_recording()
        else:
            self._stop_recording_and_analyze()

    def _resolve_recorder_device(self):
        """从设备下拉框解析设备 id（返回 int 或 None）。"""
        text = self.device_combo.currentText()
        if text and ":" in text:
            try:
                # 文本格式 "id: name"，取冒号前整数
                return int(text.split(":", 1)[0].strip())
            except ValueError:
                return None
        return None

    def _start_recording(self) -> None:
        """开始录音。"""
        if self.manager is None:
            QMessageBox.warning(self, "未就绪", "模型尚未加载完成，请稍候。")
            self.btn_record.setChecked(False)
            return

        audio_cfg = self.config["audio"]
        device = self._resolve_recorder_device()
        try:
            self.recorder = AudioRecorder(
                sr=int(audio_cfg["sample_rate"]),
                channels=1,
                chunk_size=int(audio_cfg["chunk_size"]),
                device=device,
            )
            self.recorder.start()
        except ImportError as e:
            # sounddevice 等录音依赖缺失
            logger.exception("录音依赖缺失")
            QMessageBox.critical(
                self, "录音依赖缺失",
                f"录音所需的依赖未安装：{e}\n\n"
                "请确认已激活 mandarin-emo-stim 环境并执行：\n"
                "    pip install -r requirements.txt\n\n"
                "（不影响「上传音频」分析功能）",
            )
            self.recorder = None
            self.btn_record.setChecked(False)
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("录音启动失败")
            QMessageBox.critical(self, "录音失败",
                                 f"无法启动录音：{e}\n请检查麦克风设备与系统权限设置。")
            self.recorder = None
            self.btn_record.setChecked(False)
            return

        self.is_recording = True
        self.btn_record.setText("■ 停止录音")
        # 录音中禁用上传，避免并发冲突
        self.btn_upload.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.record_elapsed_label.setText("已录制：0.0 秒")
        self.status_block.set_status("录音中…")
        # 每 100ms 刷新计时显示
        self.record_timer.start(100)

    def _stop_recording_and_analyze(self) -> None:
        """停止录音，落盘临时 WAV 并触发分析。"""
        self.record_timer.stop()
        if self.recorder is None or not self.is_recording:
            return
        self.is_recording = False
        self.btn_record.setText("开始录音")
        self.btn_upload.setEnabled(True)

        waveform = self.recorder.stop()
        sr = self.recorder.sr
        elapsed = self.recorder.elapsed()
        self.recorder = None

        if len(waveform) == 0:
            self.status_block.set_status("就绪")
            QMessageBox.warning(self, "录音为空", "未采集到音频数据，请检查麦克风。")
            return

        # 过短录音（<0.5s）视为无效
        if elapsed < 0.5:
            self.status_block.set_status("就绪")
            QMessageBox.warning(self, "录音过短", "录音时长不足 0.5 秒，请重新录制。")
            return

        # 落盘临时 WAV（便携目录下），再交给分析 worker（接受文件路径）
        portable.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_wav = portable.TEMP_DIR / f"rec_{ts}.wav"
        try:
            save_wav(waveform, sr, tmp_wav)
        except OSError as e:
            logger.exception("录音落盘失败")
            self.status_block.set_status("就绪")
            QMessageBox.critical(self, "保存失败",
                                 f"录音无法保存：{e}\n可能是磁盘已满或目录无写权限。")
            return
        logger.info("录音已落盘: %s（%.2fs）", tmp_wav, elapsed)
        self.status_block.set_status(f"录音完成（{elapsed:.1f}s），开始分析…")
        self._start_analysis(str(tmp_wav), source="record")

    def _update_record_elapsed(self) -> None:
        """刷新录音计时显示，到上限自动停止。"""
        if self.recorder is None or not self.is_recording:
            return
        elapsed = self.recorder.elapsed()
        max_dur = float(self.config["audio"]["record_duration_max"])
        self.record_elapsed_label.setText(f"已录制：{elapsed:.1f} 秒（上限 {max_dur:.0f}s）")
        if elapsed >= max_dur:
            logger.info("达到录音时长上限 %.0fs，自动停止", max_dur)
            self.btn_record.setChecked(False)
            self._stop_recording_and_analyze()

    def on_upload_clicked(self) -> None:
        if self.manager is None:
            QMessageBox.warning(self, "未就绪", "模型尚未加载完成，请稍候。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a)",
        )
        if path:
            self._start_analysis(path, source="upload")

    def on_reset_clicked(self) -> None:
        # 若正在录音，先停止（不触发分析）
        if self.is_recording:
            self.record_timer.stop()
            if self.recorder is not None:
                self.recorder.stop()
                self.recorder = None
            self.is_recording = False
            self.btn_record.setText("开始录音")
            self.btn_record.setChecked(False)
            self.btn_upload.setEnabled(True)
            self.record_elapsed_label.setText("已录制：0.0 秒")
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

    def _start_analysis(self, audio_path: str, source: str = "upload") -> None:
        # 并发防护：若已有分析在运行，拒绝重复触发（避免孤儿线程崩溃）
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            logger.warning("已有分析任务在运行，忽略重复触发")
            return
        self._pending_audio_path = audio_path
        self._pending_source = source
        self.status_block.set_status("分析中…")
        # 分析期间禁用所有输入按钮，防止并发状态污染
        self.btn_generate.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.analysis_worker = AnalysisWorker(self.manager, audio_path)
        self.analysis_worker.progress.connect(
            lambda s, p: self.status_block.set_status(f"分析：{s} {p}%")
        )
        self.analysis_worker.finished_ok.connect(self._on_analysis_done)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.interrupted.connect(self._on_analysis_interrupted)
        self.analysis_worker.start()

    def _reenable_input_buttons(self) -> None:
        """分析结束（成功/失败/中断）后恢复输入按钮。"""
        self.btn_record.setEnabled(True)
        self.btn_upload.setEnabled(True)

    def _on_analysis_interrupted(self) -> None:
        self.status_block.set_status("已中断")
        self._reenable_input_buttons()

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
        self._reenable_input_buttons()

        # 存入历史（录制的音频归档到 history/audio，上传的仅记录路径）
        try:
            audio_path = getattr(self, "_pending_audio_path", None)
            source = getattr(self, "_pending_source", "upload")
            if source == "record" and audio_path:
                # 把临时录音归档到 history/audio
                import numpy as np
                import soundfile as sf
                y, sr = sf.read(audio_path)
                archived = self.history.save_audio(
                    np.ascontiguousarray(y.astype(np.float32)), sr, kind="rec"
                )
                audio_path = str(archived)
            self._save_to_history(result, audio_path=audio_path, source=source)
        except HistoryFull:
            QMessageBox.warning(self, "历史记录已满",
                                f"已达上限（{self.history.max_records} 条），请先导出并清空。")
        except OSError as e:
            # 历史/音频归档失败不阻塞结果展示，仅记录日志
            logger.exception("历史记录写入失败")
            self.snr_warning.setText(f"⚠ 历史记录保存失败：{e}")

    def _on_analysis_failed(self, msg: str) -> None:
        self.status_block.set_status("分析失败")
        self._reenable_input_buttons()
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
            try:
                import soundfile as sf
                sf.write(path, self.last_stimulus, self.player.sr, subtype="PCM_16")
                QMessageBox.information(self, "保存成功", f"已保存到：{path}")
            except OSError as e:
                logger.exception("刺激音频保存失败")
                QMessageBox.critical(self, "保存失败",
                                     f"无法保存：{e}\n可能是磁盘已满或路径无写权限。")

    def on_export_history(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出历史记录", "history.json",
                                              "JSON (*.json);;CSV (*.csv)")
        if path:
            fmt = "csv" if path.lower().endswith(".csv") else "json"
            try:
                self.history.export(Path(path), fmt=fmt)
                QMessageBox.information(self, "导出完成", f"已导出到：{path}")
            except OSError as e:
                logger.exception("历史记录导出失败")
                QMessageBox.critical(self, "导出失败",
                                     f"无法导出：{e}\n可能是磁盘已满或路径无写权限。")

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

    def _save_to_history(self, result: dict, audio_path, source: str = "upload") -> None:
        record = {
            "source": source,
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
            # 关键：若有工作线程仍在运行，先请求中断并等待其退出，
            # 否则 QThread 被销毁时会触发 "Destroyed while thread is still running" 崩溃。
            for worker in (self.model_worker, self.analysis_worker, self.stim_worker):
                if worker is not None and worker.isRunning():
                    worker.requestInterruption()
                    worker.quit()
                    worker.wait(3000)  # 最多等 3 秒

            if self.is_recording:
                self.record_timer.stop()
                if self.recorder is not None:
                    self.recorder.stop()
            self.player.stop()
            if self.manager is not None:
                self.manager.unload_all()
        except Exception:
            pass
        super().closeEvent(event)
