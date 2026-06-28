"""GUI 冒烟测试（offscreen 渲染，无需 GPU/模型）。

验证 MainWindow 能正常构建、渲染五块面、响应控件交互。
使用桩数据驱动界面，不加载真实模型。
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _noop_messagebox(monkeypatch):
    """全局把 QMessageBox 的模态方法替换为无操作，避免 headless 下阻塞。"""
    from PySide6.QtWidgets import QMessageBox

    class _NoOp:
        def __init__(self, *a, **k):
            pass

    for name in ("warning", "information", "critical", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: _NoOp()), raising=False)


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def window(qt_app):
    from src.gui.main_window import MainWindow
    w = MainWindow(auto_load_models=False)
    w.show()
    yield w
    w.close()


def test_main_window_constructs(window):
    """主窗口能构建并包含五块面。"""
    assert window.windowTitle().startswith("Mandarin-EmoStim")
    # 五大区域控件存在
    assert window.metric_negative is not None
    assert window.metric_valence is not None
    assert window.metric_arousal is not None
    assert window.modal_bars is not None
    assert window.waveform is not None
    assert window.status_block is not None
    assert window.player is not None


def test_metric_bar_updates(window):
    window.metric_negative.set_value(0.72)
    assert "0.72" in window.metric_negative.value_label.text()
    window.metric_valence.set_value(0.28)
    window.metric_arousal.set_value(0.61)


def test_modal_bars_update(window):
    scores = {
        "acoustic": {"negative": 0.68, "arousal": 0.5},
        "prosody": {"negative": 0.55, "arousal": 0.5},
        "text_llm": {"negative": 0.78, "arousal": 0.6},
    }
    window.modal_bars.update_scores(scores, axis="negative")
    window.modal_bars.update_events([{"name_zh": "笑声", "confidence": 0.7}])


def test_waveview_render(window):
    """波形控件能渲染合成数据。"""
    data = (np.random.randn(44100, 2) * 0.1).astype(np.float32)
    window.waveform.set_waveform(data, 44100)
    assert window.waveform.duration == pytest.approx(1.0, abs=1e-2)
    window.waveform.set_position(0.5)
    window.waveform.clear_position()


def test_status_block(window):
    window.status_block.set_status("录音中")
    window.status_block.set_mode("CUDA")
    window.status_block.set_model_progress(3, 4)
    assert "录音中" in window.status_block.status_label.text()
    assert "CUDA" in window.status_block.mode_label.text()
    assert "3/4" in window.status_block.model_label.text()


def test_analysis_result_drives_ui(window):
    """分析结果能驱动全部 UI 更新（使用桩数据）。"""
    result = {
        "negative": 0.72, "valence": 0.28, "arousal": 0.61,
        "dominant_quadrant": "Q2",
        "modal_scores": {
            "acoustic": {"negative": 0.68, "arousal": 0.7},
            "prosody": {"negative": 0.55, "arousal": 0.6},
        },
        "asr_text": "测试文本",
        "audio_quality": {"snr_db": 22.0},
        "paralang_events": [],
        "memberships": {"Q1": 0.1, "Q2": 0.7, "Q3": 0.1, "Q4": 0.1},
        "duration": 3.5,
        "asr_confidence": 0.9,
    }
    window._on_analysis_done(result)
    assert "Q2" in window.quadrant_label.text()
    assert window.asr_text.toPlainText() == "测试文本"
    assert window.btn_generate.isEnabled()


def test_player_volume(window):
    window.player.set_volume(0.5)
    assert window.player._volume == 0.5


# ====================================================================
# 录音流程测试（用桩 recorder，不依赖真实麦克风）
# ====================================================================
class _StubRecorder:
    """桩录音器：start() 不真正采集，stop() 返回预置波形。

    elapsed() 默认返回 1.5（有效时长），可被测试篡改以模拟空/短录音。
    """

    def __init__(self, sr=16000, channels=1, chunk_size=1024, device=None):
        self.sr = sr
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self._t0 = 0.0
        self._running = False
        self._elapsed = 1.5  # 默认有效时长（>= 0.5s）
        # 预置 1.5 秒波形
        self._waveform = (np.random.randn(int(sr * 1.5)) * 0.1).astype(np.float32)

    def start(self):
        import time
        self._t0 = time.time()
        self._running = True

    def stop(self):
        self._running = False
        return self._waveform

    def elapsed(self):
        return self._elapsed


@pytest.fixture
def record_window(window, monkeypatch):
    """注入桩 recorder 并拦截分析，使录音流程可无麦克风/无模型测试。"""
    import src.gui.main_window as mw

    # 确保进入测试时无遗留录音状态
    if window.record_timer.isActive():
        window.record_timer.stop()
    window.is_recording = False
    window.recorder = None
    window.btn_record.setChecked(False)
    window.btn_record.setText("开始录音")
    window.btn_upload.setEnabled(True)

    # 拦截 AnalysisWorker，避免触发真实模型推理。
    # 需提供 .progress/.finished_ok/.failed 三个「信号」（带 .connect 方法）与 .start()。
    started = {}

    class _SignalStub:
        def connect(self, *a, **k):
            pass

    class _StubWorker:
        progress = _SignalStub()
        finished_ok = _SignalStub()
        failed = _SignalStub()

        def __init__(self, manager, audio_path):
            started["path"] = audio_path
            started["called"] = True

        def start(self):
            started["started"] = True

    monkeypatch.setattr(mw, "AudioRecorder", _StubRecorder)
    monkeypatch.setattr(mw, "AnalysisWorker", _StubWorker)
    window.manager = object()  # 标记为已就绪，绕过「模型未加载」检查
    window._stub_started = started
    yield window
    # 测试后清理定时器
    if window.record_timer.isActive():
        window.record_timer.stop()


def test_record_start_sets_button_and_timer(record_window):
    """点击「开始录音」后按钮文字切换、计时器启动、上传禁用。"""
    w = record_window
    w.btn_record.setChecked(True)
    w.on_record_clicked()

    try:
        assert w.is_recording is True
        assert "停止" in w.btn_record.text()
        assert w.record_timer.isActive()
        assert w.btn_upload.isEnabled() is False
    finally:
        # 必须停止计时器并清理录音态，否则活动 QTimer 会阻塞 pytest 退出
        if w.record_timer.isActive():
            w.record_timer.stop()
        if w.recorder is not None:
            w.recorder.stop()
        w.is_recording = False
        w.recorder = None
        w.btn_record.setChecked(False)
        w.btn_record.setText("开始录音")
        w.btn_upload.setEnabled(True)


def test_record_stop_lands_wav_and_triggers_analysis(record_window, tmp_path, monkeypatch):
    """停止录音后：落盘临时 WAV 并触发分析（路径非空）。

    直接置入「正在录音」状态并调用停止逻辑，绕过 QTimer（避免 Qt 定时器
    与 pytest 事件循环交互导致的卡顿），聚焦验证波形落盘与分析衔接。
    """
    import src.portable as portable
    monkeypatch.setattr(portable, "TEMP_DIR", tmp_path)

    w = record_window
    # 直接构造录音态（不经过 start，不启动 QTimer）
    rec = _StubRecorder(sr=16000)
    w.recorder = rec
    w.is_recording = True
    w.btn_record.setText("■ 停止录音")

    # 调用停止逻辑
    w._stop_recording_and_analyze()

    assert w.is_recording is False
    assert w.btn_record.text() == "开始录音"
    assert w.btn_upload.isEnabled() is True
    # 触发了分析 worker，且传入了落盘的 wav 路径
    assert w._stub_started.get("called") is True
    path = w._stub_started.get("path", "")
    assert path.endswith(".wav")
    from pathlib import Path
    assert Path(path).exists()


def test_record_empty_shows_no_analysis(record_window):
    """空录音不触发分析（不抛异常即通过）。"""
    w = record_window
    rec = _StubRecorder(sr=16000)
    rec._waveform = rec._waveform[:0]  # 清空
    w.recorder = rec
    w.is_recording = True
    # _stop_recording_and_analyze 判空后仅 warning，不触发分析
    w._stop_recording_and_analyze()
    assert w.is_recording is False
    assert w._stub_started.get("called") is None


def test_record_auto_stop_at_max_duration(record_window, monkeypatch):
    """达到 max_duration 时计时回调自动停止录音。"""
    w = record_window
    rec = _StubRecorder(sr=16000)
    w.recorder = rec
    w.is_recording = True
    # 篡改 elapsed 让其超过上限（60s）
    monkeypatch.setattr(rec, "elapsed", lambda: 61.0)

    w._update_record_elapsed()
    # 应已自动停止
    assert w.is_recording is False
    assert w.btn_record.isChecked() is False


def test_reset_stops_active_recording(record_window):
    """重置时若正在录音，先停止（不触发分析）。"""
    w = record_window
    # 直接置入录音态（绕过 QTimer）
    w.recorder = _StubRecorder(sr=16000)
    w.is_recording = True
    w.btn_record.setText("■ 停止录音")
    w.btn_record.setChecked(True)
    assert w.is_recording is True

    w.on_reset_clicked()
    assert w.is_recording is False
    assert w.btn_record.text() == "开始录音"
    assert w.btn_record.isChecked() is False
    assert "0.0" in w.record_elapsed_label.text()
    assert "0.0" in w.record_elapsed_label.text()


def test_resolve_recorder_device_parses_id(record_window):
    """设备下拉框文本 '12: 某设备' 解析为 id=12。"""
    w = record_window
    w.device_combo.clear()
    w.device_combo.addItem("12: USB 麦克风")
    assert w._resolve_recorder_device() == 12
    w.device_combo.addItem("无设备")
    w.device_combo.setCurrentIndex(1)
    assert w._resolve_recorder_device() is None
