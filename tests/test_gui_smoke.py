"""GUI 冒烟测试（offscreen 渲染，无需 GPU/模型）。

验证 MainWindow 能正常构建、渲染五块面、响应控件交互。
使用桩数据驱动界面，不加载真实模型。
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.slow


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
