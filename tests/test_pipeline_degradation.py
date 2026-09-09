"""管线降级测试：非关键模态失败时不崩溃，返回中性分并标记降级。

用桩 ModelManager 驱动 AnalysisPipeline，不加载真实模型。
"""

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "mandarin_sample.wav"


class _StubASR:
    def transcribe(self, path):
        return {"text": "今天天气不错我很开心", "confidence": 0.9,
                "timestamp": [[0, 300], [300, 600], [600, 900], [900, 1200],
                              [1200, 1500], [1500, 1800], [1800, 2100],
                              [2100, 2400], [2400, 2700], [2700, 3000]]}


class _StubEmotion:
    def predict(self, path):
        return {"scores": [0.0] * 9, "labels": [], "s_acoustic": 0.3, "a_acoustic": 0.6}


class _StubPANN:
    def detect(self, y, sr):
        return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5}


class _StubLLM:
    def analyze_text(self, text):
        return {"s_text_llm": 0.2, "a_text_llm": 0.7, "raw": "0.2 0.7", "fallback": False}


class _Failing:
    """任何方法调用都抛异常。"""
    def __getattr__(self, name):
        def _boom(*a, **k):
            raise RuntimeError(f"stub failure in {name}")
        return _boom


class _StubManager:
    def __init__(self, asr=None, emotion=None, pann=None, llm=None):
        self._asr = asr or _StubASR()
        self._emotion = emotion or _StubEmotion()
        self._pann = pann or _StubPANN()
        self._llm = llm or _StubLLM()

    def get_asr_model(self):
        return self._asr

    def get_emotion_model(self):
        return self._emotion

    def get_pann_model(self):
        return self._pann

    def get_llm_model(self):
        return self._llm


@pytest.fixture
def make_pipeline():
    from src.pipeline import AnalysisPipeline

    def _make(**kw):
        return AnalysisPipeline(_StubManager(**kw))
    return _make


def test_all_stubs_ok_no_degradation(make_pipeline):
    result = make_pipeline().analyze(str(FIXTURE))
    assert result["degraded_modalities"] == []
    assert 0.0 <= result["negative"] <= 1.0


def test_emotion_failure_degrades_not_crashes(make_pipeline):
    """emotion2vec 失败：结果仍返回，acoustic 标记降级且为中性分。"""
    result = make_pipeline(emotion=_Failing()).analyze(str(FIXTURE))
    assert result["degraded_modalities"] == ["acoustic"]
    assert result["modal_scores"]["acoustic"] == {"negative": 0.5, "arousal": 0.5}
    assert 0.0 <= result["negative"] <= 1.0
    assert abs(sum(result["memberships"].values()) - 1.0) < 1e-6


def test_asr_failure_degrades_text_modalities(make_pipeline):
    """ASR 失败：文本两模态降级，声学模态照常，管线不抛异常。"""
    result = make_pipeline(asr=_Failing()).analyze(str(FIXTURE))
    assert "text_llm" in result["degraded_modalities"]
    assert "text_stat" in result["degraded_modalities"]
    assert result["asr_text"] == ""
    assert result["asr_confidence"] == 0.0
    # 声学模态仍来自桩模型
    assert result["modal_scores"]["acoustic"]["negative"] == pytest.approx(0.3)


def test_multiple_failures_listed_in_order(make_pipeline):
    result = make_pipeline(emotion=_Failing(), pann=_Failing(), llm=_Failing()).analyze(str(FIXTURE))
    assert result["degraded_modalities"] == ["acoustic", "paralang", "text_llm"]


def test_missing_audio_still_fatal(make_pipeline):
    """加载音频失败仍是致命错误（无数据可分析）。"""
    with pytest.raises(FileNotFoundError):
        make_pipeline().analyze("definitely_missing.wav")
