"""模型加载与推理测试（需要 GPU + 已下载模型，默认跳过）。

运行方式::

    pytest tests/test_models.py -m gpu
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

FIXTURE = Path(__file__).parent / "fixtures" / "mandarin_sample.wav"


@pytest.fixture(scope="module")
def manager():
    from src.models.model_manager import ModelManager
    mgr = ModelManager()
    mgr.load_all()
    return mgr


def test_device_is_cuda(manager):
    assert manager.device == "cuda"
    assert manager.loaded_count == 4


def test_vram_within_budget(manager):
    """全部模型加载后显存占用应 < 6GB（实际 12GB 余量充裕）。"""
    vram = manager.vram_usage_mb()
    assert vram["allocated"] < 6144


def test_asr_outputs_chinese(manager):
    """ASR 对中文夹具输出非空中文。"""
    asr = manager.get_asr_model()
    result = asr.transcribe(str(FIXTURE))
    assert result["text"], "ASR 转写结果为空"
    assert result["confidence"] > 0.3


def test_emotion2vec_nine_scores(manager):
    """emotion2vec 输出 9 类置信度，聚合分在 [0,1]。"""
    emo = manager.get_emotion_model()
    result = emo.predict(str(FIXTURE))
    assert len(result["scores"]) == 9
    assert 0.0 <= result["s_acoustic"] <= 1.0
    assert 0.0 <= result["a_acoustic"] <= 1.0


def test_llm_outputs_two_floats(manager):
    """LLM 输出两个可解析的浮点数。"""
    llm = manager.get_llm_model()
    result = llm.analyze_text("我今天非常开心，天气真好！")
    assert "s_text_llm" in result
    assert "a_text_llm" in result
    assert 0.0 <= result["s_text_llm"] <= 1.0
    assert 0.0 <= result["a_text_llm"] <= 1.0


def test_llm_fallback_on_garbage(manager):
    """空文本时 LLM 降级返回中性分。"""
    llm = manager.get_llm_model()
    result = llm.analyze_text("")
    assert result["fallback"] is True
    assert result["s_text_llm"] == 0.5


def test_panns_detect_no_crash(manager):
    """PANNs 对夹具推理不报错，返回事件列表与聚合分。"""
    import soundfile as sf
    import librosa
    y, sr = sf.read(str(FIXTURE))
    # PANNs 期望 32000Hz
    if sr != 32000:
        y = librosa.resample(y.astype("f"), orig_sr=sr, target_sr=32000)
        sr = 32000
    pann = manager.get_pann_model()
    result = pann.detect(y, sr)
    assert "events" in result
    assert 0.0 <= result["s_paralang"] <= 1.0
    assert 0.0 <= result["a_paralang"] <= 1.0
