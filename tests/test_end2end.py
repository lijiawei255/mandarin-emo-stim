"""端到端管线测试（需要 GPU + 模型，默认跳过）。

运行方式::

    pytest tests/test_end2end.py -m gpu
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

FIXTURE = Path(__file__).parent / "fixtures" / "mandarin_sample.wav"


@pytest.fixture(scope="module")
def pipeline():
    from src.models.model_manager import ModelManager
    from src.pipeline import AnalysisPipeline
    mgr = ModelManager()
    mgr.load_all()
    return AnalysisPipeline(mgr)


def test_pipeline_returns_all_fields(pipeline):
    """端到端管线输出包含全部标准化字段。"""
    result = pipeline.analyze(str(FIXTURE))
    for key in ("negative", "valence", "arousal", "dominant_quadrant",
                "memberships", "modal_scores", "asr_text", "audio_quality",
                "duration", "paralang_events"):
        assert key in result, f"缺少字段 {key}"


def test_pipeline_values_in_range(pipeline):
    result = pipeline.analyze(str(FIXTURE))
    assert 0.0 <= result["negative"] <= 1.0
    assert 0.0 <= result["valence"] <= 1.0
    assert 0.0 <= result["arousal"] <= 1.0
    assert result["dominant_quadrant"] in ("Q1", "Q2", "Q3", "Q4")
    assert abs(sum(result["memberships"].values()) - 1.0) < 1e-6


def test_pipeline_modal_scores_six(pipeline):
    result = pipeline.analyze(str(FIXTURE))
    assert len(result["modal_scores"]) == 6


def test_pipeline_asr_text_nonempty(pipeline):
    result = pipeline.analyze(str(FIXTURE))
    assert result["asr_text"], "ASR 文本为空"


def test_pipeline_generates_stimulus(pipeline, tmp_path):
    """管线结果可驱动声刺激生成。"""
    import soundfile as sf
    from src.stimulus.generator import StimulusGenerator

    result = pipeline.analyze(str(FIXTURE))
    gen = StimulusGenerator()
    stim = gen.generate(result["valence"], result["arousal"],
                        memberships=result["memberships"], duration=10.0)
    out = tmp_path / "e2e_stim.wav"
    sf.write(str(out), stim, gen.sr, subtype="PCM_16")
    assert out.exists()
    assert stim.shape[1] == 2


def test_pipeline_progress_callback(pipeline):
    """进度回调被正确调用。"""
    stages = []

    def cb(stage, pct):
        stages.append((stage, pct))

    pipeline.analyze(str(FIXTURE), progress_cb=cb)
    assert len(stages) >= 8  # 至少 9 个阶段
    assert stages[-1][1] == 100  # 最终 100%


def test_pipeline_unknown_format_raises(pipeline, tmp_path):
    """不支持的音频格式应抛出异常。"""
    path = tmp_path / "bad.txt"
    path.write_text("not audio")
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.analyze(str(path))
