"""音频 I/O 测试（loader / vad，不含模型，无需 GPU）。"""

import numpy as np
import pytest
import soundfile as sf

from src.audio import loader, vad


@pytest.fixture
def sample_wav(tmp_path):
    """生成一个 2 秒 440Hz 测试 WAV。"""
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test.wav"
    sf.write(str(path), y, sr)
    return path, y, sr


def test_load_audio_resamples(sample_wav):
    path, _, _ = sample_wav
    y, sr = loader.load_audio(str(path), target_sr=16000)
    assert sr == 16000
    assert y.dtype == np.float32
    assert len(y) == 32000  # 2 秒 @ 16kHz


def test_load_audio_to_different_sr(sample_wav):
    path, _, _ = sample_wav
    y, sr = loader.load_audio(str(path), target_sr=8000)
    assert sr == 8000
    assert len(y) == 16000  # 2 秒 @ 8kHz


def test_load_audio_mono_mix(tmp_path):
    """立体声文件被混合为单声道。"""
    sr = 16000
    y_stereo = np.zeros((16000, 2), dtype=np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), y_stereo, sr)
    y, loaded_sr = loader.load_audio(str(path), target_sr=sr)
    assert y.ndim == 1
    assert loaded_sr == sr


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_audio(str(tmp_path / "nonexistent.wav"))


def test_load_audio_unsupported_format(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("not audio")
    with pytest.raises(ValueError):
        loader.load_audio(str(path))


def test_save_and_get_duration(tmp_path):
    sr = 16000
    y = np.zeros(16000, dtype=np.float32)
    path = loader.save_wav(y, sr, tmp_path / "out.wav")
    assert path.exists()
    assert loader.get_duration(y, sr) == 1.0


# ---------------- VAD ----------------
def test_segments_from_timestamps():
    ts = [[0, 500], [800, 1500]]
    segs = vad.segments_from_timestamps(ts)
    assert segs == [(0.0, 0.5), (0.8, 1.5)]


def test_segments_empty():
    assert vad.segments_from_timestamps([]) == []
    assert vad.segments_from_timestamps(None) == []


def test_effective_duration():
    segs = [(0.0, 0.5), (0.8, 1.5)]
    assert abs(vad.effective_duration(segs) - 1.2) < 1e-9


def test_extract_voiced():
    sr = 16000
    y = np.ones(32000, dtype=np.float32)  # 2 秒
    segs = [(0.0, 0.5), (1.0, 1.5)]  # 共 1 秒
    voiced = vad.extract_voiced(y, sr, segs)
    assert len(voiced) == 16000  # 1 秒


def test_extract_voiced_no_segments_returns_original():
    sr = 16000
    y = np.ones(32000, dtype=np.float32)
    voiced = vad.extract_voiced(y, sr, [])
    assert len(voiced) == 32000


def test_vad_from_asr_result():
    sr = 16000
    y = np.ones(32000, dtype=np.float32)
    asr_result = {"text": "test", "timestamp": [[0, 500], [1000, 2000]]}
    info = vad.vad_from_asr_result(y, sr, asr_result)
    assert len(info["segments"]) == 2
    assert info["effective_duration"] == 1.5
    assert len(info["voiced_y"]) == 24000  # 0.5s + 1s = 1.5s @ 16kHz
