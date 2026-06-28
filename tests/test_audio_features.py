"""音频特征（物理声学 + 韵律学）测试，使用合成信号。"""

import numpy as np
import pytest

from src.features import physical, prosody


@pytest.fixture
def sr():
    return 16000


@pytest.fixture
def pure_tone(sr):
    """440Hz 纯音（干净信号，高 SNR）。"""
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    return 0.2 * np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def noisy_signal(sr):
    """440Hz 纯音 + 强噪声（低 SNR）。"""
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    rng = np.random.default_rng(42)
    return 0.05 * np.sin(2 * np.pi * 440 * t) + 0.3 * rng.standard_normal(len(t)), sr


@pytest.fixture
def silence(sr):
    return np.zeros(int(sr * 1.0), dtype=np.float64), sr


# ---------------- physical ----------------
def test_physical_extract_returns_features(pure_tone):
    y, sr = pure_tone
    feat = physical.extract(y, sr)
    assert feat.rms > 0
    assert feat.spectral_centroid > 0
    assert 0.0 <= feat.hf_energy_ratio <= 1.0


def test_physical_snr_pure_higher_than_noisy(pure_tone, noisy_signal):
    """纯音 SNR 应明显高于含噪信号。"""
    clean_feat = physical.extract(*pure_tone)
    noisy_feat = physical.extract(*noisy_signal)
    assert clean_feat.snr_db > noisy_feat.snr_db


def test_physical_score_in_range(pure_tone):
    y, sr = pure_tone
    feat = physical.extract(y, sr)
    s, a, detail = physical.score(feat)
    assert 0.0 <= s <= 1.0
    assert 0.0 <= a <= 1.0
    assert detail["snr_db"] == feat.snr_db


def test_physical_empty_safe():
    """空音频不报错。"""
    feat = physical.extract(np.array([]), 16000)
    s, a, _ = physical.score(feat)
    assert 0.0 <= s <= 1.0


# ---------------- prosody ----------------
def test_prosody_pure_tone_detects_f0(pure_tone):
    """对 440Hz 纯音应检测出接近 440Hz 的 F0。"""
    y, sr = pure_tone
    feat = prosody.extract(y, sr)
    # parselmouth pitch_ceiling=500，440 应能被检出
    assert 400 < feat.mean_f0 < 500


def test_prosody_score_in_range(pure_tone):
    y, sr = pure_tone
    feat = prosody.extract(y, sr)
    s, a, detail = prosody.score(feat)
    assert 0.0 <= s <= 1.0
    assert 0.0 <= a <= 1.0


def test_prosody_silence_safe(silence):
    """静音音频不报错，返回中性。"""
    y, sr = silence
    feat = prosody.extract(y, sr)
    s, a, _ = prosody.score(feat)
    assert 0.0 <= s <= 1.0
    assert 0.0 <= a <= 1.0


def test_prosody_short_safe(sr):
    """极短音频不报错。"""
    y = np.zeros(100, dtype=np.float64)
    feat = prosody.extract(y, sr)
    s, a, _ = prosody.score(feat)
    assert 0.0 <= s <= 1.0
