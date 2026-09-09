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


# ---------------- prosody: 语速来自 ASR 时间戳、F0 斜率、HNR 负值 ----------------
def test_syllable_rate_from_asr_timestamps():
    """字数 / 时间戳覆盖的语音时长 = 音节率。3 字覆盖 0.6 s → 5.0。"""
    from src.audio import vad
    asr = {"text": "你好吗", "timestamp": [[0, 200], [200, 400], [400, 600]]}
    assert vad.syllable_rate(asr) == pytest.approx(5.0)


def test_syllable_rate_ignores_punctuation_and_handles_empty():
    from src.audio import vad
    asr = {"text": "你好，吗？", "timestamp": [[0, 300], [300, 600], [600, 900]]}
    assert vad.syllable_rate(asr) == pytest.approx(3 / 0.9)
    assert vad.syllable_rate({"text": "", "timestamp": []}) is None
    assert vad.syllable_rate({"text": "你好", "timestamp": []}) is None


def test_prosody_extract_uses_external_syllable_rate(pure_tone):
    y, sr = pure_tone
    feat = prosody.extract(y, sr, syllable_rate=5.0)
    assert feat.speech_rate == pytest.approx(5.0)
    # 未提供时回退到能量法估计（非负）
    feat2 = prosody.extract(y, sr)
    assert feat2.speech_rate >= 0.0


def _glide(sr, f_start, f_end, dur=1.5, amp=0.2):
    """线性滑音（F0 从 f_start 到 f_end），相位连续。"""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f_inst = f_start + (f_end - f_start) * t / dur
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    return amp * np.sin(phase)


def test_prosody_f0_slope_sign(sr):
    """F0 下滑 → f0_slope < 0；上升 → > 0（Hz/s，线性回归斜率）。"""
    falling = prosody.extract(_glide(sr, 300, 200), sr)
    rising = prosody.extract(_glide(sr, 200, 300), sr)
    assert falling.f0_slope < -20
    assert rising.f0_slope > 20
    # 斜率量级应接近真实值 ±100 Hz / 1.5 s ≈ ±67 Hz/s
    assert abs(abs(falling.f0_slope) - 66.7) < 25


def test_prosody_score_uses_f0_slope_for_negative():
    """其它指标相同时，F0 下滑越陡负面分越高（n_f0_drop 用真实斜率）。"""
    base = dict(mean_f0=180.0, std_f0=25.0, f0_range=80.0, speech_rate=4.5,
                pause_ratio=0.25, hnr=15.0, jitter_local=0.02,
                shimmer_local=0.08, duration=2.0)
    flat = prosody.ProsodyFeatures(f0_slope=0.0, **base)
    falling = prosody.ProsodyFeatures(f0_slope=-60.0, **base)
    s_flat, _, _ = prosody.score(flat)
    s_fall, _, _ = prosody.score(falling)
    assert s_fall > s_flat


def test_prosody_hnr_keeps_negative_frames(monkeypatch):
    """HNR 聚合不再丢弃合法的负值帧（仅排除 parselmouth 的 -200 无定义哨兵）。"""
    vals = np.array([[-200.0, -3.0, 5.0, -200.0, 2.0]])
    assert prosody._aggregate_hnr(vals) == pytest.approx(np.mean([-3.0, 5.0, 2.0]))
    assert prosody._aggregate_hnr(np.array([[-200.0, -200.0]])) == 0.0
