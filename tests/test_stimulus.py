"""声刺激生成测试。"""


import numpy as np
import pytest
import soundfile as sf

from src.config_loader import load_stimulus_params
from src.fusion.quadrant import compute_quadrant_memberships
from src.stimulus.generator import StimulusGenerator
from src.stimulus.strategies import compute_params


@pytest.fixture
def generator():
    return StimulusGenerator()


@pytest.fixture
def stim_config():
    return load_stimulus_params()


# ---------------- params ----------------
def test_params_in_range(stim_config):
    """各象限下计算出的声学参数应落在合法映射区间内。"""
    ranges = stim_config["mapping_ranges"]
    for v in (0.1, 0.4, 0.6, 0.9):
        for a in (0.1, 0.4, 0.6, 0.9):
            m = compute_quadrant_memberships(v, a)
            p = compute_params(v, a, m, stim_config)
            assert ranges["f0"]["min"] <= p.f0 <= ranges["f0"]["max"]
            assert ranges["pr"]["min"] <= p.pr <= ranges["pr"]["max"]
            assert ranges["loud_db"]["min"] <= p.loud_db <= ranges["loud_db"]["max"]
            assert ranges["sc"]["min"] <= p.sc <= ranges["sc"]["max"]
            assert ranges["attack_ms"]["min"] <= p.attack_ms <= ranges["attack_ms"]["max"]
            assert ranges["mod_depth"]["min"] <= p.mod_depth <= ranges["mod_depth"]["max"]
            assert 0.0 <= p.noise_ratio <= ranges["noise_ratio"]["max"]


def test_params_harmony_freqs(stim_config):
    m = {"Q1": 1, "Q2": 0, "Q3": 0, "Q4": 0}
    p = compute_params(0.8, 0.8, m, stim_config)
    assert len(p.freqs) == len(p.amps)
    assert len(p.freqs) > 0
    # 大三和弦（Q1）应有 3 个频率分量
    assert len(p.freqs) == 3


# ---------------- generator output ----------------
def test_generate_shape_and_dtype(generator):
    # 使用合法时长（>= stimulus_min_sec=10）
    dur = generator.config["audio"]["stimulus_min_sec"]
    wave = generator.generate(0.5, 0.5, duration=dur)
    assert wave.ndim == 2
    assert wave.shape[1] == 2  # 双声道
    assert wave.dtype == np.float32
    sr = generator.config["audio"]["stimulus_sample_rate"]
    assert abs(wave.shape[0] - sr * dur) <= 1


def test_generate_peak_below_safety(generator):
    """峰值不超过 -10 dBFS（安全限幅）。"""
    for v in (0.0, 0.5, 1.0):
        for a in (0.0, 0.5, 1.0):
            wave = generator.generate(v, a, duration=0.5)
            peak = np.max(np.abs(wave))
            max_peak_amp = 10 ** (-10 / 20.0)
            # 因叠加了 0.7 系数与限幅，峰值应 <= -10dBFS 等效
            assert peak <= max_peak_amp + 1e-6


def test_generate_q2_slows_pulse_at_high_arousal(generator, stim_config):
    """Q2（焦虑）高 arousal 时脉冲率应较慢（引导呼吸放缓，反线性）。"""
    m_low = {"Q2": 1, "Q1": 0, "Q3": 0, "Q4": 0}
    m_low = {k: float(v) for k, v in m_low.items()}
    p_low_a = compute_params(0.2, 0.1, m_low, stim_config)
    p_high_a = compute_params(0.2, 0.9, m_low, stim_config)
    assert p_high_a.pr < p_low_a.pr


def test_generate_q3_speeds_pulse_at_high_arousal(generator, stim_config):
    """Q3（抑郁）高 arousal 时脉冲率应较快（激活，正线性）。"""
    m = {"Q3": 1.0, "Q1": 0.0, "Q2": 0.0, "Q4": 0.0}
    p_low_a = compute_params(0.2, 0.1, m, stim_config)
    p_high_a = compute_params(0.2, 0.9, m, stim_config)
    assert p_high_a.pr > p_low_a.pr


def test_generate_duration_clamped(generator):
    """时长被约束到 [min, max] 区间。"""
    audio = generator.config["audio"]
    wave_short = generator.generate(0.5, 0.5, duration=1.0)  # 低于 min
    sr = generator.sr
    expected = audio["stimulus_min_sec"]
    assert abs(wave_short.shape[0] - sr * expected) <= 1


def test_generate_default_duration(generator):
    """未指定时长时使用配置默认值。"""
    audio = generator.config["audio"]
    wave = generator.generate(0.5, 0.5)
    assert abs(wave.shape[0] - generator.sr * audio["stimulus_duration_sec"]) <= 1


def test_generate_no_nans(generator):
    for v in (0.0, 0.5, 1.0):
        for a in (0.0, 0.5, 1.0):
            wave = generator.generate(v, a, duration=0.5)
            assert np.all(np.isfinite(wave))


def test_wav_roundtrip(generator, tmp_path):
    """合成波形写为 WAV 再读回，数据一致（PCM_16 量化误差内）。"""
    wave = generator.generate(0.5, 0.5, duration=1.0)
    out = tmp_path / "stim.wav"
    sf.write(str(out), wave, generator.sr, subtype="PCM_16")
    data, sr = sf.read(str(out))
    assert sr == generator.sr
    assert data.shape == wave.shape
    # PCM_16 量化误差 < 1/32767
    assert np.max(np.abs(data - wave)) < 1e-3


# ---------------- 响度 / 限幅 语义一致性 ----------------
def _peak_db(w):
    return 20 * np.log10(np.max(np.abs(w)) + 1e-12)


def _rms_db(w):
    return 20 * np.log10(np.sqrt(np.mean(w[:, 0] ** 2)) + 1e-12)


def test_peak_limiter_actually_engages(generator, stim_config):
    """loud_db 取上限 -10 时，峰值恰好被限幅到 max_peak_dbfs（-10 dBFS）。

    此前实现按峰值归一后再统一 ×0.7，限幅永不触发，实际峰值恒为 -13 dBFS，
    与文档「-10 dBFS 限幅」不符。
    """
    from src.stimulus.strategies import compute_params
    from src.stimulus.synthesizer import synthesize
    m = compute_quadrant_memberships(0.9, 1.0)
    p = compute_params(0.9, 1.0, m, stim_config)
    assert p.loud_db == pytest.approx(-10.0)
    w = synthesize(p, 2.0, generator.sr, generator.stimulus_config)
    max_peak = generator.stimulus_config["max_peak_dbfs"]
    assert _peak_db(w) == pytest.approx(max_peak, abs=0.05)


def test_peak_never_exceeds_config_ceiling(generator):
    max_peak = generator.stimulus_config["max_peak_dbfs"]
    for v in (0.0, 0.5, 1.0):
        for a in (0.0, 0.5, 1.0):
            w = generator.generate(v, a, duration=1.0)
            assert _peak_db(w) <= max_peak + 0.05


def test_loudness_is_rms_based_and_monotonic(generator, stim_config):
    """loud_db 越高 RMS 越高（响度按 RMS 归一，非峰值）。"""
    from src.stimulus.strategies import compute_params
    from src.stimulus.synthesizer import synthesize
    m = {"Q4": 1.0, "Q1": 0.0, "Q2": 0.0, "Q3": 0.0}
    rms_levels = []
    for a in (0.0, 0.3, 0.6):
        p = compute_params(0.8, a, m, stim_config)
        w = synthesize(p, 2.0, generator.sr, generator.stimulus_config)
        rms_levels.append(_rms_db(w))
    assert rms_levels[0] < rms_levels[1] < rms_levels[2]


def test_generator_merges_settings_stimulus_into_config(generator):
    """generator 把 settings.stimulus（max_peak_dbfs/fade/haas）并入合成配置。"""
    for k in ("max_peak_dbfs", "fade_ms", "haas_delay_ms"):
        assert k in generator.stimulus_config
