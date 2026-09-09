"""科学行为回归测试（合成受控信号，无需外部数据，可进 CI）。

每个用例检验一个模块对已知声学操控的**响应方向**是否与文档所述的
心理声学 / 情绪声学依据一致。这些断言不证明方法「准确」，只保证
后续改动不会悄悄反转方法学方向。
"""

import numpy as np
import pytest

from src.config_loader import load_stimulus_params
from src.features import physical, prosody
from src.fusion.quadrant import compute_quadrant_memberships, dominant_quadrant
from src.stimulus.strategies import compute_params

SR = 16000


def _tone(f0: float, dur: float = 1.5, amp: float = 0.2,
          harmonics: int = 3, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """带泛音的「类语音」稳态音 + 可选白噪。"""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    y = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        y += (amp / k) * np.sin(2 * np.pi * f0 * k * t)
    if noise > 0:
        y += noise * np.random.default_rng(seed).standard_normal(len(t))
    return y


# ---------------- 韵律：唤醒度方向 ----------------
def test_higher_f0_raises_prosodic_arousal():
    """F0 升高（其他不变）→ a_prosody 升高（Juslin & Laukka 2003：高唤醒情绪 F0 更高）。"""
    _, a_low, _ = prosody.score(prosody.extract(_tone(150), SR, syllable_rate=4.5))
    _, a_high, _ = prosody.score(prosody.extract(_tone(300), SR, syllable_rate=4.5))
    assert a_high > a_low


def test_faster_syllable_rate_raises_prosodic_arousal():
    y = _tone(200)
    _, a_slow, _ = prosody.score(prosody.extract(y, SR, syllable_rate=2.5))
    _, a_fast, _ = prosody.score(prosody.extract(y, SR, syllable_rate=7.0))
    assert a_fast > a_slow


# ---------------- 韵律：负面方向 ----------------
def test_lower_hnr_raises_prosodic_negativity():
    """加噪降低 HNR → s_prosody 升高（嗓音粗糙/气声与负面情绪相关）。"""
    clean = prosody.extract(_tone(200), SR, syllable_rate=4.5)
    noisy = prosody.extract(_tone(200, noise=0.05), SR, syllable_rate=4.5)
    assert noisy.hnr < clean.hnr
    s_clean, _, _ = prosody.score(clean)
    s_noisy, _, _ = prosody.score(noisy)
    assert s_noisy > s_clean


# ---------------- 物理声学：粗糙度 ----------------
def test_beating_dyad_is_rougher_than_pure_tone():
    """拍频 70 Hz 的双音 roughness 高于纯音（Plomp & Levelt 1965 / Sethares 1993）。"""
    t = np.linspace(0, 1.5, int(SR * 1.5), endpoint=False)
    pure = 0.2 * np.sin(2 * np.pi * 440 * t)
    dyad = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 510 * t)
    assert physical.extract(dyad, SR).roughness > physical.extract(pure, SR).roughness


def test_roughness_raises_physical_negativity():
    t = np.linspace(0, 1.5, int(SR * 1.5), endpoint=False)
    pure = 0.2 * np.sin(2 * np.pi * 440 * t)
    dyad = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 510 * t)
    s_pure, _, _ = physical.score(physical.extract(pure, SR))
    s_dyad, _, _ = physical.score(physical.extract(dyad, SR))
    assert s_dyad > s_pure


# ---------------- 象限 ----------------
@pytest.mark.parametrize("v,a,expected", [
    (0.9, 0.9, "Q1"), (0.1, 0.9, "Q2"), (0.1, 0.1, "Q3"), (0.9, 0.1, "Q4"),
])
def test_quadrant_corners(v, a, expected):
    assert dominant_quadrant(compute_quadrant_memberships(v, a)) == expected


def test_quadrant_soft_band_is_continuous():
    """跨过中线时隶属度连续变化，无跳变（软判定的存在意义）。"""
    prev = None
    for v in np.linspace(0.40, 0.60, 41):
        m = compute_quadrant_memberships(v, 0.9)
        if prev is not None:
            assert abs(m["Q1"] - prev) < 0.06
        prev = m["Q1"]


# ---------------- 声刺激干预策略方向 ----------------
@pytest.fixture(scope="module")
def stim_cfg():
    return load_stimulus_params()


def test_q2_counter_stimulation_slows_pulse_with_arousal(stim_cfg):
    """Q2（焦虑）：arousal 越高脉冲率越慢（降唤醒干预，文档 §4.2 的反向分支）。"""
    m = {"Q2": 1.0, "Q1": 0.0, "Q3": 0.0, "Q4": 0.0}
    assert compute_params(0.2, 0.9, m, stim_cfg).pr < compute_params(0.2, 0.3, m, stim_cfg).pr


def test_q3_activation_raises_f0_as_valence_drops(stim_cfg):
    """Q3（低落）：valence 越低基频越高（注入明亮感的激活干预）。"""
    m = {"Q3": 1.0, "Q1": 0.0, "Q2": 0.0, "Q4": 0.0}
    assert compute_params(0.1, 0.2, m, stim_cfg).f0 > compute_params(0.45, 0.2, m, stim_cfg).f0


def test_q1_and_q4_follow_iso_direction(stim_cfg):
    """Q1/Q4 为匹配式：arousal 越高脉冲越快，valence 越高基频越高。"""
    for q in ("Q1", "Q4"):
        m = {k: (1.0 if k == q else 0.0) for k in ("Q1", "Q2", "Q3", "Q4")}
        assert compute_params(0.7, 0.9, m, stim_cfg).pr > compute_params(0.7, 0.2, m, stim_cfg).pr
        assert compute_params(0.9, 0.5, m, stim_cfg).f0 > compute_params(0.6, 0.5, m, stim_cfg).f0


def test_pink_noise_only_in_down_regulation_quadrants(stim_cfg):
    """粉噪仅用于 Q2/Q4（遮蔽/填充），Q1/Q3 不混入。"""
    for q, expect_noise in (("Q1", False), ("Q2", True), ("Q3", False), ("Q4", True)):
        m = {k: (1.0 if k == q else 0.0) for k in ("Q1", "Q2", "Q3", "Q4")}
        p = compute_params(0.5, 0.5, m, stim_cfg)
        assert (p.noise_ratio > 0) is expect_noise


def test_params_continuous_across_quadrant_boundary(stim_cfg):
    """软混合真实生效：valence 跨过 0.5（Q2↔Q1）时脉冲率/基频/粉噪连续变化。

    此前实现只取主象限分支，隶属度未参与计算，v=0.49→0.51 时 pr 从 Q2 公式跳到 Q1 公式。
    """
    prev = None
    for v in np.linspace(0.40, 0.60, 81):
        m = compute_quadrant_memberships(v, 0.9)
        p = compute_params(v, 0.9, m, stim_cfg)
        if prev is not None:
            assert abs(p.pr - prev.pr) < 0.25, f"pr jump at v={v:.3f}: {prev.pr:.3f}->{p.pr:.3f}"
            assert abs(p.f0 - prev.f0) < 30, f"f0 jump at v={v:.3f}"
            assert abs(p.noise_ratio - prev.noise_ratio) < 0.03
        prev = p


def test_pure_membership_matches_branch_formula(stim_cfg):
    """隶属度为 one-hot 时退化为该象限的连续映射公式。"""
    m = {"Q2": 1.0, "Q1": 0.0, "Q3": 0.0, "Q4": 0.0}
    p = compute_params(0.2, 0.6, m, stim_cfg)
    assert p.pr == pytest.approx(0.25 + 0.75 * 0.4)
    assert p.f0 == pytest.approx(200 + 200 * 0.2)
