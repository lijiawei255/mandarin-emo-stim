"""波形合成器。

依据 :class:`~src.stimulus.strategies.StimulusParams` 合成双声道声刺激波形，
流程：基础谐和音 → 频谱塑形（带通）→ 振幅调制（脉冲率）→ ADSR 包络 →
混入粉噪 → 全局响度归一化 → 安全限幅（峰值 -10dBFS）→ 淡入淡出 →
立体声（Haas 效应）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import slab
from scipy import signal

from src.stimulus.strategies import StimulusParams


def _pink_noise(duration: float, sr: int) -> np.ndarray:
    """生成粉噪（单声道），振幅归一化到 [-1, 1]。

    用 slab 生成；slab 不可用时回退到 Voss-McCartney 算法。
    """
    try:
        snd = slab.Sound.pinknoise(duration=duration, samplerate=sr)
        data = np.asarray(snd.data)
        if data.ndim > 1:
            data = data[:, 0]
        return data
    except Exception:
        # Voss-McCartney 近似粉噪
        n = int(sr * duration)
        n_rows = 16
        rows = np.random.randint(0, 2, size=(n_rows, n))
        rows[:, 0] = 1
        pink = np.cumsum(rows, axis=1).astype(np.float64)
        pink -= pink.mean()
        peak = np.max(np.abs(pink))
        if peak > 0:
            pink /= peak
        return pink


def _adsr_envelope(n: int, attack_ms: float, sr: int,
                   decay_ms: float, sustain_level: float,
                   release_ms: float) -> np.ndarray:
    """生成 ADSR 包络。

    Attack 线性上升到 1.0；Decay 线性下降到 sustain_level；
    Sustain 保持；Release 线性下降到 0。
    """
    env = np.ones(n, dtype=np.float64)
    n_att = max(1, int(attack_ms / 1000.0 * sr))
    n_dec = max(1, int(decay_ms / 1000.0 * sr))
    n_rel = max(1, int(release_ms / 1000.0 * sr))
    n_att = min(n_att, n)
    n_dec = min(n_dec, n - n_att)
    n_rel = min(n_rel, n - n_att - n_dec)

    env[:n_att] = np.linspace(0, 1, n_att)
    env[n_att:n_att + n_dec] = np.linspace(1, sustain_level, n_dec)
    env[n - n_rel:] = np.linspace(sustain_level, 0, n_rel)
    return env


def synthesize(params: StimulusParams, duration: float, sr: int,
               config: dict[str, Any]) -> np.ndarray:
    """合成双声道声刺激波形。

    Args:
        params: 声学参数。
        duration: 时长（秒）。
        sr: 采样率。
        config: stimulus_params.json（提供 adsr / stimulus fade 等）。

    Returns:
        ``np.ndarray`` shape=(n_samples, 2) dtype=float32，峰值 <= -10 dBFS。
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # 1. 基础谐和音
    tone = np.zeros(n, dtype=np.float64)
    for freq, amp in zip(params.freqs, params.amps):
        tone += amp * np.sin(2 * np.pi * freq * t)

    # 2. 频谱塑形：带通滤波器，中心约在频谱质心附近
    lo = max(20.0, params.sc * 0.4)
    hi = min(sr / 2.0 - 1.0, params.sc * 1.8)
    if hi > lo:
        sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        tone = signal.sosfilt(sos, tone)

    # 3. 振幅调制（脉冲率 pr）
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * params.pr * t - np.pi / 2)
    modulator = (1 - params.mod_depth) + params.mod_depth * modulator
    tone = tone * modulator

    # 4. ADSR 包络
    adsr = config.get("adsr", {})
    env = _adsr_envelope(
        n, params.attack_ms, sr,
        decay_ms=adsr.get("decay_ms", 50),
        sustain_level=adsr.get("sustain_level", 0.8),
        release_ms=adsr.get("release_ms", 200),
    )
    tone = tone * env

    # 5. 混入粉噪
    if params.noise_ratio > 0:
        pink = _pink_noise(duration, sr)[:n]
        peak = np.max(np.abs(pink))
        if peak > 0:
            pink = pink / peak * params.noise_ratio
        tone = tone + pink

    # 6. 全局响度归一化到目标 RMS 电平
    loud_amp = 10 ** (params.loud_db / 20.0)
    peak = np.max(np.abs(tone))
    if peak > 0:
        tone = tone / peak * loud_amp

    # 7. 安全限幅（确保不超过 -10dBFS 峰值）
    max_peak_amp = 10 ** (-10 / 20.0)
    peak = np.max(np.abs(tone))
    if peak > max_peak_amp:
        tone = tone * (max_peak_amp / peak)

    # 8. 淡入淡出（防咔嗒声）
    fade_ms = config.get("fade_ms", 50)
    fade_len = max(1, int(fade_ms / 1000.0 * sr))
    fade_len = min(fade_len, n // 2)
    tone[:fade_len] *= np.linspace(0, 1, fade_len)
    tone[-fade_len:] *= np.linspace(1, 0, fade_len)

    # 9. 双声道（Haas 效应：右声道延迟 12ms 产生自然宽度）
    haas_ms = config.get("haas_delay_ms", 12)
    haas_delay = max(0, int(haas_ms / 1000.0 * sr))
    right = np.roll(tone, haas_delay)
    right[:haas_delay] = 0.0
    stereo = np.column_stack([tone, right])

    # 防止叠加削波
    stereo = stereo * 0.7

    return stereo.astype(np.float32)
