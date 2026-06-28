"""韵律学特征提取（praat-parselmouth）。

提取 F0 相关指标、语速估计、停顿占比、HNR、Jitter、Shimmer，并按
项目计划文档 3.2 节的子权重公式聚合成 (负面分 s_prosody, 唤醒分 a_prosody)。

所有原始指标通过 :mod:`src.fusion.normalizer` 的 z-score 归一化到 [0,1]，
再按子权重加权得到最终模态分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call

from src.fusion.normalizer import PROSODY_STATS, clip01, zscore_normalize


@dataclass
class ProsodyFeatures:
    """原始韵律学指标（未归一化）。"""
    mean_f0: float
    std_f0: float
    f0_range: float
    speech_rate: float       # 音节/秒估计
    pause_ratio: float       # 0~1
    hnr: float               # dB
    jitter_local: float
    shimmer_local: float
    duration: float          # 有效语音时长（秒）


def extract(y: np.ndarray, sr: int) -> ProsodyFeatures:
    """提取韵律学原始指标。

    Args:
        y: 单声道音频波形。
        sr: 采样率。

    Returns:
        :class:`ProsodyFeatures`。空/静音音频返回中性默认值。
    """
    duration = len(y) / sr if sr > 0 else 0.0
    if len(y) < int(sr * 0.05) or np.max(np.abs(y)) < 1e-5:
        # 太短或近静音
        return ProsodyFeatures(
            mean_f0=180.0, std_f0=25.0, f0_range=0.0, speech_rate=0.0,
            pause_ratio=1.0, hnr=0.0, jitter_local=0.0, shimmer_local=0.0,
            duration=duration,
        )

    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # ---- F0 ----
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0_vals = pitch.selected_array["frequency"]
    f0_vals = f0_vals[f0_vals > 0]  # 去除无声帧的 0
    if len(f0_vals) > 0:
        mean_f0 = float(np.mean(f0_vals))
        std_f0 = float(np.std(f0_vals))
        f0_range = float(np.max(f0_vals) - np.min(f0_vals))
    else:
        mean_f0, std_f0, f0_range = 180.0, 25.0, 0.0

    # ---- HNR ----
    try:
        harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75,
                                             silence_threshold=0.1)
        hnr_vals = harmonicity.values[harmonicity.values > 0]
        hnr = float(np.mean(hnr_vals)) if len(hnr_vals) > 0 else 0.0
    except Exception:
        hnr = 0.0

    # ---- Jitter / Shimmer（基于脉冲点）----
    try:
        pulse = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = float(call(pulse, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer_local = float(call([snd, pulse], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        jitter_local, shimmer_local = 0.0, 0.0

    # ---- 语速估计（基于能量包络过零率近似）----
    speech_rate = _estimate_speech_rate(y, sr)

    # ---- 停顿占比（静音帧/总帧）----
    pause_ratio = _estimate_pause_ratio(y, sr)

    return ProsodyFeatures(
        mean_f0=mean_f0, std_f0=std_f0, f0_range=f0_range,
        speech_rate=speech_rate, pause_ratio=pause_ratio,
        hnr=hnr, jitter_local=jitter_local, shimmer_local=shimmer_local,
        duration=duration,
    )


def _estimate_speech_rate(y: np.ndarray, sr: int, frame_len: int = 0.02) -> float:
    """基于能量包络过零率近似估计音节率（音节/秒）。"""
    n_per_frame = int(sr * frame_len)
    if n_per_frame < 2 or len(y) < n_per_frame:
        return 0.0
    n_frames = len(y) // n_per_frame
    energy = np.array([
        np.mean(y[i * n_per_frame:(i + 1) * n_per_frame] ** 2)
        for i in range(n_frames)
    ])
    if len(energy) < 2:
        return 0.0
    # 能量包络归一化
    e_mean = np.mean(energy)
    if e_mean <= 0:
        return 0.0
    env = energy / e_mean
    # 计数能量越过均值的次数（近似音节数）
    crossings = np.sum(np.diff(np.sign(env - 1.0)) != 0)
    duration_sec = n_frames * frame_len
    return float(crossings / 2.0 / duration_sec) if duration_sec > 0 else 0.0


def _estimate_pause_ratio(y: np.ndarray, sr: int, frame_len: float = 0.02,
                          silence_thr: float = 0.01) -> float:
    """估计静音帧占比。"""
    n_per_frame = int(sr * frame_len)
    if n_per_frame < 2 or len(y) < n_per_frame:
        return 1.0
    n_frames = len(y) // n_per_frame
    silent = 0
    for i in range(n_frames):
        frame = y[i * n_per_frame:(i + 1) * n_per_frame]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < silence_thr:
            silent += 1
    return float(silent / n_frames) if n_frames > 0 else 1.0


def score(feat: ProsodyFeatures) -> tuple[float, float, dict[str, Any]]:
    """将韵律原始指标聚合为 (s_prosody, a_prosody, 详情)。

    子权重见项目计划文档 3.2 节（2）。
    """
    s = PROSODY_STATS
    n_hnr_inv = 1 - zscore_normalize(feat.hnr, *s["hnr"])     # HNR 越低越负面
    n_jitter = zscore_normalize(feat.jitter_local, *s["jitter_local"])
    n_shimmer = zscore_normalize(feat.shimmer_local, *s["shimmer_local"])
    # f0_drop：用 F0 结尾 - F0 开头近似不可得时，用 std 近似负斜率贡献
    n_f0_drop = zscore_normalize(max(0.0, feat.std_f0), *s["std_f0"])
    n_pause_ratio = zscore_normalize(feat.pause_ratio, *s["pause_ratio"])
    n_speech_rate_low = 1 - zscore_normalize(feat.speech_rate, *s["speech_rate"])

    s_prosody = (
        0.25 * n_hnr_inv + 0.20 * n_jitter + 0.20 * n_shimmer
        + 0.15 * n_f0_drop + 0.10 * n_pause_ratio + 0.10 * n_speech_rate_low
    )

    n_speech_rate = zscore_normalize(feat.speech_rate, *s["speech_rate"])
    n_f0_range = zscore_normalize(feat.f0_range, *s["f0_range"])
    n_std_f0 = zscore_normalize(feat.std_f0, *s["std_f0"])
    n_mean_f0 = zscore_normalize(feat.mean_f0, *s["mean_f0"])
    n_pause_ratio_inv = 1 - zscore_normalize(feat.pause_ratio, *s["pause_ratio"])

    a_prosody = (
        0.30 * n_speech_rate + 0.25 * n_f0_range + 0.20 * n_std_f0
        + 0.15 * n_mean_f0 + 0.10 * n_pause_ratio_inv
    )

    detail = {
        "mean_f0": feat.mean_f0, "std_f0": feat.std_f0, "f0_range": feat.f0_range,
        "speech_rate": feat.speech_rate, "pause_ratio": feat.pause_ratio,
        "hnr": feat.hnr, "jitter_local": feat.jitter_local,
        "shimmer_local": feat.shimmer_local,
        "s_prosody": clip01(s_prosody), "a_prosody": clip01(a_prosody),
    }
    return clip01(s_prosody), clip01(a_prosody), detail
