"""物理声学特征提取（librosa/scipy）。

【原理】除韵律外，语音的底层物理声学特征也携带情感线索（Ilie & Thompson 2006
比较了音乐与语音的声学线索）。本模块提取：

  - 响度 RMS：高强度情绪（愤怒、兴奋）响度大，低落情绪响度小。
  - 频谱质心 spectral_centroid：频谱能量的「重心」频率。高频质心=声音明亮
    （高唤醒），低频质心=声音暗沉（低唤醒）。
  - 高频能量比 hf_energy_ratio：>2kHz 能量占比。过高（刺耳）或过低（沉闷）
    都偏向负面，故负面分用 |norm-0.5|*2（两端都增负面）。
  - 信噪比 SNR：同时用于音频质量评估（低 SNR 时融合模块会降低声学权重）和
    负面分（噪声大→负向体验）。本估计用「信号帧 vs 噪声帧」功率比，并对稳态
    信号（CV<0.05，如纯音）视为高 SNR。
  - 频谱粗糙度 roughness：基于 Sethares (1993) 与 Plomp-Levelt 模型——相邻频率
    分量在 20-150Hz 拍频内会产生「粗糙」的不协和感（人耳对 ~70Hz 拍频最敏感）。
    粗糙度高→声音紧张刺耳→负面。本实现取频谱显著峰对，按拍频的高斯权重加权求和。

按项目计划文档 3.2 节（4）聚合成 (s_physical, a_physical)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from scipy import signal

from src.fusion.normalizer import clip01


@dataclass
class PhysicalFeatures:
    rms: float                 # 响度 RMS
    spectral_centroid: float   # 频谱质心 Hz
    hf_energy_ratio: float     # 高频(>2kHz)能量比
    snr_db: float              # 信噪比 dB
    roughness: float           # 频谱粗糙度


def _estimate_snr(y: np.ndarray, sr: int) -> float:
    """估计信噪比（dB）。

    采用「信号-噪声帧功率比 + 稳态信号识别」混合策略：
      1. 分帧计算功率；
      2. 噪声功率 = 最低 30% 帧均值；信号功率 = 最高 30% 帧均值；
      3. 若帧间功率方差极小（确定性/稳态信号，如纯正弦），则视为高 SNR
         （此类信号本身不含噪声成分）。
    """
    if len(y) < int(sr * 0.1):
        return 0.0
    frame_len = int(sr * 0.02)
    n_frames = len(y) // frame_len
    if n_frames < 4:
        rms = np.sqrt(np.mean(y ** 2))
        return float(20 * np.log10(rms + 1e-10))

    powers = np.array([
        np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    mean_power = float(np.mean(powers))
    if mean_power <= 1e-12:
        return 0.0

    # 帧间功率变异系数（CV）：稳态信号 CV 极小
    cv = float(np.std(powers) / (mean_power + 1e-12))
    if cv < 0.05:
        # 稳态/确定性信号 -> 高 SNR
        return 45.0

    powers_sorted = np.sort(powers)
    n_third = max(1, n_frames // 3)
    noise_power = float(np.mean(powers_sorted[:n_third]))
    signal_power = float(np.mean(powers_sorted[-n_third:]))
    if noise_power <= 1e-12:
        return 60.0
    snr = 10 * np.log10(signal_power / noise_power)
    return float(max(0.0, min(60.0, snr)))


def _estimate_roughness(y: np.ndarray, sr: int) -> float:
    """基于 Sethares (1993) 的频谱粗糙度估计。

    计算频谱峰值之间的拍频干扰加权和。简化实现：取频谱前若干个显著峰值，
    对相邻峰对按其频率差的绝对值（拍频）与振幅乘积加权求和。
    """
    if len(y) < int(sr * 0.05):
        return 0.0
    n_fft = min(8192, len(y))
    spec = np.abs(np.fft.rfft(y, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    if len(spec) < 4:
        return 0.0
    # 找局部峰值
    peaks, _ = signal.find_peaks(spec, height=np.max(spec) * 0.05)
    if len(peaks) < 2:
        return 0.0
    # 取振幅最大的若干峰
    order = np.argsort(spec[peaks])[::-1][:min(8, len(peaks))]
    peaks = np.sort(peaks[order])
    peak_freqs = freqs[peaks]
    peak_amps = spec[peaks]
    peak_amps = peak_amps / (np.max(peak_amps) + 1e-10)

    rough = 0.0
    for i in range(len(peaks) - 1):
        f1, f2 = peak_freqs[i], peak_freqs[i + 1]
        a1, a2 = peak_amps[i], peak_amps[i + 1]
        beat = abs(f2 - f1)
        # 拍频在 20~150Hz 内产生最强粗糙感（Plomp-Levelt 模型简化）
        if 0 < beat < 300:
            # 高斯式权重：peak ~ 70Hz
            w = np.exp(-((beat - 70) / 50) ** 2)
            rough += a1 * a2 * w
    return float(rough)


def extract(y: np.ndarray, sr: int) -> PhysicalFeatures:
    """提取物理声学特征。

    Args:
        y: 单声道音频波形。
        sr: 采样率。

    Returns:
        :class:`PhysicalFeatures`。
    """
    if len(y) == 0:
        return PhysicalFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    y = np.asarray(y, dtype=np.float64)

    rms = float(np.mean(librosa.feature.rms(y=y)))
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 高频能量比（>2kHz）
    fft_spec = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    total_e = np.sum(fft_spec ** 2)
    hf_e = np.sum(fft_spec[fft_freqs > 2000] ** 2)
    hf_ratio = float(hf_e / total_e) if total_e > 0 else 0.0

    snr_db = _estimate_snr(y, sr)
    roughness = _estimate_roughness(y, sr)

    return PhysicalFeatures(
        rms=rms, spectral_centroid=sc, hf_energy_ratio=hf_ratio,
        snr_db=snr_db, roughness=roughness,
    )


def score(feat: PhysicalFeatures) -> tuple[float, float, dict[str, Any]]:
    """聚合为 (s_physical, a_physical, 详情)。公式见文档 3.2 节（4）。"""
    norm_loudness = clip01((feat.rms - 0.01) / 0.15)
    norm_centroid = clip01((feat.spectral_centroid - 500) / 2500)
    norm_hf = clip01(feat.hf_energy_ratio / 0.4)
    norm_snr = clip01(feat.snr_db / 30)
    norm_roughness = clip01(feat.roughness / 0.3)
    norm_hf_extreme = abs(norm_hf - 0.5) * 2  # 过多/过少高频都增负面

    s_physical = 0.3 * norm_roughness + 0.2 * (1 - norm_snr) + 0.2 * norm_hf_extreme + 0.3 * 0.5
    a_physical = 0.35 * norm_loudness + 0.25 * norm_centroid + 0.20 * norm_hf + 0.20 * norm_roughness

    detail = {
        "rms": feat.rms, "spectral_centroid": feat.spectral_centroid,
        "hf_energy_ratio": feat.hf_energy_ratio, "snr_db": feat.snr_db,
        "roughness": feat.roughness,
        "s_physical": clip01(s_physical), "a_physical": clip01(a_physical),
    }
    return clip01(s_physical), clip01(a_physical), detail
