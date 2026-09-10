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
  - 能量动态范围 rms_dynamic_range_db（v0.3）：帧 RMS 的第 95 百分位与中位数之差
    （dB）。高唤醒语音的能量起伏更大（Banse & Scherer 1996 报告愤怒/恐惧的强度
    变异升高），是对「平均响度」的补充线索。

【聚合】（v0.3）负面分 = 粗糙度（唯一与情绪有文献关联的物理线索；v0.2 消融显示
含 SNR/高频极端度的负面分与效价参照序**负相关** ρV=-0.13，即反向噪声，已移除）；
唤醒分 = 0.30·响度 + 0.20·质心 + 0.15·高频比 + 0.15·粗糙度 + 0.20·动态范围。
SNR 仍提取，仅用于音频质量评估与动态权重。
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
    rms_dynamic_range_db: float = 0.0   # 帧 RMS 动态范围（P95 − 中位数，dB）


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
        return PhysicalFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    y = np.asarray(y, dtype=np.float64)

    rms_frames = librosa.feature.rms(y=y)[0]
    rms = float(np.mean(rms_frames))
    dyn_range = _dynamic_range_db(rms_frames)
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
        snr_db=snr_db, roughness=roughness, rms_dynamic_range_db=dyn_range,
    )


def _dynamic_range_db(rms_frames: np.ndarray, floor: float = 1e-5) -> float:
    """帧 RMS 动态范围：P95 − 中位数（dB）。只统计高于地板的帧，避免静音拉大范围。"""
    r = np.asarray(rms_frames, dtype=np.float64)
    r = r[r > floor]
    if len(r) < 4:
        return 0.0
    hi = 20 * np.log10(np.percentile(r, 95))
    mid = 20 * np.log10(np.median(r))
    return float(max(0.0, hi - mid))


def score(feat: PhysicalFeatures) -> tuple[float, float, dict[str, Any]]:
    """聚合为 (s_physical, a_physical, 详情)。公式见文档 3.2 节（4）。"""
    norm_loudness = clip01((feat.rms - 0.01) / 0.15)
    norm_centroid = clip01((feat.spectral_centroid - 500) / 2500)
    norm_hf = clip01(feat.hf_energy_ratio / 0.4)
    norm_snr = clip01(feat.snr_db / 30)
    norm_roughness = clip01(feat.roughness / 0.3)
    norm_dyn = clip01((feat.rms_dynamic_range_db - 4.0) / 12.0)   # 4 dB → 0，16 dB → 1

    # v0.3：负面分只保留粗糙度（v0.2 的 SNR / 高频极端度项在消融中与效价负相关，属反向噪声）；
    # 唤醒分加入能量动态范围。绝对偏置由融合层的中性校准处理。
    s_physical = norm_roughness
    a_physical = (0.30 * norm_loudness + 0.20 * norm_centroid + 0.15 * norm_hf
                  + 0.15 * norm_roughness + 0.20 * norm_dyn)

    detail = {
        "rms": feat.rms, "spectral_centroid": feat.spectral_centroid,
        "hf_energy_ratio": feat.hf_energy_ratio, "snr_db": feat.snr_db,
        "roughness": feat.roughness, "rms_dynamic_range_db": feat.rms_dynamic_range_db,
        "s_physical": clip01(s_physical), "a_physical": clip01(a_physical),
    }
    return clip01(s_physical), clip01(a_physical), detail
