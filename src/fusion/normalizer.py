"""各模态特征归一化工具（z-score → [0,1]）。

【为何要归一化】各模态特征的量纲差异巨大（F0 是 Hz、HNR 是 dB、Jitter 是比值…），
无法直接加权融合。需统一映射到 [0,1] 连续值，且要保留「相对情绪倾向」语义：
0=极端低/负面，1=极端高/正面（或反之，取决于特征）。

【z-score 归一化】以「该特征在正常语音中的分布」为参照：
    z = (value - mu) / sigma        # 标准化：偏离均值多少个标准差
    z 截断到 [-2, 2]                 # 抑制极端离群点，避免单特征主导
    norm = (z + 2) / 4              # 线性映射到 [0, 1]
效果：value=mu(典型值) → 0.5（中性）；比典型值高 2σ → 1.0；低 2σ → 0.0。

【参考值来源】PROSODY_STATS 的 mu/sigma 来自中文普通话语音研究文献的语料统计
（男女混合），是「正常说话」的基准。偏离基准的程度即反映情绪强度。例如
mean_f0=180Hz 是中性，280Hz(+2σ)可能对应高唤醒情绪。

注意：z-score 假设特征近似正态分布。对明显偏态的特征（如 Jitter）这是近似，
但工程上足够鲁棒。
"""

from __future__ import annotations

# 韵律学特征的 z-score 参考统计量（基于中文普通话语料）
PROSODY_STATS: dict[str, tuple[float, float]] = {
    # 特征名: (均值 mu, 标准差 sigma)
    "mean_f0": (180.0, 50.0),      # Hz，男女混合
    "std_f0": (25.0, 15.0),        # Hz
    "f0_range": (80.0, 40.0),      # Hz（max_f0 - min_f0）
    "speech_rate": (4.5, 1.5),     # 音节/秒
    "pause_ratio": (0.25, 0.12),   # 0~1
    "hnr": (15.0, 5.0),            # dB
    "jitter_local": (0.02, 0.015),
    "shimmer_local": (0.08, 0.04),
}


def zscore_normalize(value: float, mu: float, sigma: float,
                     clip: float = 2.0) -> float:
    """z-score 归一化到 [0, 1]。

    z 先截断到 [-clip, clip]，再 ``norm = (z + clip) / (2 * clip)``。
    sigma <= 0 时返回 0.5（中性）。
    """
    if sigma <= 0:
        return 0.5
    z = (value - mu) / sigma
    z = max(-clip, min(clip, z))
    return (z + clip) / (2.0 * clip)


def clip01(value: float) -> float:
    """截断到 [0, 1]。"""
    return max(0.0, min(1.0, value))
