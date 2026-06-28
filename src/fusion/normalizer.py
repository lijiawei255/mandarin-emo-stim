"""各模态特征归一化工具。

将原始声学/文本统计量通过 z-score 归一化到 [0, 1]：
    z = (value - mu) / sigma        # 截断到 [-2, 2]
    norm = (z + 2) / 4              # 映射到 [0, 1]

参考均值/标准差来自中文普通话语音研究文献（见项目计划文档 3.2 节）。
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
