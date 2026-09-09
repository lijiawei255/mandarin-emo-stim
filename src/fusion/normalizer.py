"""各模态特征归一化工具（z-score → [0,1]）。

【为何要归一化】各模态特征的量纲差异巨大（F0 是 Hz、HNR 是 dB、Jitter 是比值…），
无法直接加权融合。需统一映射到 [0,1] 连续值，且要保留「相对情绪倾向」语义：
0=极端低/负面，1=极端高/正面（或反之，取决于特征）。

【z-score 归一化】以「该特征在正常语音中的分布」为参照：
    z = (value - mu) / sigma        # 标准化：偏离均值多少个标准差
    z 截断到 [-2, 2]                 # 抑制极端离群点，避免单特征主导
    norm = (z + 2) / 4              # 线性映射到 [0, 1]
效果：value=mu(典型值) → 0.5（中性）；比典型值高 2σ → 1.0；低 2σ → 0.0。

【参考值来源】优先读取 ``config/prosody_norms.json``——由
``scripts/evaluate.py calibrate`` 在 AISHELL-3（Apache-2.0，218 位普通话说话人的
情绪中性朗读语料）上分层抽样**实测**得到，含样本量、抽样方式与日期。该文件
缺失时回退到 LEGACY_PROSODY_STATS：早期版本**凭空设定**的启发式常数，仅作兜底，
不应被当作「中文语料统计」引用。偏离基准的程度即反映情绪强度。

注意：z-score 假设特征近似正态分布。对明显偏态的特征（如 Jitter）这是近似，
但工程上足够鲁棒。
"""

from __future__ import annotations

import json
from pathlib import Path

# 早期版本的启发式常数（未经实测），仅在 prosody_norms.json 缺失时兜底
LEGACY_PROSODY_STATS: dict[str, tuple[float, float]] = {
    # 特征名: (均值 mu, 标准差 sigma)
    "mean_f0": (180.0, 50.0),      # Hz，男女混合
    "std_f0": (25.0, 15.0),        # Hz
    "f0_range": (80.0, 40.0),      # Hz（max_f0 - min_f0）
    "speech_rate": (4.5, 1.5),     # 音节/秒
    "pause_ratio": (0.25, 0.12),   # 0~1
    "hnr": (15.0, 5.0),            # dB
    "jitter_local": (0.02, 0.015),
    "shimmer_local": (0.08, 0.04),
    "f0_slope": (0.0, 40.0),       # Hz/s，语句级 F0 线性斜率
}


def load_prosody_stats(path: Path | None = None,
                       group: str = "mixed") -> dict[str, tuple[float, float]]:
    """加载韵律 z-score 参考 μ/σ。

    Args:
        path: ``prosody_norms.json`` 路径；默认 ``config/prosody_norms.json``。
        group: ``"mixed"`` / ``"female"`` / ``"male"``。

    Returns:
        ``{特征名: (mu, sigma)}``。文件缺失、损坏或缺少某特征时，该特征回退到
        :data:`LEGACY_PROSODY_STATS`；sigma 非正时同样回退。
    """
    if path is None:
        from src import portable
        path = portable.CONFIG_DIR / "prosody_norms.json"
    stats = dict(LEGACY_PROSODY_STATS)
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        block = data.get(group) or data.get("mixed") or {}
        for k in LEGACY_PROSODY_STATS:
            v = block.get(k)
            if v and float(v.get("sigma", 0)) > 0:
                stats[k] = (float(v["mu"]), float(v["sigma"]))
    except (OSError, ValueError, TypeError):
        pass
    return stats


# 模块级默认（实测优先，缺失回退）
PROSODY_STATS: dict[str, tuple[float, float]] = load_prosody_stats()


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
