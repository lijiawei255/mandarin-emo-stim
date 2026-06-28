"""声刺激参数映射策略。

依据情绪象限与 (valence, arousal) 连续值，计算出声刺激所需的全部声学参数。

设计依据 Russell 情绪环模型与音乐心理学实证映射（见项目计划文档 3.6 节）：
    - 四象限锚点参数软混合（避免硬切换突兀）；
    - 在锚点基础上，按主象限做 valence/arousal 连续微调。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.fusion.quadrant import dominant_quadrant


@dataclass
class StimulusParams:
    """单次声刺激的声学参数集合。"""
    f0: float                        # 基频 Hz
    pr: float                        # 脉冲率（振幅调制频率）Hz
    loud_db: float                   # 目标 RMS 电平 dBFS
    sc: float                        # 频谱质心 Hz
    harmony: str                     # 谐和结构名
    attack_ms: float                 # 起音时间 ms
    mod_depth: float                 # 振幅调制深度 0~1
    noise_ratio: float               # 粉噪振幅混合比 0~1
    freqs: list[float] = field(default_factory=list)   # 谐和频率
    amps: list[float] = field(default_factory=list)    # 各频率振幅


def _harmony_freqs_amps(harmony: str, f0: float,
                        harmony_defs: dict[str, Any]) -> tuple[list[float], list[float]]:
    """根据谐和结构定义生成 (频率列表, 振幅列表)。"""
    spec = harmony_defs.get(harmony, harmony_defs.get("natural_harmonics"))
    ratios = spec["freq_ratios"]
    amps = spec["amplitudes"]
    freqs = [f0 * r for r in ratios]
    return freqs, amps


def _dominant_for_mapping(memberships: dict[str, float]) -> str:
    """返回用于连续微调的主象限标签。"""
    return dominant_quadrant(memberships)


def compute_params(
    valence: float,
    arousal: float,
    memberships: dict[str, float],
    config: dict[str, Any],
) -> StimulusParams:
    """计算单次刺激的声学参数。

    Args:
        valence: 效价 [0,1]。
        arousal: 唤醒度 [0,1]。
        memberships: 四象限隶属度 {"Q1".."Q4"}。
        config: stimulus_params.json 全部内容。

    Returns:
        :class:`StimulusParams`。
    """
    anchors = config["quadrant_anchors"]
    harmony_defs = config["harmony_definitions"]
    ranges = config["mapping_ranges"]

    q1, q2, q3, q4 = memberships["Q1"], memberships["Q2"], memberships["Q3"], memberships["Q4"]
    dom = _dominant_for_mapping(memberships)
    v = max(0.0, min(1.0, valence))
    a = max(0.0, min(1.0, arousal))

    # ---- 脉冲率 pr（按主象限做 arousal 连续微调）----
    if dom == "Q2":      # 焦虑：arousal 越高 -> 脉冲越慢（引导呼吸放缓）
        pr = 0.25 + 0.75 * (1 - a)
    elif dom == "Q3":    # 抑郁：arousal 越高 -> 脉冲越快（激活）
        pr = 1.5 + 2.5 * a
    elif dom == "Q1":    # 兴奋：arousal 越高 -> 越快
        pr = 2.0 + 4.0 * a
    else:                # Q4 放松
        pr = 0.5 + 1.0 * a

    # ---- 基频 f0（按主象限做 valence 连续微调）----
    if dom == "Q2":      # valence 越低 -> f0 越低（深沉感）
        f0 = 200 + 200 * v
    elif dom == "Q3":    # valence 越低 -> f0 越高（注入能量/明亮感）
        f0 = 300 + 300 * v
    elif dom == "Q1":
        f0 = 400 + 400 * v
    else:                # Q4
        f0 = 250 + 250 * v

    # ---- 响度（线性映射）----
    loud_db = -30 + 20 * a

    # ---- 频谱质心（唤醒主导，效价微调）----
    sc = 400 + 2100 * a + 400 * v

    # ---- 起音时间 ----
    attack_ms = 500 - 480 * a

    # ---- 调制深度 ----
    mod_depth = 0.20 + 0.40 * a

    # ---- 粉噪比（仅 Q2/Q4 使用降唤醒场景）----
    if dom == "Q2":
        noise_ratio = 0.05 + 0.20 * a
    elif dom == "Q4":
        noise_ratio = 0.05 + 0.10 * (1 - a)
    else:
        noise_ratio = 0.0

    # ---- 谐和结构（按主象限锚点）----
    harmony = anchors[dom]["harmony"]

    # ---- 截断到合法映射范围 ----
    f0 = _clip(f0, ranges["f0"]["min"], ranges["f0"]["max"])
    pr = _clip(pr, ranges["pr"]["min"], ranges["pr"]["max"])
    loud_db = _clip(loud_db, ranges["loud_db"]["min"], ranges["loud_db"]["max"])
    sc = _clip(sc, ranges["sc"]["min"], ranges["sc"]["max"])
    attack_ms = _clip(attack_ms, ranges["attack_ms"]["min"], ranges["attack_ms"]["max"])
    mod_depth = _clip(mod_depth, ranges["mod_depth"]["min"], ranges["mod_depth"]["max"])
    noise_ratio = max(0.0, min(ranges["noise_ratio"]["max"], noise_ratio))

    freqs, amps = _harmony_freqs_amps(harmony, f0, harmony_defs)

    return StimulusParams(
        f0=f0, pr=pr, loud_db=loud_db, sc=sc, harmony=harmony,
        attack_ms=attack_ms, mod_depth=mod_depth, noise_ratio=noise_ratio,
        freqs=freqs, amps=amps,
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
