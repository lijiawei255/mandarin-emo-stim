"""声刺激参数映射策略（音乐心理学实证映射）。

【核心目标】根据检测到的情绪状态，生成「差异化」的声刺激——不是统一放松音，
而是针对不同情绪象限给出有理论依据的声学干预。

【理论依据】音乐心理学研究（Juslin & Laukka 2004 表演情感交流；Bresin &
Friberg 2011 情绪渲染；Ilie & Thompson 2006 音乐/语音声学线索比较）系统量化了
声学参数与情绪感知的映射关系。本项目据此把 Russell 四象限映射到声学参数：

  - 脉冲率 pr（节奏）：高唤醒→快脉冲（激活感），低唤醒→慢脉冲（平复感）。
    但 Q2 焦虑例外：高唤醒反而用慢脉冲引导呼吸放缓（降唤醒干预）。
  - 基频 f0：正面情绪→高基频（明亮），负面→低基频（深沉）；Q3 抑郁例外：
    valence 越低反而提 f0（注入能量/明亮感，激活干预）。
  - 响度 loud：随 arousal 线性增加（[-30,-10]dBFS）。
  - 频谱质心 sc：唤醒主导（高频→明亮/激动），效价微调。
  - 起音 attack：高唤醒→陡起音（<50ms，冲击感），低唤醒→缓起音（~500ms，柔和）。
  - 谐和结构 harmony：正面用大三和弦（协和明亮），焦虑用整数泛音（紧张感），
    放松用五度+八度（空灵）。
  - 粉噪 noise：仅 Q2/Q4 降唤醒场景混入（粉噪有平复作用，参考 Soderlund 2007
    关于粉噪对注意力的影响）。

【映射方式】两步：
    1. 按四象限隶属度对锚点参数做加权混合（软混合，避免象限边界突变）；
    2. 在主象限内，按 valence/arousal 连续微调（见 compute_params 内分支）。
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
