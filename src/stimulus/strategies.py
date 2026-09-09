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
  - 粉噪 noise：仅 Q2/Q4 混入，作用是能量遮蔽与频谱填充（使纯音不显单薄）。
    注意：本项目**不主张**粉噪有生理平复效应——早期引用的 Söderlund 2007
    研究的是白噪对 ADHD 儿童认知表现的影响，与放松无关，已更正
    （见 docs/research_notes.md §4.4）。

【映射方式】对四个象限各自计算 (valence, arousal) 的连续映射（脉冲率 / 基频 /
粉噪比），再按象限隶属度加权求和（软混合）；响度、频谱质心、起音、调制深度
只依赖 (v, a) 本身；谐和结构为离散量，取主象限锚点。这样 v 或 a 跨过 0.5 时
连续参数不会跳变（tests/test_scientific_behavior.py 有连续性断言）。
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

    dom = _dominant_for_mapping(memberships)
    v = max(0.0, min(1.0, valence))
    a = max(0.0, min(1.0, arousal))

    # ---- 软混合：按四象限隶属度对各象限的连续映射结果加权 ----
    # 此前实现只用主象限分支（q1..q4 解包后未使用），与文档「软混合避免边界突变」
    # 不符：v 跨过 0.5 时 pr/f0 会跳变。现对每个象限分别计算再按隶属度加权。
    pr = sum(memberships[q] * _pulse_rate(q, a) for q in _QUADRANTS)
    f0 = sum(memberships[q] * _base_freq(q, v) for q in _QUADRANTS)
    noise_ratio = sum(memberships[q] * _noise_ratio(q, a) for q in _QUADRANTS)

    # ---- 响度（线性映射）----
    loud_db = -30 + 20 * a

    # ---- 频谱质心（唤醒主导，效价微调）----
    sc = 400 + 2100 * a + 400 * v

    # ---- 起音时间 ----
    attack_ms = 500 - 480 * a

    # ---- 调制深度 ----
    mod_depth = 0.20 + 0.40 * a

    # ---- 谐和结构（离散量，取主象限锚点）----
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


_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


def _pulse_rate(q: str, a: float) -> float:
    """各象限的脉冲率连续映射（Hz）。Q2 为降唤醒反向分支。"""
    if q == "Q2":      # 焦虑：arousal 越高 -> 脉冲越慢（引导放缓）
        return 0.25 + 0.75 * (1 - a)
    if q == "Q3":      # 低落：arousal 越高 -> 脉冲越快（激活）
        return 1.5 + 2.5 * a
    if q == "Q1":      # 兴奋：越高越快（匹配）
        return 2.0 + 4.0 * a
    return 0.5 + 1.0 * a   # Q4 放松（匹配）


def _base_freq(q: str, v: float) -> float:
    """各象限的基频连续映射（Hz）。Q3 为提亮反向分支。"""
    if q == "Q2":      # valence 越低 -> f0 越低（深沉）
        return 200 + 200 * v
    if q == "Q3":      # valence 越低 -> f0 越高（注入明亮感）；v=0.5 时 = Q3 锚点 450 Hz
        return 300 + 300 * (1 - v)
    if q == "Q1":
        return 400 + 400 * v
    return 250 + 250 * v   # Q4


def _noise_ratio(q: str, a: float) -> float:
    """粉噪比：仅 Q2/Q4 混入（遮蔽/填充）。"""
    if q == "Q2":
        return 0.05 + 0.20 * a
    if q == "Q4":
        return 0.05 + 0.10 * (1 - a)
    return 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
