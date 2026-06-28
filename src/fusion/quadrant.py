"""情绪象限软判定（隶属度计算）。

【原理】基于 Russell (1980) 情绪环模型（Circumplex Model of Affect）：任意
情绪状态都可投影到 Valence（正负效价）× Arousal（唤醒度）二维平面，四个象限
对应不同情绪簇：
    Q1: 高V高A — 积极/兴奋（如喜悦、狂喜）
    Q2: 低V高A — 焦虑/紧张（如愤怒、恐惧、焦虑）  ← 本工具干预重点
    Q3: 低V低A — 低落/抑郁（如悲伤、沮丧）
    Q4: 高V低A — 放松/满足（如平静、满足）

【为何用「软」隶属度而非硬切分】若用阈值硬切（v>0.5 且 a>0.5 → Q1），
边界附近（如 v=0.49/0.51）会产生象限跳变，导致生成的声刺激参数突变（刺耳）。
故采用线性过渡带（band，默认 0.05）做软分配：每个轴在 [mid-band, mid+band]
区间内线性插值隶属度，四象限隶属度归一化后和为 1。最终刺激参数用四象限锚点
按隶属度加权混合（见 stimulus/strategies.py），实现平滑过渡无突变。

隶属度还作为 UI 展示与下游声刺激参数软混合的输入。
"""

from __future__ import annotations

QUADRANT_NAMES = {
    "Q1": "积极/兴奋",
    "Q2": "焦虑/紧张",
    "Q3": "低落/抑郁",
    "Q4": "放松/满足",
}


def soft_membership(value: float, mid: float, band: float,
                    high_side: bool) -> float:
    """单轴软隶属度（线性过渡带）。

    ``high_side=True`` 时表示 value 越大隶属度越高；
    ``False`` 时表示 value 越小隶属度越高。
    """
    if high_side:
        return max(0.0, (value - (mid - band)) / (2.0 * band))
    return max(0.0, ((mid + band) - value) / (2.0 * band))


def compute_quadrant_memberships(
    valence: float,
    arousal: float,
    mid_v: float = 0.5,
    mid_a: float = 0.5,
    band: float = 0.05,
) -> dict[str, float]:
    """计算四象限归一化隶属度。

    返回 ``{"Q1": ..., "Q2": ..., "Q3": ..., "Q4": ...}``，和为 1。
    """
    # 高 Valence 轴隶属度
    v_high = soft_membership(valence, mid_v, band, high_side=True)
    v_low = soft_membership(valence, mid_v, band, high_side=False)
    # 高 Arousal 轴隶属度
    a_high = soft_membership(arousal, mid_a, band, high_side=True)
    a_low = soft_membership(arousal, mid_a, band, high_side=False)

    q1 = v_high * a_high  # 高V高A
    q2 = v_low * a_high   # 低V高A
    q3 = v_low * a_low    # 低V低A
    q4 = v_high * a_low   # 高V低A

    total = q1 + q2 + q3 + q4
    if total > 0:
        q1 /= total
        q2 /= total
        q3 /= total
        q4 /= total
    else:
        q1 = q2 = q3 = q4 = 0.25

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


def dominant_quadrant(memberships: dict[str, float]) -> str:
    """返回隶属度最高的象限标签。"""
    return max(memberships, key=memberships.get)
