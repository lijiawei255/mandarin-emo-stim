"""情绪象限软判定（隶属度计算）。

基于 Russell 情绪环模型，把 (valence, arousal) 映射到四个象限的隶属度，
隶属度归一化后和为 1。在象限边界附近使用软切换（过渡带），避免参数突变。

象限定义：
    Q1: 高 Valence 高 Arousal（积极兴奋）
    Q2: 低 Valence 高 Arousal（焦虑紧张）
    Q3: 低 Valence 低 Arousal（低落抑郁）
    Q4: 高 Valence 低 Arousal（放松满足）
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
