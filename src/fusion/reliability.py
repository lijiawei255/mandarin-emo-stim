"""不确定性 → 可信度等级（v0.4）。

融合结果的 ``uncertainty.negative_sd / arousal_sd`` 是各模态的分歧度。直觉上「分歧小
= 可信」，但在 CSEMOTIONS 的 5 折留出预测上实测**相反**：分歧度与判对正相关
（Spearman ≈ +0.35；分歧度最低的三分之一样本准确率约 0.48，最高的三分之一约 0.85）。
原因是「六个模态都接近 0.5」也算一致，但那是**没有证据**而非有把握；分歧大通常
意味着至少有强模态（emotion2vec、文本语义）给出了远离中性的信号。

因此本模块**不假设方向**：``config/uncertainty_thresholds.json`` 由
``scripts/evaluate.py emotion`` 在留出折预测上把分歧度三等分，记录每档的实测
象限准确率，并按准确率高低给三档命名 high / medium / low。GUI 显示的是
「可信度 X（该档实测准确率 y）」，让用户看到经验数字而非抽象等级。

阈值文件缺失时返回 ``None``（GUI 不显示等级）。等级来自表演型语料，是经验性的，
不是统计置信区间；分数定义与方向都写在文件里，换语料重跑即自动更新。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import portable

DEFAULT_PATH: Path = portable.CONFIG_DIR / "uncertainty_thresholds.json"
GRADES = ("high", "medium", "low")
GRADE_ZH = {"high": "可信度高", "medium": "可信度中", "low": "可信度低"}


def score_of(uncertainty: dict[str, float]) -> float:
    """分歧度分数：negative 与 arousal 加权 SD 的较大者。"""
    return max(float(uncertainty.get("negative_sd", 0.0)), float(uncertainty.get("arousal_sd", 0.0)))


def load_thresholds(path: Path | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if "cuts" not in data or "terciles" not in data:
        return None
    return data


def grade(uncertainty: dict[str, float] | None, thresholds: dict[str, Any] | None = None,
          path: Path | None = None) -> dict[str, Any] | None:
    """把 uncertainty 映射为等级。

    Returns:
        ``{"grade": "high|medium|low", "label_zh": str, "score": float,
           "accuracy_in_bin": float | None, "tercile": "t0|t1|t2"}``；
        无阈值文件或无不确定性时 ``None``。
    """
    if not uncertainty:
        return None
    th = thresholds if thresholds is not None else load_thresholds(path)
    if not th:
        return None
    s = score_of(uncertainty)
    cuts = th["cuts"]
    t = "t0" if s < cuts[0] else ("t1" if s < cuts[1] else "t2")
    info = (th.get("terciles") or {}).get(t) or {}
    g = info.get("grade", "medium")
    if g not in GRADES:
        g = "medium"
    return {"grade": g, "label_zh": GRADE_ZH[g], "score": round(s, 4),
            "accuracy_in_bin": info.get("accuracy"), "tercile": t}
