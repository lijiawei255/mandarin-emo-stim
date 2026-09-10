"""用户级基线校准（个人中性基线）。

【为何需要】语料级中性校准（``config/modality_calibration.json``）只能消除模型对
「一般中文语音」的系统偏置；不同说话人的嗓音（F0 高低、气声程度、语速习惯）
会让同一模态在**个人**中性状态下的分数再次偏离 0.5。情感计算中的 speaker
normalization（Schuller et al. 2011）正是为此：用说话人自己的中性样本做基线。
v0.3 的评测用 CSEMOTIONS 每位配音员的中性句做「说话人级基线」验证了这一做法
（见 docs/evaluation.md §0）。

【做法】用户录一段（默认 30 s）平静朗读，跑一次完整管线得到各模态的**原始**分，
offset = 0.5 − 个人均值（钳制 ±0.3），保存到 ``portable_data/calibration/user_baseline.json``。
融合时个人偏移**替代**语料偏移（对有个人数据的模态），其余模态仍用语料偏移。

【边界】30 s 单次录音的估计有抽样误差；建议用多段录音求均值（本模块支持传入多条
结果）。个人基线应在与实际使用相同的设备与环境下录制。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src import portable

MODALITIES = ("acoustic", "prosody", "paralang", "physical", "text_llm", "text_stat")
OFFSET_CLIP = 0.3
DEFAULT_PATH: Path = portable.PORTABLE_DATA_ROOT / "calibration" / "user_baseline.json"


def compute_offsets(raw_scores_list: list[dict[str, dict[str, float]]],
                    exclude: set[str] | frozenset[str] = frozenset({"paralang"}),
                    clip: float = OFFSET_CLIP) -> dict[str, dict[str, float]]:
    """由一条或多条分析结果的 ``modal_scores_raw`` 计算个人偏移。

    Args:
        raw_scores_list: 每条为 ``{模态: {"negative": x, "arousal": y}}``。
        exclude: 不做个人校准的模态。默认排除 paralang：副语言事件在平静朗读中
            本就应为 0.5，校准无意义且会把偶发事件误当基线。
        clip: 偏移钳制范围。

    Returns:
        ``{模态: {"negative": offset, "arousal": offset}}``，仅含有数据的模态。
    """
    if not raw_scores_list:
        return {}
    out: dict[str, dict[str, float]] = {}
    for m in MODALITIES:
        if m in exclude:
            continue
        vals = [r[m] for r in raw_scores_list if m in r]
        if not vals:
            continue
        mean_s = sum(float(v["negative"]) for v in vals) / len(vals)
        mean_a = sum(float(v["arousal"]) for v in vals) / len(vals)
        out[m] = {
            "negative": round(max(-clip, min(clip, 0.5 - mean_s)), 4),
            "arousal": round(max(-clip, min(clip, 0.5 - mean_a)), 4),
        }
    return out


def save(offsets: dict[str, dict[str, float]], path: Path | None = None,
         meta: dict[str, Any] | None = None) -> Path:
    """保存个人基线（含时间与来源说明）。"""
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {"created": datetime.now().isoformat(timespec="seconds"),
                  "definition": f"offset = 0.5 - personal neutral mean, clipped to ±{OFFSET_CLIP}",
                  **(meta or {})},
        "offsets": offsets,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """读取个人基线；文件缺失/损坏返回空 dict。返回 ``{模态: (ds, da)}``。"""
    path = Path(path or DEFAULT_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for m, v in (data.get("offsets") or {}).items():
        if m in MODALITIES and isinstance(v, dict):
            out[m] = (max(-OFFSET_CLIP, min(OFFSET_CLIP, float(v.get("negative", 0.0)))),
                      max(-OFFSET_CLIP, min(OFFSET_CLIP, float(v.get("arousal", 0.0)))))
    return out


def clear(path: Path | None = None) -> bool:
    """删除个人基线文件。返回是否确实删除了文件。"""
    path = Path(path or DEFAULT_PATH)
    if path.exists():
        path.unlink()
        return True
    return False
