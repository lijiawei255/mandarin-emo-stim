"""用户级基线校准：按**受试者档案**保存的个人中性基线。

【为何需要】语料级中性校准（``config/modality_calibration.json``）只能消除模型对
「一般中文语音」的系统偏置；不同说话人的嗓音（F0 高低、气声程度、语速习惯）
会让同一模态在**个人**中性状态下的分数再次偏离 0.5。情感计算中的 speaker
normalization（Schuller et al. 2011）正是为此。v0.3 的评测用 CSEMOTIONS 每位配音员
的中性句验证了这一做法：折间标准差从 0.076 降到 0.014（docs/evaluation.md §0.1）。

【档案（v0.4）】基线**必须能预存**：受试者来做实验时往往已不平静，基线应在另一个
平静的场合录好、按人保存，实验当天选用。因此：

- 每位受试者一个档案：``portable_data/calibration/profiles/<name>.json``，
  内含多次录音的原始模态均值（``samples``）与由其平均得到的偏移（``offsets``）；
  多次录音取平均可降低单次 30 s 的抽样误差。
- ``active_profile.json`` 记录当前选用的档案；GUI 输入区有「受试者档案」下拉框，
  CLI 用 ``--profile NAME``。无档案时回到语料级校准。
- 兼容 v0.3 的单文件 ``user_baseline.json``：无激活档案而该文件存在时仍生效。

【做法】受试者用平静、日常的语气朗读约 30 s，跑一次完整管线得到各模态的**原始**分，
offset = 0.5 − 个人均值（钳制 ±0.3）。融合时个人偏移**替代**语料偏移（对有个人
数据的模态），其余模态仍用语料偏移。基线应在与实际使用相同的设备与环境下录制。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src import portable

MODALITIES = ("acoustic", "prosody", "paralang", "physical", "text_llm", "text_stat")
OFFSET_CLIP = 0.3
CALIB_DIR: Path = portable.PORTABLE_DATA_ROOT / "calibration"
PROFILES_DIR: Path = CALIB_DIR / "profiles"
ACTIVE_PATH: Path = CALIB_DIR / "active_profile.json"
DEFAULT_PATH: Path = CALIB_DIR / "user_baseline.json"   # v0.3 单文件基线（兼容）

_NAME_RE = re.compile(r"^[\w\-一-鿿]{1,40}$")


# ---------------------------------------------------------------------- #
# 基本运算
# ---------------------------------------------------------------------- #
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


def _offsets_to_tuples(offsets: dict[str, Any]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for m, v in (offsets or {}).items():
        if m in MODALITIES and isinstance(v, dict):
            out[m] = (max(-OFFSET_CLIP, min(OFFSET_CLIP, float(v.get("negative", 0.0)))),
                      max(-OFFSET_CLIP, min(OFFSET_CLIP, float(v.get("arousal", 0.0)))))
    return out


# ---------------------------------------------------------------------- #
# 受试者档案
# ---------------------------------------------------------------------- #
def validate_name(name: str) -> str:
    """档案名：1–40 个字母/数字/下划线/连字符/汉字。返回去除首尾空白的名字。"""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError("档案名只能含字母、数字、下划线、连字符或汉字，长度 1–40")
    return name


def profile_path(name: str, profiles_dir: Path | None = None) -> Path:
    return Path(profiles_dir or PROFILES_DIR) / f"{validate_name(name)}.json"


def list_profiles(profiles_dir: Path | None = None) -> list[str]:
    d = Path(profiles_dir or PROFILES_DIR)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def save_profile(name: str, raw_scores_list: list[dict[str, dict[str, float]]],
                 meta: dict[str, Any] | None = None, append: bool = True,
                 profiles_dir: Path | None = None) -> Path:
    """把一次或多次录音的原始模态分并入档案并重算偏移。

    Args:
        name: 档案名（受试者编号）。
        raw_scores_list: 本次新增的录音结果（每条一次录音）。
        meta: 记录到本次样本上的说明（来源文件、时长等）。
        append: True 则与档案中已有样本合并取平均；False 则覆盖。
    """
    path = profile_path(name, profiles_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    created = datetime.now().isoformat(timespec="seconds")
    if append and path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            samples = list(old.get("samples") or [])
            created = (old.get("_meta") or {}).get("created", created)
        except (OSError, ValueError):
            samples = []
    stamp = datetime.now().isoformat(timespec="seconds")
    for raw in raw_scores_list:
        samples.append({"recorded": stamp, "raw": raw, **(meta or {})})
    offsets = compute_offsets([s["raw"] for s in samples])
    payload = {
        "_meta": {"name": validate_name(name), "created": created, "updated": stamp,
                  "n_samples": len(samples),
                  "definition": f"offset = 0.5 - mean of personal neutral raw scores, clipped to ±{OFFSET_CLIP}"},
        "offsets": offsets,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_profile(name: str, profiles_dir: Path | None = None) -> dict[str, tuple[float, float]]:
    """读取档案偏移；不存在/损坏返回空 dict。"""
    try:
        data = json.loads(profile_path(name, profiles_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _offsets_to_tuples(data.get("offsets"))


def profile_info(name: str, profiles_dir: Path | None = None) -> dict[str, Any]:
    """档案元信息（样本数、时间），供 GUI 展示；不存在返回空 dict。"""
    try:
        data = json.loads(profile_path(name, profiles_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data.get("_meta") or {})


def delete_profile(name: str, profiles_dir: Path | None = None,
                   active_path: Path | None = None) -> bool:
    p = profile_path(name, profiles_dir)
    if get_active(active_path) == validate_name(name):
        set_active(None, active_path)
    if p.exists():
        p.unlink()
        return True
    return False


def get_active(active_path: Path | None = None) -> str | None:
    try:
        data = json.loads(Path(active_path or ACTIVE_PATH).read_text(encoding="utf-8"))
        name = data.get("profile")
        return validate_name(name) if name else None
    except (OSError, ValueError):
        return None


def set_active(name: str | None, active_path: Path | None = None) -> None:
    p = Path(active_path or ACTIVE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"profile": validate_name(name) if name else None,
                             "updated": datetime.now().isoformat(timespec="seconds")},
                            ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------- #
# 融合层入口
# ---------------------------------------------------------------------- #
def load_active(profiles_dir: Path | None = None, active_path: Path | None = None,
                legacy_path: Path | None = None) -> tuple[dict[str, tuple[float, float]], str | None]:
    """返回 (当前生效的个人偏移, 档案名)。

    优先激活的档案；无激活档案而 v0.3 的单文件基线存在时返回其偏移并以
    ``"user_baseline"`` 作为名字；都没有则 ``({}, None)``。
    """
    name = get_active(active_path)
    if name:
        off = load_profile(name, profiles_dir)
        if off:
            return off, name
    off = load(legacy_path)
    return (off, "user_baseline") if off else ({}, None)


# ---------------------------------------------------------------------- #
# v0.3 单文件基线（兼容保留）
# ---------------------------------------------------------------------- #
def save(offsets: dict[str, dict[str, float]], path: Path | None = None,
         meta: dict[str, Any] | None = None) -> Path:
    """保存单文件基线（v0.3 兼容接口）。"""
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
    """读取单文件基线；文件缺失/损坏返回空 dict。"""
    try:
        data = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _offsets_to_tuples(data.get("offsets"))


def clear(path: Path | None = None) -> bool:
    """删除单文件基线。返回是否确实删除了文件。"""
    path = Path(path or DEFAULT_PATH)
    if path.exists():
        path.unlink()
        return True
    return False
