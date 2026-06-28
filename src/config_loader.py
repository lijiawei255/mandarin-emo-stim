"""配置文件加载工具。

统一加载 ``config/`` 目录下的 JSON 配置，避免各模块重复读取逻辑。
配置路径定义在 :mod:`src.portable` 中，保证不硬编码绝对路径。

``load_settings`` 会做基本的 schema 校验：缺失必需键时抛出清晰的
``ConfigError``，而非在后续使用中冒出晦涩的 ``KeyError``。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src import portable


class ConfigError(ValueError):
    """配置缺失或非法时抛出。"""


# settings.json 必需的键结构（顶层键 → 二级键）。缺失时给出明确错误。
_REQUIRED_SETTINGS = {
    "fusion_weights": ["negative", "arousal"],
    "audio": ["sample_rate", "stimulus_sample_rate",
              "stimulus_duration_sec", "stimulus_min_sec", "stimulus_max_sec",
              "record_duration_max", "chunk_size"],
    "models": ["device", "asr_device", "asr_model", "emotion_model",
               "llm_model", "panns_checkpoint", "max_new_tokens"],
    "thresholds": ["snr_warning_db", "snr_weight_floor",
                   "paralang_confidence_threshold", "asr_confidence_threshold",
                   "quadrant_mid_v", "quadrant_mid_a", "quadrant_band"],
    "stimulus": ["max_peak_dbfs", "fade_ms", "crossfade_ms", "haas_delay_ms"],
    "history": ["max_records"],
}


def _load_json(path: portable.Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件 {path} 不是合法 JSON: {e}") from e


def _validate_settings(data: dict[str, Any]) -> None:
    """校验 settings.json 必需键结构。"""
    missing = []
    for top_key, sub_keys in _REQUIRED_SETTINGS.items():
        if top_key not in data:
            missing.append(top_key)
            continue
        for sub in sub_keys:
            if sub not in data[top_key]:
                missing.append(f"{top_key}.{sub}")
    if missing:
        raise ConfigError(
            "settings.json 缺少必需配置项：" + "、".join(missing)
            + "。请对照 config/settings.json 模板补全。"
        )


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """加载并校验 ``config/settings.json``。"""
    data = _load_json(portable.SETTINGS_PATH)
    _validate_settings(data)
    return data


@lru_cache(maxsize=1)
def load_emotion_mapping() -> dict[str, Any]:
    """加载 ``config/emotion_mapping.json``。"""
    return _load_json(portable.EMOTION_MAPPING_PATH)


@lru_cache(maxsize=1)
def load_stimulus_params() -> dict[str, Any]:
    """加载 ``config/stimulus_params.json``。"""
    return _load_json(portable.STIMULUS_PARAMS_PATH)


def reload_all() -> None:
    """清空缓存，强制重新读取全部配置（测试/运行时改配置后调用）。"""
    load_settings.cache_clear()
    load_emotion_mapping.cache_clear()
    load_stimulus_params.cache_clear()
