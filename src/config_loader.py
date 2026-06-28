"""配置文件加载工具。

统一加载 ``config/`` 目录下的 JSON 配置，避免各模块重复读取逻辑。
配置路径定义在 :mod:`src.portable` 中，保证不硬编码绝对路径。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src import portable


def _load_json(path: portable.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """加载 ``config/settings.json``。"""
    return _load_json(portable.SETTINGS_PATH)


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
