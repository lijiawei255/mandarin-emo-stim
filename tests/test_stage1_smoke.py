"""阶段 1 冒烟测试：环境、配置与便携模式可用性。"""

import json
import os

import pytest


def test_config_jsons_parse():
    """三个配置文件均为合法 JSON 且结构完整。"""
    from src import portable

    for path, key in [
        (portable.SETTINGS_PATH, "fusion_weights"),
        (portable.EMOTION_MAPPING_PATH, "emotion_classes"),
        (portable.STIMULUS_PARAMS_PATH, "quadrant_anchors"),
    ]:
        assert path.exists(), f"配置文件缺失: {path}"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert key in data, f"{path.name} 缺少键 {key}"


def test_fusion_weights_sum_to_one():
    """默认融合权重之和精确为 1。"""
    from src.config_loader import load_settings

    w = load_settings()["fusion_weights"]
    for axis in ("negative", "arousal"):
        total = sum(w[axis].values())
        assert abs(total - 1.0) < 1e-9, f"{axis} 权重和={total}，应为 1.0"


def test_emotion_mapping_nine_classes():
    """emotion_mapping 含 9 类情绪。"""
    from src.config_loader import load_emotion_mapping

    classes = load_emotion_mapping()["emotion_classes"]
    assert len(classes) == 9
    assert [c["id"] for c in classes] == list(range(9))


def test_portable_env_redirects_hf():
    """便携模式把 HF 缓存重定向到 portable_data 并启用国内镜像。"""
    from src import portable  # 导入即触发 apply_env_overrides

    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"
    assert "portable_data" in os.environ.get("HF_HOME", "")


def test_core_python_deps_importable():
    """关键依赖可正常导入（不含重型模型推理）。"""
    import librosa  # noqa: F401
    import numpy as np  # noqa: F401
    import scipy  # noqa: F401
    import slab  # noqa: F401
    import jieba  # noqa: F401
    import soundfile  # noqa: F401
    import sounddevice  # noqa: F401
    assert np.__version__.startswith("1.26")


def test_torch_cuda_available():
    """torch 已安装且 CUDA 可用（本版本目标：Windows + NVIDIA GPU）。"""
    import torch

    assert torch.cuda.is_available(), "CUDA 不可用，本版本要求 NVIDIA GPU"
    assert torch.version.cuda is not None
