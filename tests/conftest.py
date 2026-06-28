"""pytest 全局配置。"""

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，便于测试直接 import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """注册自定义标记。"""
    config.addinivalue_line(
        "markers", "gpu: 需要 NVIDIA GPU 与已下载模型的测试（默认跳过）"
    )
    config.addinivalue_line(
        "markers", "slow: 耗时较长的测试（默认跳过）"
    )


def pytest_collection_modifyitems(config, items):
    """未显式指定 -m 时，默认跳过 gpu / slow 标记的测试。"""
    skip_markers = ("gpu", "slow")
    selected = config.getoption("-m")
    if selected:  # 用户显式指定了 marker 过滤，不做自动跳过
        return

    import pytest
    skip_gpu = pytest.mark.skip(reason="需要 GPU/模型，使用 -m gpu 显式运行")
    skip_slow = pytest.mark.skip(reason="耗时测试，使用 -m slow 显式运行")
    for item in items:
        if any(m in {kw.name for kw in item.iter_markers()} for m in skip_markers):
            if "gpu" in {kw.name for kw in item.iter_markers()}:
                item.add_marker(skip_gpu)
            if "slow" in {kw.name for kw in item.iter_markers()}:
                item.add_marker(skip_slow)
