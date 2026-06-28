"""便携模式：所有运行时产生的数据均重定向到项目根目录下的 ``portable_data/``。

设计目标：
1. 不污染用户系统的其它目录（绿色便携），删除项目文件夹即可清除全部数据。
2. 模型文件、用户录音、历史记录、日志等隐私/大文件全部集中在 ``portable_data/``，
   该目录通过 ``.gitignore`` 排除，不进入 Git 仓库。

本模块应在程序启动时最先导入，确保各类缓存目录在第三方库读取环境变量前就已设置好。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录：本文件位于 <root>/src/portable.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# 便携数据根目录
PORTABLE_DATA_ROOT: Path = PROJECT_ROOT / "portable_data"

# 各类子目录（懒创建）
MODELS_DIR: Path = PORTABLE_DATA_ROOT / "models"
FUNASR_DIR: Path = MODELS_DIR / "funasr"
MODELSCOPE_DIR: Path = MODELS_DIR / "modelscope"
HF_CACHE_DIR: Path = MODELS_DIR / "huggingface"
PANNS_DIR: Path = MODELS_DIR / "panns"
GGUF_DIR: Path = MODELS_DIR / "gguf"

HISTORY_DIR: Path = PORTABLE_DATA_ROOT / "history"
HISTORY_DB_PATH: Path = HISTORY_DIR / "records.db"
HISTORY_AUDIO_DIR: Path = HISTORY_DIR / "audio"
HISTORY_STIMULI_DIR: Path = HISTORY_DIR / "stimuli"

LOGS_DIR: Path = PORTABLE_DATA_ROOT / "logs"
TEMP_DIR: Path = PORTABLE_DATA_ROOT / "temp"

# 配置目录
CONFIG_DIR: Path = PROJECT_ROOT / "config"
SETTINGS_PATH: Path = CONFIG_DIR / "settings.json"
EMOTION_MAPPING_PATH: Path = CONFIG_DIR / "emotion_mapping.json"
STIMULUS_PARAMS_PATH: Path = CONFIG_DIR / "stimulus_params.json"

# 资源目录
RESOURCES_DIR: Path = PROJECT_ROOT / "resources"
DICTIONARIES_DIR: Path = RESOURCES_DIR / "dictionaries"

# HF 镜像端点（国内网络环境下 HuggingFace 不可达，统一走 hf-mirror.com）
HF_MIRROR_ENDPOINT: str = "https://hf-mirror.com"


def ensure_dirs() -> None:
    """创建全部便携数据子目录（幂等）。"""
    for d in (
        PORTABLE_DATA_ROOT,
        MODELS_DIR,
        FUNASR_DIR,
        MODELSCOPE_DIR,
        HF_CACHE_DIR,
        PANNS_DIR,
        GGUF_DIR,
        HISTORY_DIR,
        HISTORY_AUDIO_DIR,
        HISTORY_STIMULI_DIR,
        LOGS_DIR,
        TEMP_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def apply_env_overrides() -> None:
    """把各类缓存目录与镜像端点写入环境变量。

    必须在加载 ModelScope / transformers / FunASR 之前调用，
    以确保模型下载与缓存都落到 ``portable_data/models/`` 下。
    """
    ensure_dirs()

    # HuggingFace / transformers 缓存重定向
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PORTABLE_DATA_ROOT / "datasets"))

    # 国内网络：强制走 hf-mirror.com（仅当用户未显式设置时）
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)

    # ModelScope 缓存重定向
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_DIR))

    # FunASR 缓存重定向
    os.environ.setdefault("FUNASR_CACHE", str(FUNASR_DIR))

    # PANNs 默认 checkpoint 目录
    os.environ.setdefault("PANNS_CACHE", str(PANNS_DIR))

    # 临时目录重定向（numba / matplotlib 等会用到）
    os.environ.setdefault("TMPDIR", str(TEMP_DIR))
    os.environ.setdefault("TEMP", str(TEMP_DIR))
    os.environ.setdefault("TMP", str(TEMP_DIR))


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置按天滚动的文件日志 + 控制台日志。

    日志文件存于 ``portable_data/logs/app_YYYY-MM-DD.log``。
    """
    ensure_dirs()
    log_file = LOGS_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    # 避免重复添加 handler（重复导入时）
    if not root.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    return logging.getLogger("mandarin_emo_stim")


# 导入即生效：保证任何依赖该模块的代码都已具备正确的环境
apply_env_overrides()
