"""独立的模型预下载脚本。

预先下载全部模型到 ``portable_data/models/``，便于调试与首次启动加速。
用法::

    python scripts/download_models.py

下载来源（国内网络友好）：
    - Paraformer-large / emotion2vec_plus_large : ModelScope（直连）
    - Qwen3-1.7B-Instruct : hf-mirror.com（通过 HF_ENDPOINT）
    - PANNs CNN10 checkpoint : Zenodo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import portable  # noqa: F401  (触发环境变量重定向)
from src.config_loader import load_settings


def progress(stage: str, pct: int) -> None:
    print(f"[模型下载] {stage}: {pct}%", flush=True)


def download_funasr_models() -> None:
    """通过 FunASR（ModelScope）下载 Paraformer + emotion2vec。"""
    from funasr import AutoModel

    settings = load_settings()["models"]
    print("=== 下载 Paraformer-large ASR ===", flush=True)
    AutoModel(
        model=settings["asr_model"],
        hub="ms",
        model_revision=settings["asr_revision"],
        device="cpu",  # 仅下载，不占用 GPU
    )
    print("=== 下载 emotion2vec_plus_large ===", flush=True)
    AutoModel(
        model=settings["emotion_model"],
        hub="ms",
        model_revision=settings["emotion_revision"],
        device="cpu",
    )


def download_qwen() -> None:
    """通过 transformers（hf-mirror.com）下载 Qwen3-1.7B-Instruct。"""
    from transformers import AutoTokenizer

    settings = load_settings()["models"]
    model_name = settings["llm_model"]
    print(f"=== 下载 {model_name}（tokenizer + 权重） ===", flush=True)
    # 仅下载 tokenizer 与权重文件（不加载到显存）
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 触发权重文件下载（snapshot_download）
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_name,
        local_dir=str(portable.HF_CACHE_DIR / model_name.replace("/", "_")),
        local_dir_use_symlinks=False,
    )


def download_panns() -> None:
    """下载 PANNs CNN10 checkpoint。"""
    from src.models.downloader import download_panns_checkpoint
    print("=== 下载 PANNs CNN10 checkpoint ===", flush=True)
    download_panns_checkpoint(progress)


def main() -> int:
    t0 = time.time()
    try:
        download_funasr_models()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] FunASR 模型下载失败: {e}", flush=True)
    try:
        download_qwen()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] Qwen3 下载失败: {e}", flush=True)
    try:
        download_panns()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] PANNs checkpoint 下载失败: {e}", flush=True)
    print(f"=== 全部下载流程结束，耗时 {time.time() - t0:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
