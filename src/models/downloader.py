"""模型自动下载器。

支持三种来源：
    - FunASR 模型（Paraformer / emotion2vec）：通过 ModelScope（``hub="ms"``）下载，国内直连。
    - Qwen3 LLM：通过 transformers 从 hf-mirror.com 下载（HF_ENDPOINT 已在 portable 设置）。
    - PANNs CNN10 checkpoint：从 Zenodo 下载，支持重试与大小校验。

所有下载均落到 ``portable_data/models/`` 下，断点续传由各 SDK / HTTP Range 提供，
叠加 3 次指数退避重试 + 关键文件大小校验。
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Callable

from src import portable

logger = logging.getLogger("mandarin_emo_stim.downloader")

# PANNs CNN10 checkpoint（Zenodo）
PANNS_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth"
)
PANNS_CHECKPOINT_SIZE = 30  # MB（约，用于校验下限）

ProgressCallback = Callable[[str, int], None]


def download_panns_checkpoint(progress_cb: ProgressCallback | None = None) -> Path:
    """下载 PANNs CNN10 checkpoint 到 ``portable_data/models/panns/``。

    Returns:
        checkpoint 文件路径。
    """
    portable.PANNS_DIR.mkdir(parents=True, exist_ok=True)
    dest = portable.PANNS_DIR / "Cnn10_mAP=0.380.pth"
    if dest.exists() and dest.stat().st_size > PANNS_CHECKPOINT_SIZE * 1024 * 1024:
        logger.info("PANNs checkpoint 已存在，跳过下载: %s", dest)
        return dest

    _download_with_retry(PANNS_CHECKPOINT_URL, dest, progress_cb, "PANNs")
    return dest


def _download_with_retry(url: str, dest: Path, progress_cb: ProgressCallback | None,
                         name: str, max_retries: int = 3) -> None:
    """带指数退避重试的 HTTP 下载（支持 Range 断点续传）。"""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if progress_cb:
                progress_cb(f"下载 {name}（第 {attempt} 次）", 0)
            _download_range(url, dest)
            if progress_cb:
                progress_cb(f"{name} 下载完成", 100)
            logger.info("%s 下载完成: %s", name, dest)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("%s 下载失败（第 %d 次）: %s", name, attempt, e)
            if attempt < max_retries:
                wait = 5 * (2 ** (attempt - 1))  # 5s, 15s, 45s
                time.sleep(wait)
    raise RuntimeError(f"{name} 下载失败（重试 {max_retries} 次）: {last_err}")


def _download_range(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """支持断点续传的单次下载。"""
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.getheader("Content-Length")
        total = int(total) + existing if total else None
        mode = "ab" if existing > 0 and resp.status == 206 else "wb"
        if mode == "wb":
            existing = 0
        with open(dest, mode) as f:
            downloaded = existing
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
    logger.info("下载 %s -> %s（%d 字节）", url, dest, dest.stat().st_size)


def ensure_all_models(progress_cb: ProgressCallback | None = None) -> dict[str, Path]:
    """确保全部模型就绪（FunASR / Qwen3 由各自 SDK 在加载时按需下载，
    PANNs checkpoint 需显式下载）。

    Returns:
        ``{"panns": <path>}`` 等本地路径映射。
    """
    panns_path = download_panns_checkpoint(progress_cb)
    return {"panns": panns_path}
