"""音频文件加载与重采样。

支持 WAV/MP3/FLAC，自动重采样到 16kHz 单声道（模型输入标准）。
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.config_loader import load_settings

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载音频文件并重采样到目标采样率、单声道。

    Args:
        path: 音频文件路径。
        target_sr: 目标采样率（默认 16000）。

    Returns:
        ``(y, sr)`` —— y 为 float32 单声道波形，sr 为 target_sr。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 格式不支持。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的音频格式: {path.suffix}（支持 {SUPPORTED_FORMATS}）")

    # librosa.load 自动处理多格式解码、单声道混合、重采样
    # 使用 soxr_hq（高质量、快速，soxr 已随 librosa 安装）；kaiser_best 需额外的 resampy
    y, sr = librosa.load(str(path), sr=target_sr, mono=True, res_type="soxr_hq")
    return np.ascontiguousarray(y.astype(np.float32)), sr


def save_wav(y: np.ndarray, sr: int, path: str | Path,
             subtype: str = "PCM_16") -> Path:
    """保存波形为 WAV 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
    return path


def get_duration(y: np.ndarray, sr: int) -> float:
    """有效语音时长（秒）。"""
    return float(len(y) / sr) if sr > 0 else 0.0


def default_target_sr() -> int:
    """配置中的默认目标采样率。"""
    return int(load_settings()["audio"]["sample_rate"])
