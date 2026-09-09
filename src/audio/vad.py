"""语音端点检测（VAD）封装。

复用 Paraformer-large 内置 VAD 的时间戳输出，不单独加载 FSMN VAD 模型。
提供语音段裁剪与有效语音时长统计。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def segments_from_timestamps(timestamps: list[list[int]]) -> list[tuple[float, float]]:
    """把 Paraformer 输出的时间戳（ms 整数对列表）转为 (start_sec, end_sec)。"""
    segments = []
    for ts in timestamps or []:
        if len(ts) >= 2:
            segments.append((ts[0] / 1000.0, ts[1] / 1000.0))
    return segments


def effective_duration(segments: list[tuple[float, float]]) -> float:
    """语音段总时长（秒）。"""
    return float(sum(end - start for start, end in segments))


def extract_voiced(y: np.ndarray, sr: int,
                   segments: list[tuple[float, float]]) -> np.ndarray:
    """根据语音段裁剪出有效人声波形（拼接各段）。

    若无语音段，返回原始波形（避免完全丢弃）。
    """
    if not segments:
        return y
    parts = []
    for start, end in segments:
        i0 = max(0, int(start * sr))
        i1 = min(len(y), int(end * sr))
        if i1 > i0:
            parts.append(y[i0:i1])
    if not parts:
        return y
    return np.concatenate(parts)


_PUNCT = set("，。！？、；：,.!?;:\"'“”‘’（）()《》<>【】[]…—-— \t\n")


def syllable_rate(asr_result: dict[str, Any]) -> float | None:
    """由 ASR 字级时间戳直接计算音节率（音节/秒）。

    中文一字一音节，音节数 = 去标点后的字符数；时长 = 各时间戳段的并集长度。
    这比能量包络穿越计数（``prosody._estimate_speech_rate``）更接近标准的
    音节率定义（de Jong & Wempe 2009 以音节核计数），且无需额外模型。

    Returns:
        音节率；文本为空或无时间戳时返回 ``None``（调用方回退到能量法）。
    """
    text = asr_result.get("text") or ""
    n_syl = sum(1 for ch in text if ch not in _PUNCT and not ch.isspace())
    segments = segments_from_timestamps(asr_result.get("timestamp", []))
    dur = effective_duration(segments)
    if n_syl == 0 or dur <= 0:
        return None
    return float(n_syl / dur)


def vad_from_asr_result(y: np.ndarray, sr: int,
                        asr_result: dict[str, Any]) -> dict[str, Any]:
    """从 ASR 结果中提取 VAD 信息并裁剪有效语音。

    Args:
        y: 原始波形。
        sr: 采样率。
        asr_result: ASRModel.transcribe 的输出（含 timestamp）。

    Returns:
        ``{"segments": [...], "effective_duration": float, "voiced_y": np.ndarray}``
    """
    segments = segments_from_timestamps(asr_result.get("timestamp", []))
    voiced = extract_voiced(y, sr, segments)
    return {
        "segments": segments,
        "effective_duration": effective_duration(segments),
        "voiced_y": voiced,
    }
