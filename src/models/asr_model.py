"""Paraformer-large 中文离线 ASR 封装。

基于 FunASR 加载，内置 VAD（人声端点检测）与标点恢复。
输出转写文本与置信度（通过文本长度/有效字符占比近似估计）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config_loader import load_settings

logger = logging.getLogger("mandarin_emo_stim.asr")


class ASRModel:
    """Paraformer-large ASR 封装。"""

    def __init__(self, device: str = "cuda", model: Any = None):
        """
        Args:
            device: 推理设备（``"cuda"`` / ``"cpu"``）。
            model: 已加载的 FunASR AutoModel（用于测试注入）；为 ``None`` 时按配置加载。
        """
        self.device = device
        if model is not None:
            self.model = model
        else:
            self._load()
        logger.info("ASR 模型就绪（device=%s）", self.device)

    def _load(self) -> None:
        from funasr import AutoModel
        settings = load_settings()["models"]
        logger.info("加载 Paraformer-large ASR: %s", settings["asr_model"])
        self.model = AutoModel(
            model=settings["asr_model"],
            hub="ms",
            device=self.device,
            model_revision=settings["asr_revision"],
        )

    def transcribe(self, wav_path: str) -> dict[str, Any]:
        """转写音频文件。

        Args:
            wav_path: 16kHz 单声道 WAV 路径。

        Returns:
            ``{"text": str, "confidence": float, "timestamp": list}``。
        """
        res = self.model.generate(input=wav_path, batch_size_s=300)
        if not res:
            return {"text": "", "confidence": 0.0, "timestamp": []}
        text = res[0].get("text", "")
        timestamp = res[0].get("timestamp", [])
        confidence = self._estimate_confidence(text)
        return {"text": text, "confidence": confidence, "timestamp": timestamp}

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """通过文本长度与有效字符占比近似估计 ASR 置信度（非模型直接输出）。

        规则：
            - 文本过短（<4 字）或为空 -> 低置信度；
            - 重复字符占比高 -> 低置信度；
            - 否则随长度增长趋近 1。
        """
        if not text or len(text) < 4:
            return 0.3
        # 去除标点后的有效字符
        chars = re.sub(r"[，。！？、,.!?;:\"'()\s]+", "", text)
        if len(chars) < 3:
            return 0.35
        # 重复字符占比
        from collections import Counter
        counts = Counter(chars)
        max_repeat = max(counts.values())
        repeat_ratio = max_repeat / len(chars)
        if repeat_ratio > 0.5:
            return 0.4
        # 长度因子：8 字以上趋于高置信
        length_factor = min(1.0, len(chars) / 8.0)
        confidence = 0.6 + 0.4 * length_factor
        # 重复惩罚
        confidence *= (1.0 - 0.3 * repeat_ratio)
        return max(0.3, min(1.0, confidence))
