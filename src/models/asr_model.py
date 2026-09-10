"""Paraformer-large 中文离线 ASR 封装。

基于 FunASR 加载，内置 VAD（人声端点检测）与标点恢复。输出转写文本、字级时间戳
与**置信度**。

【置信度来源】（v0.4）FunASR 1.0.25 的 ``Paraformer.inference`` 内部用
``cal_decoder_with_predictor`` 得到每个预测位置在词表上的 log-softmax（``am_scores``），
取 argmax 解码，但不把这些后验返回。本封装在模型实例上**包一层**
``cal_decoder_with_predictor``，把最近一次的 ``decoder_out`` 暂存，转写后据此计算
每个保留 token 的最大后验概率，置信度取其均值（``confidence_source="posterior"``）。
若模型对象没有该方法（如测试桩、其他 ASR 后端），回退到文本长度/重复率代理指标
（``confidence_source="proxy"``，v0.1–v0.3 的做法）。

后验均值在干净语音上通常 > 0.9，与 v0.3 之前代理指标的量纲不同；
``settings.json`` 的 ``asr_confidence_threshold`` 已随之调整（见 evaluation.md）。
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
        self._posterior_hooked = self._install_posterior_hook()
        logger.info("ASR 模型就绪（device=%s, 置信度来源=%s）", self.device,
                    "posterior" if self._posterior_hooked else "proxy")

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

    # ------------------------------------------------------------------ #
    # token 后验钩子
    # ------------------------------------------------------------------ #
    def _install_posterior_hook(self) -> bool:
        """把内部 Paraformer 的 cal_decoder_with_predictor 包一层以暂存 decoder_out。"""
        inner = getattr(self.model, "model", None)
        fn = getattr(inner, "cal_decoder_with_predictor", None)
        if inner is None or not callable(fn):
            return False
        if getattr(inner, "_emo_stim_hooked", False):
            return True

        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            try:
                decoder_out, ys_pad_lens = out[0], out[1]
                inner._emo_stim_last = (decoder_out.detach(), ys_pad_lens)
            except Exception:  # noqa: BLE001
                inner._emo_stim_last = None
            return out

        try:
            inner.cal_decoder_with_predictor = wrapped
            inner._emo_stim_hooked = True
            return True
        except Exception:  # noqa: BLE001
            return False

    def _posterior_confidence(self) -> tuple[float, list[float]] | None:
        """由暂存的 decoder_out 计算保留 token 的最大后验均值。"""
        inner = getattr(self.model, "model", None)
        last = getattr(inner, "_emo_stim_last", None)
        if not last:
            return None
        try:
            import torch
            decoder_out, lens = last
            n = int(lens[0]) if lens is not None else decoder_out.shape[1]
            scores = decoder_out[0, :n, :]                # log-softmax
            probs, ids = scores.max(dim=-1)
            probs = torch.exp(probs)
            skip = {int(getattr(inner, "sos", -1)), int(getattr(inner, "eos", -1)),
                    int(getattr(inner, "blank_id", 0))}
            keep = [float(p) for p, t in zip(probs.tolist(), ids.tolist(), strict=False) if int(t) not in skip]
            if not keep:
                return None
            return sum(keep) / len(keep), keep
        except Exception as e:  # noqa: BLE001
            logger.debug("后验置信度计算失败，回退代理指标: %s", e)
            return None

    # ------------------------------------------------------------------ #
    def transcribe(self, wav_path: str) -> dict[str, Any]:
        """转写音频文件。

        Args:
            wav_path: 16kHz 单声道 WAV 路径。

        Returns:
            ``{"text", "confidence", "confidence_source", "token_confidences", "timestamp"}``。
        """
        inner = getattr(self.model, "model", None)
        if inner is not None:
            inner._emo_stim_last = None
        res = self.model.generate(input=wav_path, batch_size_s=300)
        if not res:
            return {"text": "", "confidence": 0.0, "confidence_source": "empty",
                    "token_confidences": [], "timestamp": []}
        text = res[0].get("text", "")
        timestamp = res[0].get("timestamp", [])
        post = self._posterior_confidence() if self._posterior_hooked else None
        if post is not None:
            confidence, tokens = post
            source = "posterior"
        else:
            confidence, tokens, source = self._estimate_confidence(text), [], "proxy"
        return {"text": text, "confidence": float(confidence), "confidence_source": source,
                "token_confidences": tokens, "timestamp": timestamp}

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """代理指标（无后验时回退）：文本长度与重复率。

        规则：
            - 文本过短（<4 字）或为空 -> 低置信度；
            - 重复字符占比高 -> 低置信度；
            - 否则随长度增长趋近 1。
        """
        if not text or len(text) < 4:
            return 0.3
        chars = re.sub(r"[，。！？、,.!?;:\"'()\s]+", "", text)
        if len(chars) < 3:
            return 0.35
        from collections import Counter
        counts = Counter(chars)
        max_repeat = max(counts.values())
        repeat_ratio = max_repeat / len(chars)
        if repeat_ratio > 0.5:
            return 0.4
        length_factor = min(1.0, len(chars) / 8.0)
        confidence = 0.6 + 0.4 * length_factor
        confidence *= (1.0 - 0.3 * repeat_ratio)
        return max(0.3, min(1.0, confidence))
