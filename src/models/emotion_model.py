"""emotion2vec_plus_large 声学情感模型封装。

输出 9 类情绪 softmax 置信度，并按 emotion_mapping.json 的基准负面分/唤醒分
加权聚合为 (s_acoustic, a_acoustic)。unknown 置信度高时向中性收缩。
"""

from __future__ import annotations

import logging
from typing import Any

from src.config_loader import load_emotion_mapping, load_settings
from src.fusion.normalizer import clip01

logger = logging.getLogger("mandarin_emo_stim.emotion")


class EmotionModel:
    """emotion2vec_plus_large 封装。"""

    def __init__(self, device: str = "cuda", model: Any = None):
        self.device = device
        if model is not None:
            self.model = model
        else:
            self._load()
        self.mapping = load_emotion_mapping()["emotion_classes"]
        self.downweight_factor = float(
            load_emotion_mapping().get("unknown_downweight_factor", 0.5)
        )
        logger.info("emotion2vec 模型就绪（device=%s）", self.device)

    def _load(self) -> None:
        from funasr import AutoModel
        settings = load_settings()["models"]
        logger.info("加载 emotion2vec: %s", settings["emotion_model"])
        try:
            self.model = AutoModel(
                model=settings["emotion_model"],
                hub="ms",
                device=self.device,
                model_revision=settings["emotion_revision"],
            )
        except Exception as e:
            logger.warning("加载 large 失败，回退 base: %s", e)
            self.model = AutoModel(
                model=settings["emotion_backoff"],
                hub="ms",
                device=self.device,
            )

    def predict(self, wav_path: str) -> dict[str, Any]:
        """预测音频的情感置信度与聚合分。

        Returns:
            ``{"scores": list[9], "labels": list[str], "s_acoustic": float,
               "a_acoustic": float}``
        """
        res = self.model.generate(
            input=wav_path, granularity="utterance", extract_embedding=False
        )
        if not res:
            return self._neutral()
        scores = res[0].get("scores", [])
        if len(scores) != 9:
            # 兼容部分返回格式
            labels_out = res[0].get("labels", [])
            if labels_out:
                scores = [scores[labels_out.index(c["name_en"])] if c["name_en"] in labels_out
                          else 0.0 for c in self.mapping]
        if len(scores) != 9:
            return self._neutral()

        s_acoustic, a_acoustic = self._aggregate(scores)
        return {
            "scores": scores,
            "labels": [c["name_en"] for c in self.mapping],
            "s_acoustic": s_acoustic,
            "a_acoustic": a_acoustic,
        }

    def _aggregate(self, scores: list[float]) -> tuple[float, float]:
        """按置信度加权聚合 n_base/a_base（排除 unknown）。"""
        # 排除 unknown（id=8）
        idx = [i for i in range(9) if i != 8]
        weight_sum = sum(scores[i] for i in idx)
        if weight_sum <= 0:
            return 0.5, 0.5
        s = sum(scores[i] * self.mapping[i]["n_base"] for i in idx) / weight_sum
        a = sum(scores[i] * self.mapping[i]["a_base"] for i in idx) / weight_sum

        # unknown 置信度高时向中性收缩
        unknown_conf = scores[8] if len(scores) > 8 else 0.0
        if unknown_conf > 0.5:
            s = s * self.downweight_factor + 0.25
            a = a * self.downweight_factor + 0.25
        return clip01(s), clip01(a)

    def _neutral(self) -> dict[str, Any]:
        return {
            "scores": [0.0] * 8 + [1.0],
            "labels": [c["name_en"] for c in self.mapping],
            "s_acoustic": 0.5,
            "a_acoustic": 0.5,
        }
