"""多特征加权融合模块。

输入 6 个模态的 (负面分 s, 唤醒分 a)，结合音频质量（SNR）、ASR 置信度、
副语言事件强度，进行动态权重调整后加权融合，输出标准化情绪指标。

模态名（与配置 fusion_weights 对应）：
    acoustic  : emotion2vec 声学情感
    prosody   : parselmouth 韵律
    paralang  : PANNs 副语言
    physical  : librosa 物理声学
    text_llm  : Qwen3 文本语义
    text_stat : jieba 文本统计

动态调整规则（详见项目计划文档 3.4 节）：
    1. 低 SNR：降低声学模态权重，提升文本模态权重，再归一化。
    2. 低 ASR 置信度：降低文本模态权重，再归一化。
    3. 极端情况（SNR<5dB 且 ASR<0.3）：所有模态平均分配。
    4. 强副语言事件：副语言模态权重 ×1.5，再归一化。
"""

from __future__ import annotations

from typing import Any

from src.fusion.quadrant import (
    compute_quadrant_memberships,
    dominant_quadrant,
)

# 模态列表（固定顺序）
MODALITIES = ("acoustic", "prosody", "paralang", "physical", "text_llm", "text_stat")
ACOUSTIC_MODALITIES = ("acoustic", "prosody", "paralang", "physical")
TEXT_MODALITIES = ("text_llm", "text_stat")


class WeightedFusion:
    """6 模态加权融合器。"""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: settings.json 全部内容（读取 fusion_weights 与 thresholds）。
        """
        self.weights_s: dict[str, float] = dict(config["fusion_weights"]["negative"])
        self.weights_a: dict[str, float] = dict(config["fusion_weights"]["arousal"])
        thr = config["thresholds"]
        self.snr_weight_floor = thr["snr_weight_floor"]
        self.asr_confidence_threshold = thr["asr_confidence_threshold"]
        self.paralang_boost_threshold = thr["paralang_boost_threshold"]
        self.mid_v = thr["quadrant_mid_v"]
        self.mid_a = thr["quadrant_mid_a"]
        self.band = thr["quadrant_band"]

    # ------------------------------------------------------------------ #
    # 动态权重调整
    # ------------------------------------------------------------------ #
    def _normalize(self, w: dict[str, float]) -> dict[str, float]:
        total = sum(w.values())
        if total <= 0:
            n = len(w)
            return {k: 1.0 / n for k in w}
        return {k: v / total for k, v in w.items()}

    def _apply_snr(self, w_s: dict[str, float], w_a: dict[str, float],
                   snr_db: float) -> tuple[dict[str, float], dict[str, float]]:
        """低 SNR 调整：降低声学模态、提升文本模态。"""
        snr_factor = max(self.snr_weight_floor, min(1.0, snr_db / 15.0))
        for m in ACOUSTIC_MODALITIES:
            w_s[m] *= snr_factor
            w_a[m] *= snr_factor
        w_s = self._normalize(w_s)
        w_a = self._normalize(w_a)
        return w_s, w_a

    def _apply_asr(self, w_s: dict[str, float], w_a: dict[str, float],
                   asr_confidence: float) -> tuple[dict[str, float], dict[str, float]]:
        """低 ASR 置信度调整：降低文本模态。"""
        if asr_confidence >= self.asr_confidence_threshold:
            return w_s, w_a
        asr_factor = max(0.0, min(1.0, asr_confidence / self.asr_confidence_threshold))
        for m in TEXT_MODALITIES:
            w_s[m] *= asr_factor
            w_a[m] *= asr_factor
        w_s = self._normalize(w_s)
        w_a = self._normalize(w_a)
        return w_s, w_a

    def _apply_paralang_boost(self, w_s: dict[str, float], w_a: dict[str, float],
                              paralang_events: list[dict[str, Any]]) -> None:
        """强副语言事件加权：副语言模态 ×1.5。"""
        if any(ev.get("confidence", 0.0) > self.paralang_boost_threshold
               for ev in paralang_events):
            w_s["paralang"] *= 1.5
            w_a["paralang"] *= 1.5

    def _extreme_fallback(self, w_s: dict[str, float], w_a: dict[str, float],
                          snr_db: float, asr_confidence: float) -> bool:
        """极端情况：SNR<5dB 且 ASR<0.3 时所有模态平均分配。"""
        if snr_db < 5.0 and asr_confidence < 0.3:
            n = len(MODALITIES)
            for m in MODALITIES:
                w_s[m] = 1.0 / n
                w_a[m] = 1.0 / n
            return True
        return False

    def _compute_weights(
        self,
        audio_quality: dict[str, Any],
        asr_confidence: float,
        paralang_events: list[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """计算最终动态权重（已归一化，和为 1）。"""
        w_s = dict(self.weights_s)
        w_a = dict(self.weights_a)

        snr_db = float(audio_quality.get("snr_db", 15.0))

        if self._extreme_fallback(w_s, w_a, snr_db, asr_confidence):
            return w_s, w_a

        self._apply_snr(w_s, w_a, snr_db)
        self._apply_asr(w_s, w_a, asr_confidence)
        self._apply_paralang_boost(w_s, w_a, paralang_events)

        w_s = self._normalize(w_s)
        w_a = self._normalize(w_a)
        return w_s, w_a

    # ------------------------------------------------------------------ #
    # 主融合入口
    # ------------------------------------------------------------------ #
    def fuse(
        self,
        modality_scores: dict[str, tuple[float, float]],
        audio_quality: dict[str, Any] | None = None,
        asr_confidence: float = 0.8,
        paralang_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """执行加权融合。

        Args:
            modality_scores: {模态名: (负面分 s, 唤醒分 a)}，各值应为 [0,1]。
            audio_quality: 含 ``snr_db`` 的字典（缺失时按 15dB 中等质量处理）。
            asr_confidence: ASR 置信度 [0,1]，默认 0.8（高）。
            paralang_events: 检测到的副语言事件列表，每项含 ``confidence``。

        Returns:
            ``{negative, valence, arousal, dominant_quadrant, memberships,
               modal_scores}``
        """
        audio_quality = audio_quality or {"snr_db": 15.0}
        paralang_events = paralang_events or []

        w_s, w_a = self._compute_weights(audio_quality, asr_confidence, paralang_events)

        negative = 0.0
        arousal = 0.0
        modal_scores: dict[str, dict[str, float]] = {}
        for m in MODALITIES:
            s, a = modality_scores.get(m, (0.5, 0.5))
            s = max(0.0, min(1.0, float(s)))
            a = max(0.0, min(1.0, float(a)))
            negative += w_s[m] * s
            arousal += w_a[m] * a
            modal_scores[m] = {"negative": s, "arousal": a}

        negative = max(0.0, min(1.0, negative))
        valence = 1.0 - negative
        arousal = max(0.0, min(1.0, arousal))

        memberships = compute_quadrant_memberships(
            valence, arousal, self.mid_v, self.mid_a, self.band
        )

        return {
            "negative": negative,
            "valence": valence,
            "arousal": arousal,
            "dominant_quadrant": dominant_quadrant(memberships),
            "memberships": memberships,
            "modal_scores": modal_scores,
            "weights": {"negative": w_s, "arousal": w_a},
        }
