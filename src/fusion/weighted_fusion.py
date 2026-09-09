"""多特征加权融合模块。

【核心思想】单一模态的情感识别鲁棒性有限（声学受噪声干扰、文本受 ASR 错误影响）。
本项目采用「声学 + 文本」双通道、6 模态加权融合，借鉴多模态情感计算的可靠性
原则：当某一通道信号质量下降时，自动把信任度转移到更可靠的通道，使整体估计
对噪声/口音/ASR 错误更鲁棒。

【模态】（默认权重见 config/settings.json，权重和=1）：
    acoustic  : emotion2vec 声学情感    (s=0.30, a=0.35) ← 主干，捕捉语调音色
    prosody   : parselmouth 韵律        (s=0.15, a=0.25) ← F0/节奏/HNR
    paralang  : PANNs 副语言事件        (s=0.10, a=0.15) ← 笑/哭/尖叫等
    physical  : librosa 物理声学        (s=0.05, a=0.10) ← 响度/频谱/粗糙度
    text_llm  : Qwen3 文本语义          (s=0.30, a=0.10) ← 语义层负面情绪
    text_stat : jieba 文本统计          (s=0.10, a=0.05) ← 词法层情感极性
说明：负面分(negative)中文本权重较高（语义直接表达负面），唤醒分(arousal)中
声学权重较高（唤醒主要由声学能量/节奏体现）。

【动态权重调整】（依据信号质量自适应，是鲁棒性的关键）：
    1. 低 SNR：声学模态（acoustic/prosody/paralang/physical）受噪声污染不可信，
       按比例衰减权重并转移给文本模态，再归一化保持权重和=1。
       —— 对称地，低 ASR 置信度时反向：文本不可信，权重转回声学。
    2. 极端情况（SNR<5dB 且 ASR<0.3）：两通道都不可信，所有模态平均分配(各1/6)，
       避免任一方噪声主导。
    3. 强副语言事件（如尖叫 confidence>0.8）：副语言是强情感信号，权重×1.5放大。
每次调整后重新归一化，确保权重和恒为 1。

【中性校准】（v0.2）各模态的绝对分在情绪中性语音上并不落在 0.5：v0.1 在 AISHELL-3
上实测 text_llm 负面 0.66、physical 负面 0.60 / 唤醒 0.34、text_stat 唤醒 0.17，导致
62% 的中性语音被判入 Q3。融合前按 ``config/modality_calibration.json`` 给每个模态
加一个常数偏移（offset = 0.5 − 中性均值，由 ``scripts/evaluate.py neutral`` 在
校准集上实测），这是情感计算中常规的「基线归一化」。可在 settings.json 的
``fusion_calibration.enabled`` 关闭；结果同时返回校准前后的模态分。

【输出】加权求和得 Negative/Arousal，Valence = 1 - Negative（负向效价度量），
再由 quadrant.py 做软象限判定。
"""

from __future__ import annotations

import json
from pathlib import Path
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
        self.offsets: dict[str, tuple[float, float]] = load_modality_calibration(config)

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
        skip_calibration: set[str] | frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """执行加权融合。

        Args:
            modality_scores: {模态名: (负面分 s, 唤醒分 a)}，各值应为 [0,1]。
            audio_quality: 含 ``snr_db`` 的字典（缺失时按 15dB 中等质量处理）。
            asr_confidence: ASR 置信度 [0,1]，默认 0.8（高）。
            paralang_events: 检测到的副语言事件列表，每项含 ``confidence``。
            skip_calibration: 不施加中性校准偏移的模态（如已降级为中性分的模态，
                其 0.5 是「无信息」而非测量值，不应再被平移）。

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
        modal_scores_raw: dict[str, dict[str, float]] = {}
        for m in MODALITIES:
            s0, a0 = modality_scores.get(m, (0.5, 0.5))
            s0 = max(0.0, min(1.0, float(s0)))
            a0 = max(0.0, min(1.0, float(a0)))
            ds, da = (0.0, 0.0) if m in skip_calibration else self.offsets.get(m, (0.0, 0.0))
            s = max(0.0, min(1.0, s0 + ds))
            a = max(0.0, min(1.0, a0 + da))
            negative += w_s[m] * s
            arousal += w_a[m] * a
            modal_scores_raw[m] = {"negative": s0, "arousal": a0}
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
            "modal_scores_raw": modal_scores_raw,
            "weights": {"negative": w_s, "arousal": w_a},
        }


def load_modality_calibration(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """读取中性校准偏移 ``{模态: (negative_offset, arousal_offset)}``。

    settings.json 的 ``fusion_calibration``：``{"enabled": bool, "path": str}``；
    缺省 enabled=True、path=config/modality_calibration.json。文件缺失 / 损坏 /
    禁用时返回空 dict（不做偏移）。偏移被钳制到 ±0.3，防止校准文件异常时反转判断。
    """
    fc = config.get("fusion_calibration", {}) or {}
    if not fc.get("enabled", True):
        return {}
    from src import portable
    path = Path(fc.get("path") or (portable.CONFIG_DIR / "modality_calibration.json"))
    if not path.is_absolute():
        path = portable.PROJECT_ROOT / path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for m in MODALITIES:
        v = (data.get("offsets") or {}).get(m)
        if not v:
            continue
        ds = max(-0.3, min(0.3, float(v.get("negative", 0.0))))
        da = max(-0.3, min(0.3, float(v.get("arousal", 0.0))))
        out[m] = (ds, da)
    return out
