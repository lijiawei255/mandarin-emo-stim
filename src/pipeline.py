"""端到端分析管线编排器。

把音频 → VAD/ASR → 6 模态分数提取 → 加权融合 → 象限判定 → 标准化输出
串联为一条无头可调用的管线，供 GUI 工作线程与 CLI 共用。

6 模态：
    acoustic  (emotion2vec)   prosody   (parselmouth)
    paralang  (PANNs CNN10)   physical  (librosa)
    text_llm  (Qwen3)         text_stat (jieba)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src import portable  # noqa: F401  便携环境副作用
from src.audio import loader, vad
from src.config_loader import load_settings
from src.features import physical as physical_feat
from src.features import prosody as prosody_feat
from src.features import text_stats
from src.fusion.weighted_fusion import WeightedFusion

logger = logging.getLogger("mandarin_emo_stim.pipeline")

ProgressCallback = Callable[[str, int], None]


class AnalysisPipeline:
    """端到端情感分析管线。"""

    def __init__(self, model_manager, config: dict | None = None):
        """
        Args:
            model_manager: 已加载 4 模型的 :class:`ModelManager`。
            config: settings.json；为 ``None`` 时自动加载。
        """
        self.manager = model_manager
        self.config = config if config is not None else load_settings()
        self.fusion = WeightedFusion(self.config)

    def analyze(self, audio_path: str | Path,
                progress_cb: ProgressCallback | None = None) -> dict[str, Any]:
        """分析音频文件，输出标准化情绪指标。

        Args:
            audio_path: 音频文件路径（WAV/MP3/FLAC）。
            progress_cb: 进度回调 ``(stage_name, percent)``。

        Returns:
            完整结果字典（含 negative/valence/arousal/quadrant/memberships/
            modal_scores/asr_text/audio_quality/paralang_events/duration/
            degraded_modalities）。``degraded_modalities`` 列出因异常而以
            中性分参与融合的模态；仅「加载音频」与「融合」失败会抛异常。
        """
        stages = [
            ("加载音频", self._step_load, None),
            ("ASR 转写", self._step_asr, ("text_llm", "text_stat")),
            ("声学情感", self._step_emotion, ("acoustic",)),
            ("副语言事件", self._step_panns, ("paralang",)),
            ("韵律特征", self._step_prosody, ("prosody",)),
            ("物理声学", self._step_physical, ("physical",)),
            ("文本统计", self._step_text_stat, ("text_stat",)),
            ("文本语义", self._step_text_llm, ("text_llm",)),
            ("融合", self._step_fuse, None),
        ]
        ctx: dict[str, Any] = {"audio_path": str(audio_path), "degraded": []}
        n = len(stages)
        for i, (name, fn, modalities) in enumerate(stages):
            if progress_cb:
                progress_cb(name, int(i / n * 100))
            try:
                fn(ctx)
            except Exception as e:  # noqa: BLE001
                # 「加载音频」与「融合」失败没有可降级的余地，向上抛；
                # 其余任一模态失败 → 记录、写入中性分、继续。
                if modalities is None:
                    logger.error("管线步骤「%s」失败: %s", name, e)
                    raise
                logger.warning("管线步骤「%s」失败，该模态降级为中性分: %s", name, e)
                self._degrade(ctx, modalities)
        if progress_cb:
            progress_cb("完成", 100)
        return self._finalize(ctx)

    @staticmethod
    def _degrade(ctx: dict, modalities: tuple[str, ...]) -> None:
        """把指定模态置为中性分 (0.5, 0.5) 并记入降级列表。"""
        for m in modalities:
            ctx[f"s_{m}"] = 0.5
            ctx[f"a_{m}"] = 0.5
            if m not in ctx["degraded"]:
                ctx["degraded"].append(m)
        if "text_llm" in modalities and "asr_text" not in ctx:
            # ASR 本身失败：无文本、置信度 0
            ctx["asr_text"] = ""
            ctx["asr_confidence"] = 0.0
            ctx["asr"] = {"text": "", "confidence": 0.0, "timestamp": []}

    # ------------------------------------------------------------------ #
    def _step_load(self, ctx: dict) -> None:
        target_sr = int(self.config["audio"]["sample_rate"])
        y, sr = loader.load_audio(ctx["audio_path"], target_sr=target_sr)
        ctx["y"], ctx["sr"] = y, sr
        ctx["duration"] = loader.get_duration(y, sr)

    def _step_asr(self, ctx: dict) -> None:
        # Paraformer 接受文件路径
        asr = self.manager.get_asr_model()
        result = asr.transcribe(ctx["audio_path"])
        ctx["asr"] = result
        ctx["asr_text"] = result["text"]
        ctx["asr_confidence"] = result["confidence"]
        ctx["asr_confidence_source"] = result.get("confidence_source", "proxy")
        # VAD
        vad_info = vad.vad_from_asr_result(ctx["y"], ctx["sr"], result)
        ctx["vad"] = vad_info
        ctx["effective_duration"] = vad_info["effective_duration"] or ctx["duration"]

    def _step_emotion(self, ctx: dict) -> None:
        emo = self.manager.get_emotion_model()
        result = emo.predict(ctx["audio_path"])
        ctx["emotion"] = result
        ctx["s_acoustic"] = result["s_acoustic"]
        ctx["a_acoustic"] = result["a_acoustic"]

    def _step_panns(self, ctx: dict) -> None:
        pann = self.manager.get_pann_model()
        # PANNs 期望 32000Hz
        import librosa
        y32 = librosa.resample(ctx["y"].astype("f"), orig_sr=ctx["sr"], target_sr=32000)
        result = pann.detect(y32, 32000)
        ctx["paralang"] = result
        ctx["paralang_events"] = result["events"]
        ctx["s_paralang"] = result["s_paralang"]
        ctx["a_paralang"] = result["a_paralang"]
        if result.get("degraded") and "paralang" not in ctx["degraded"]:
            ctx["degraded"].append("paralang")

    def _step_prosody(self, ctx: dict) -> None:
        # 语速优先由 ASR 字级时间戳给出（ASR 降级时为 None → 能量法回退）
        rate = vad.syllable_rate(ctx["asr"]) if ctx.get("asr") else None
        feat = prosody_feat.extract(ctx["y"], ctx["sr"], syllable_rate=rate)
        s, a, detail = prosody_feat.score(feat)
        ctx["prosody"] = detail
        ctx["s_prosody"], ctx["a_prosody"] = s, a

    def _step_physical(self, ctx: dict) -> None:
        feat = physical_feat.extract(ctx["y"], ctx["sr"])
        s, a, detail = physical_feat.score(feat)
        ctx["physical"] = detail
        ctx["s_physical"], ctx["a_physical"] = s, a
        # SNR 用于动态权重与 UI 警告
        ctx["snr_db"] = feat.snr_db

    def _step_text_stat(self, ctx: dict) -> None:
        result = text_stats.analyze(ctx.get("asr_text", ""))
        ctx["text_stat"] = result.detail
        ctx["s_text_stat"], ctx["a_text_stat"] = result.s, result.a

    def _step_text_llm(self, ctx: dict) -> None:
        llm = self.manager.get_llm_model()
        result = llm.analyze_text(ctx.get("asr_text", ""))
        ctx["text_llm"] = result
        # 降级：解析失败时用 text_stat 的负面分，唤醒用 0.5
        if result["fallback"]:
            ctx["s_text_llm"] = ctx.get("s_text_stat", 0.5)
            ctx["a_text_llm"] = 0.5
        else:
            ctx["s_text_llm"] = result["s_text_llm"]
            ctx["a_text_llm"] = result["a_text_llm"]

    def _step_fuse(self, ctx: dict) -> None:
        modality_scores = {
            "acoustic": (ctx["s_acoustic"], ctx["a_acoustic"]),
            "prosody": (ctx["s_prosody"], ctx["a_prosody"]),
            "paralang": (ctx["s_paralang"], ctx["a_paralang"]),
            "physical": (ctx["s_physical"], ctx["a_physical"]),
            "text_llm": (ctx["s_text_llm"], ctx["a_text_llm"]),
            "text_stat": (ctx["s_text_stat"], ctx["a_text_stat"]),
        }
        audio_quality = {"snr_db": ctx.get("snr_db", 15.0)}
        ctx["fusion_result"] = self.fusion.fuse(
            modality_scores,
            audio_quality=audio_quality,
            asr_confidence=ctx.get("asr_confidence", 0.8),
            paralang_events=ctx.get("paralang_events", []),
            skip_calibration=set(ctx.get("degraded", [])),
        )

    # ------------------------------------------------------------------ #
    def _finalize(self, ctx: dict) -> dict[str, Any]:
        from src.fusion.reliability import grade as _reliability
        fr = ctx["fusion_result"]
        return {
            "negative": fr["negative"],
            "valence": fr["valence"],
            "arousal": fr["arousal"],
            "dominant_quadrant": fr["dominant_quadrant"],
            "memberships": fr["memberships"],
            "modal_scores": fr["modal_scores"],
            "modal_scores_raw": fr.get("modal_scores_raw", fr["modal_scores"]),
            "weights": fr["weights"],
            "uncertainty": fr.get("uncertainty", {}),
            "reliability": _reliability(fr.get("uncertainty")),
            "fusion_mode": fr.get("fusion_mode", "weighted"),
            "calibration_source": fr.get("calibration_source", "none"),
            "calibration_profile": fr.get("calibration_profile"),
            "asr_confidence_source": ctx.get("asr_confidence_source", "proxy"),
            "asr_text": ctx.get("asr_text", ""),
            "asr_confidence": ctx.get("asr_confidence", 0.0),
            "audio_quality": {"snr_db": ctx.get("snr_db", 0.0)},
            "duration": ctx.get("effective_duration", ctx.get("duration", 0.0)),
            "paralang_events": ctx.get("paralang_events", []),
            "degraded_modalities": list(ctx.get("degraded", [])),
            "modal_details": {
                "acoustic": ctx.get("emotion", {}),
                "prosody": ctx.get("prosody", {}),
                "physical": ctx.get("physical", {}),
                "text_stat": ctx.get("text_stat", {}),
                "text_llm": ctx.get("text_llm", {}),
            },
        }
