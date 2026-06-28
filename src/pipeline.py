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
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src import portable
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
            modal_scores/asr_text/audio_quality/paralang_events/duration）。
        """
        stages = [
            ("加载音频", self._step_load),
            ("ASR 转写", self._step_asr),
            ("声学情感", self._step_emotion),
            ("副语言事件", self._step_panns),
            ("韵律特征", self._step_prosody),
            ("物理声学", self._step_physical),
            ("文本统计", self._step_text_stat),
            ("文本语义", self._step_text_llm),
            ("融合", self._step_fuse),
        ]
        ctx: dict[str, Any] = {"audio_path": str(audio_path)}
        n = len(stages)
        for i, stage_def in enumerate(stages):
            name = stage_def[0]
            fn = stage_def[1]
            if progress_cb:
                progress_cb(name, int(i / n * 100))
            try:
                fn(ctx)
            except Exception as e:  # noqa: BLE001
                logger.error("管线步骤「%s」失败: %s", name, e)
                # 关键步骤失败则向上抛
                if name in ("加载音频", "ASR 转写"):
                    raise
        if progress_cb:
            progress_cb("完成", 100)
        return self._finalize(ctx)

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

    def _step_prosody(self, ctx: dict) -> None:
        feat = prosody_feat.extract(ctx["y"], ctx["sr"])
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
            ctx["s_text_llm"] = ctx["s_text_stat"]
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
        )

    # ------------------------------------------------------------------ #
    def _finalize(self, ctx: dict) -> dict[str, Any]:
        fr = ctx["fusion_result"]
        return {
            "negative": fr["negative"],
            "valence": fr["valence"],
            "arousal": fr["arousal"],
            "dominant_quadrant": fr["dominant_quadrant"],
            "memberships": fr["memberships"],
            "modal_scores": fr["modal_scores"],
            "weights": fr["weights"],
            "asr_text": ctx.get("asr_text", ""),
            "asr_confidence": ctx.get("asr_confidence", 0.0),
            "audio_quality": {"snr_db": ctx.get("snr_db", 0.0)},
            "duration": ctx.get("effective_duration", ctx.get("duration", 0.0)),
            "paralang_events": ctx.get("paralang_events", []),
            "modal_details": {
                "acoustic": ctx.get("emotion", {}),
                "prosody": ctx.get("prosody", {}),
                "physical": ctx.get("physical", {}),
                "text_stat": ctx.get("text_stat", {}),
                "text_llm": ctx.get("text_llm", {}),
            },
        }
