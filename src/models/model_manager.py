"""模型管理器：统一加载/卸载/设备管理/显存监控。

按配置的 ``device`` 策略（auto/cuda/cpu）选择推理设备，加载 4 个模型，
并提供显存占用查询与按需卸载（OOM 降级）。
"""

from __future__ import annotations

import logging
from typing import Callable

from src import portable
from src.config_loader import load_settings

logger = logging.getLogger("mandarin_emo_stim.model_manager")

ProgressCallback = Callable[[str, int], None]


class ModelManager:
    """4 模型统一管理器。"""

    def __init__(self, config: dict | None = None, portable_root=None):
        # portable_root 仅用于兼容文档接口；实际路径由 src.portable 决定
        self.config = config if config is not None else load_settings()
        self.device: str = ""
        self._asr = None
        self._emotion = None
        self._pann = None
        self._llm = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    def resolve_device(self) -> str:
        """根据配置与硬件选择推理设备。"""
        requested = self.config["models"].get("device", "auto")
        if requested == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = requested
        logger.info("推理设备: %s", self.device)
        return self.device

    def load_all(self, progress_cb: ProgressCallback | None = None) -> None:
        """加载全部 4 个模型。"""
        self.resolve_device()
        stages = [
            ("ASR (Paraformer)", self._load_asr),
            ("emotion2vec", self._load_emotion),
            ("PANNs (CNN10)", self._load_pann),
            ("LLM (Qwen3)", self._load_llm),
        ]
        n = len(stages)
        for i, (name, loader) in enumerate(stages):
            if progress_cb:
                progress_cb(name, int(i / n * 100))
            try:
                loader()
            except Exception as e:  # noqa: BLE001
                logger.error("加载 %s 失败: %s", name, e)
                raise
        self._loaded = True
        if progress_cb:
            progress_cb("全部就绪", 100)

    def _load_asr(self):
        from src.models.asr_model import ASRModel
        device = self.config["models"].get("asr_device", self.device)
        self._asr = ASRModel(device=device)

    def _load_emotion(self):
        from src.models.emotion_model import EmotionModel
        self._emotion = EmotionModel(device=self.device)

    def _load_pann(self):
        from src.models.pann_model import PANNModel
        self._pann = PANNModel(device=self.device)

    def _load_llm(self):
        from src.models.llm_model import LLMModel
        self._llm = LLMModel(device=self.device)

    # ------------------------------------------------------------------ #
    @property
    def loaded_count(self) -> int:
        return sum(m is not None for m in (self._asr, self._emotion, self._pann, self._llm))

    def get_asr_model(self):
        return self._asr

    def get_emotion_model(self):
        return self._emotion

    def get_pann_model(self):
        return self._pann

    def get_llm_model(self):
        return self._llm

    def get_device(self) -> str:
        return self.device

    def vram_usage_mb(self) -> dict[str, float]:
        """返回当前 GPU 显存占用（MB）。仅 CUDA 有效。"""
        if self.device != "cuda":
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
        try:
            import torch
            return {
                "allocated": torch.cuda.memory_allocated() / 1024 ** 2,
                "reserved": torch.cuda.memory_reserved() / 1024 ** 2,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
            }
        except Exception:
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

    def unload_all(self) -> None:
        """卸载全部模型，释放显存。"""
        import gc
        self._asr = self._emotion = self._pann = self._llm = None
        self._loaded = False
        gc.collect()
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("全部模型已卸载")

    def fallback_to_cpu(self) -> None:
        """OOM 降级：卸载后切到 CPU 重新加载。"""
        logger.warning("触发 OOM 降级，切换到 CPU 模式")
        self.unload_all()
        self.config["models"]["device"] = "cpu"
        self.config["models"]["asr_device"] = "cpu"
        self.load_all()
