"""GUI 工作线程（QThread + Signal）。

分析与刺激生成在工作线程执行以保持界面响应。模型加载**不在**此处：
CUDA 上下文跨线程会触发段错误，故由 MainWindow 在主线程分阶段加载。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.models.model_manager import ModelManager  # noqa: F401  (类型注解)
from src.pipeline import AnalysisPipeline
from src.stimulus.generator import StimulusGenerator

logger = logging.getLogger("mandarin_emo_stim.threads")


class AnalysisWorker(QThread):
    """端到端分析工作线程。"""
    progress = Signal(str, int)
    finished_ok = Signal(dict)   # 分析结果
    failed = Signal(str)
    interrupted = Signal()

    def __init__(self, manager: ModelManager, audio_path: str, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.audio_path = audio_path

    def run(self) -> None:
        def _cb(stage: str, pct: int) -> None:
            if self.isInterruptionRequested():
                raise InterruptedError()
            self.progress.emit(stage, pct)

        try:
            pipeline = AnalysisPipeline(self.manager)
            result = pipeline.analyze(self.audio_path, progress_cb=_cb)
            self.finished_ok.emit(result)
        except InterruptedError:
            logger.info("分析被用户中断")
            self.interrupted.emit()
        except Exception as e:  # noqa: BLE001
            logger.exception("分析失败")
            self.failed.emit(str(e))


class StimulusWorker(QThread):
    """声刺激生成工作线程。"""
    finished_ok = Signal(object, object)  # (waveform np.ndarray, params)
    failed = Signal(str)

    def __init__(self, result: dict, duration: float | None = None, parent=None):
        super().__init__(parent)
        self.result = result
        self.duration = duration

    def run(self) -> None:
        try:
            gen = StimulusGenerator()
            wave = gen.generate(
                self.result["valence"], self.result["arousal"],
                memberships=self.result["memberships"],
                duration=self.duration,
            )
            params = gen.params_for(
                self.result["valence"], self.result["arousal"],
                memberships=self.result["memberships"],
            )
            self.finished_ok.emit(wave, params)
        except Exception as e:  # noqa: BLE001
            logger.exception("刺激生成失败")
            self.failed.emit(str(e))
