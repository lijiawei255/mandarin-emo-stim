"""GUI 工作线程（QThread + Signal）。

避免在主线程执行耗时的模型加载、分析与播放，保持界面响应。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.models.model_manager import ModelManager
from src.pipeline import AnalysisPipeline
from src.stimulus.generator import StimulusGenerator

logger = logging.getLogger("mandarin_emo_stim.threads")


class ModelLoadWorker(QThread):
    """模型加载工作线程。"""
    progress = Signal(str, int)   # (阶段名, 进度%)
    finished_ok = Signal(object)  # ModelManager
    failed = Signal(str)

    def __init__(self, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            mgr = ModelManager(config=self.config)
            mgr.load_all(progress_cb=lambda s, p: self.progress.emit(s, p))
            self.finished_ok.emit(mgr)
        except Exception as e:  # noqa: BLE001
            logger.exception("模型加载失败")
            self.failed.emit(str(e))


class AnalysisWorker(QThread):
    """端到端分析工作线程。"""
    progress = Signal(str, int)
    finished_ok = Signal(dict)   # 分析结果
    failed = Signal(str)

    def __init__(self, manager: ModelManager, audio_path: str, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.audio_path = audio_path

    def run(self) -> None:
        try:
            pipeline = AnalysisPipeline(self.manager)
            result = pipeline.analyze(
                self.audio_path,
                progress_cb=lambda s, p: self.progress.emit(s, p),
            )
            self.finished_ok.emit(result)
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
