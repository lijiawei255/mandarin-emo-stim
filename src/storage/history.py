"""历史记录高级管理。

封装 :class:`~src.storage.database.HistoryDB` 与导出工具，提供：
    - 新增记录（含 200 条上限检测）；
    - 录音/刺激音频文件落盘（命名 ``rec_/stim_时间戳.wav``）；
    - 满额时引导导出（返回 ``HistoryFull`` 信号）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src import portable
from src.config_loader import load_settings
from src.storage import exporter
from src.storage.database import HistoryDB


@dataclass
class HistoryFull(Exception):
    """历史记录已满，需导出后清空。"""
    max_records: int

    def __str__(self) -> str:  # pragma: no cover - 简单格式化
        return f"历史记录已达上限（{self.max_records} 条），请先导出并清空"


class HistoryManager:
    """历史记录管理器。"""

    def __init__(self, db: HistoryDB | None = None):
        settings = load_settings()
        max_records = int(settings["history"]["max_records"])
        self.db = db if db is not None else HistoryDB(max_records=max_records)
        self.max_records = self.db.max_records

    @property
    def is_full(self) -> bool:
        return self.db.get_count() >= self.max_records

    def remaining(self) -> int:
        return max(0, self.max_records - self.db.get_count())

    # ------------------------------------------------------------------ #
    def save_audio(self, audio: np.ndarray, sr: int, kind: str = "rec") -> Path:
        """把录音/刺激波形落盘到 ``portable_data/history/audio|stimuli/``。

        Args:
            audio: 波形数据（单声道或双声道）。
            sr: 采样率。
            kind: ``"rec"``（用户录音）或 ``"stim"``（生成的刺激）。

        Returns:
            文件路径。
        """
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if kind == "stim":
            out_dir = portable.HISTORY_STIMULI_DIR
            name = f"stim_{ts}.wav"
        else:
            out_dir = portable.HISTORY_AUDIO_DIR
            name = f"rec_{ts}.wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path

    def add(self, record: dict[str, Any]) -> int:
        """新增记录；满额时抛出 :class:`HistoryFull` 引导导出。

        Returns:
            新记录 id。
        """
        rid = self.db.add_record(record)
        if rid == -1:
            raise HistoryFull(max_records=self.max_records)
        return rid

    def export(self, path: Path, fmt: str = "json") -> Path:
        """导出全部记录为 JSON/CSV。"""
        records = self.db.get_all()
        return exporter.export(records, path, fmt=fmt)

    def clear(self) -> int:
        """清空全部记录，返回删除条数。"""
        return self.db.clear_all()
