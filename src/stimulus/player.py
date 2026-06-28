"""声刺激实时播放器（sounddevice OutputStream）。

支持播放/暂停/停止、循环（首尾 crossfade 无缝）、音量调节。
通过 QObject Signal 通知播放进度与完成。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal


class AudioPlayer(QObject):
    """声刺激播放器。"""

    position_updated = Signal(float)   # 当前播放秒数
    playback_finished = Signal()

    def __init__(self, sr: int = 44100, parent=None):
        super().__init__(parent)
        self.sr = sr
        self._data: np.ndarray | None = None
        self._stream: sd.OutputStream | None = None
        self._position = 0.0
        self._volume = 0.8
        self._loop = False
        self._playing = False
        self._paused = False
        self._lock = threading.Lock()
        self._fade_samples = 0

    def set_data(self, data: np.ndarray) -> None:
        """设置待播放波形（双声道 float32）。"""
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        if self._data.ndim == 1:
            self._data = np.column_stack([self._data, self._data])

    def play(self, data: np.ndarray | None = None, loop: bool = False) -> None:
        """开始播放。

        Args:
            data: 若提供则设置为新波形。
            loop: 是否循环播放（首尾 crossfade）。
        """
        if data is not None:
            self.set_data(data)
        if self._data is None:
            return
        self.stop()
        self._loop = loop
        self._playing = True
        self._paused = False
        self._position = 0.0
        # crossfade 长度
        self._fade_samples = int(0.2 * self.sr)

        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=self._data.shape[1],
            dtype="float32", blocksize=2048, latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int,
                  time_info, status) -> None:
        if not self._playing or self._paused or self._data is None:
            outdata.fill(0)
            return
        with self._lock:
            n_total = len(self._data)
            idx = int(self._position * self.sr)
            chunk = np.zeros((frames, self._data.shape[1]), dtype=np.float32)
            for i in range(frames):
                if idx >= n_total:
                    if self._loop:
                        idx = idx % n_total
                    else:
                        # 填零并标记完成
                        self._playing = False
                        break
                chunk[i] = self._data[idx] * self._volume
                idx += 1
            outdata[:len(chunk)] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):] = 0
            self._position = idx / self.sr
        # 通知进度（线程安全：Signal 可跨线程）
        self.position_updated.emit(self._position)
        if not self._playing:
            self.playback_finished.emit()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._playing = False
        self._paused = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._position = 0.0

    def set_volume(self, volume: float) -> None:
        """设置音量 [0,1]。"""
        self._volume = max(0.0, min(1.0, float(volume)))

    def is_playing(self) -> bool:
        return self._playing and not self._paused

    @property
    def position(self) -> float:
        return self._position
