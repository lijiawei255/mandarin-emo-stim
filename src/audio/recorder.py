"""麦克风实时录音（sounddevice InputStream）。

提供同步「录制指定时长」与异步「开始/停止」两种模式。
采样率 16000Hz、单声道、16bit PCM，chunk_size=1024（64ms）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from src.config_loader import load_settings


def _import_sd():
    """延迟导入 sounddevice。

    避免在模块顶层 import sounddevice —— 若该依赖缺失或麦克风后端异常，
    会导致整个 GUI 在 import 阶段崩溃。改为按需导入，仅在真正录音/枚举
    设备时检查，缺失时抛出可读的 ImportError 由上层友好提示。
    """
    import sounddevice as sd
    return sd


class AudioRecorder:
    """麦克风录音器（线程安全）。"""

    def __init__(self, sr: int | None = None, channels: int = 1,
                 chunk_size: int | None = None, device: int | None = None):
        audio_cfg = load_settings()["audio"]
        self.sr = sr or int(audio_cfg["sample_rate"])
        self.channels = channels
        self.chunk_size = chunk_size or int(audio_cfg["chunk_size"])
        self.device = device
        self.max_duration = float(audio_cfg["record_duration_max"])

        self._buffer: deque[np.ndarray] = deque()
        self._stream = None  # sd.InputStream，延迟创建
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._stop_time: float = 0.0   # 停止时刻，用于 stop() 后仍能取到正确时长

    @staticmethod
    def list_devices() -> list[dict]:
        """列出可用输入设备。

        在某些中文 Windows 环境下，sounddevice 解码设备名时会抛
        ``UnicodeDecodeError``，此处捕获并回退到逐设备查询（容错）。
        若 sounddevice 未安装，返回空列表（上层据此显示「无可用设备」）。
        """
        try:
            sd = _import_sd()
        except ImportError:
            return []

        result: list[dict] = []
        try:
            devs = sd.query_devices()
        except UnicodeDecodeError:
            # 回退：逐索引查询，逐个容错
            for i in range(32):  # 设备数上限
                try:
                    d = sd.query_devices(i)
                except Exception:
                    break
                if d.get("max_input_channels", 0) > 0:
                    name = d.get("name", f"device_{i}")
                    if not isinstance(name, str):
                        name = str(name)
                    result.append({"id": i, "name": name,
                                   "channels": d["max_input_channels"],
                                   "default_sr": d.get("default_samplerate")})
            return result
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", f"device_{i}")
                if not isinstance(name, str):
                    name = str(name)
                result.append({"id": i, "name": name,
                               "channels": d["max_input_channels"],
                               "default_sr": d.get("default_samplerate")})
        return result

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status) -> None:
        if status:
            pass  # 可记录溢出，但暂不中断
        if self._recording:
            with self._lock:
                self._buffer.append(indata.copy())

    def start(self) -> None:
        """开始录音（异步）。

        Raises:
            ImportError: sounddevice 未安装。
            Exception: 麦克风设备不可用或权限被拒。
        """
        if self._recording:
            return
        sd = _import_sd()  # 缺失时抛 ImportError，由上层友好提示
        self._buffer.clear()
        self._stream = sd.InputStream(
            samplerate=self.sr, channels=self.channels, dtype="float32",
            blocksize=self.chunk_size, device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._recording = True
        self._start_time = time.time()
        self._stop_time = 0.0  # 重置停止时刻

    def stop(self) -> np.ndarray:
        """停止录音并返回完整波形。

        Returns:
            单声道 float32 波形（shape=(n,)）。
        """
        if not self._recording:
            return np.array([], dtype=np.float32)
        self._recording = False
        self._stop_time = time.time()  # 记录停止时刻，供 elapsed() 取用
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            data = np.concatenate(list(self._buffer), axis=0)
        # 单声道：取第一列
        if data.ndim > 1:
            data = data[:, 0]
        return np.ascontiguousarray(data.astype(np.float32))

    def is_recording(self) -> bool:
        return self._recording

    def elapsed(self) -> float:
        """已录制时长（秒）。

        录音中返回当前已录时长；停止后返回 _start_time 到 _stop_time 的时长
        （不再依赖 _recording 标志，避免 stop() 先于 elapsed() 调用时返回 0）。
        """
        if self._start_time <= 0:
            return 0.0
        end = self._stop_time if self._stop_time > 0 else time.time()
        return max(0.0, end - self._start_time)

    def record(self, duration: float,
               progress_cb: Callable[[float], None] | None = None) -> np.ndarray:
        """同步录制指定时长。

        Args:
            duration: 录制秒数（会被限制到 max_duration）。
            progress_cb: 进度回调（传入已录制秒数）。

        Returns:
            单声道 float32 波形。
        """
        duration = min(duration, self.max_duration)
        self.start()
        try:
            t0 = time.time()
            while time.time() - t0 < duration:
                if progress_cb:
                    progress_cb(time.time() - t0)
                time.sleep(0.05)
        finally:
            # finally 仅保证停止录制状态；不在 finally 中 return（避免吞掉
            # try 体内的异常），在正常路径返回结果
            self._recording = False
        return self.stop()
