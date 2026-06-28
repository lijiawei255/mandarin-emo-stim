"""录音器单元测试（不依赖真实麦克风）。

AudioRecorder.list_devices 与构造参数校验可在无设备环境测试；
真实的 start/stop 采集由 GUI 录音测试用桩 recorder 覆盖。
"""

import pytest

from src.audio.recorder import AudioRecorder


def test_recorder_construction_defaults():
    rec = AudioRecorder()
    assert rec.sr == 16000
    assert rec.channels == 1
    assert rec.chunk_size == 1024
    assert rec.is_recording() is False
    assert rec.elapsed() == 0.0


def test_recorder_construction_custom_params():
    rec = AudioRecorder(sr=8000, channels=1, chunk_size=512, device=None)
    assert rec.sr == 8000
    assert rec.chunk_size == 512
    assert rec.device is None


def test_recorder_stop_when_not_recording_returns_empty():
    """未开始录音时 stop 返回空数组，不报错。"""
    rec = AudioRecorder()
    out = rec.stop()
    assert len(out) == 0
    assert out.dtype.name == "float32"


def test_list_devices_returns_list():
    """list_devices 在任何环境都应返回列表（可能为空）。"""
    devs = AudioRecorder.list_devices()
    assert isinstance(devs, list)
    # 每项应含 id/name 键
    for d in devs:
        assert "id" in d and "name" in d


def test_elapsed_correct_after_stop(monkeypatch):
    """回归测试：stop() 后 elapsed() 必须返回真实录制时长，而非 0。

    历史 bug：elapsed() 依赖 _recording 标志，stop() 先把它设为 False，
    随后调用 elapsed() 永远返回 0，导致 GUI 误判「录音过短」。
    """
    import time
    import numpy as np
    import src.audio.recorder as rec_mod

    class _FakeStream:
        def __init__(self, **kw):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    def _fake_import_sd():
        class _SD:
            InputStream = _FakeStream
        return _SD

    monkeypatch.setattr(rec_mod, "_import_sd", _fake_import_sd)

    rec = AudioRecorder(sr=16000)
    rec.start()
    time.sleep(0.3)
    # 注入一些模拟采集数据
    rec._buffer.append(np.zeros((4800, 1), dtype=np.float32))
    rec.stop()

    # 关键断言：停止后 elapsed() 不应为 0
    assert rec.elapsed() > 0.2, "stop() 后 elapsed() 不应返回 0（历史回归 bug）"


def test_max_duration_from_config():
    """max_duration 取自配置（默认 60）。"""
    rec = AudioRecorder()
    assert rec.max_duration == 60.0
