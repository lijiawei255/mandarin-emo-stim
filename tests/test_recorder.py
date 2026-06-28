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


def test_max_duration_from_config():
    """max_duration 取自配置（默认 60）。"""
    rec = AudioRecorder()
    assert rec.max_duration == 60.0
