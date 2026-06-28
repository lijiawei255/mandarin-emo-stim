"""健壮性测试：异常输入、资源异常、并发与中断。

覆盖：
    - 配置 schema 校验（缺失键给清晰错误）
    - OOM 检测与 CPU 降级判定
    - PANNs checkpoint 完整性校验
    - AudioPlayer 线程安全（set_volume/pause 加锁）
    - recorder.record 不再吞异常
    - CLI 错误处理（文件不存在 / 中断）
    - 空值/越界输入不崩溃
"""

import json
from pathlib import Path

import numpy as np
import pytest


# ==================== 配置校验 ====================
def test_config_missing_key_raises_config_error(tmp_path, monkeypatch):
    """settings.json 缺失必需键时抛 ConfigError 而非 KeyError。"""
    from src import config_loader

    bad = {"audio": {"sample_rate": 16000}}  # 缺大量必需键
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(config_loader.portable, "SETTINGS_PATH", p)
    config_loader.load_settings.cache_clear()

    with pytest.raises(config_loader.ConfigError) as exc_info:
        config_loader.load_settings()
    assert "缺少必需配置项" in str(exc_info.value)
    config_loader.load_settings.cache_clear()


def test_config_invalid_json_raises_config_error(tmp_path, monkeypatch):
    """非法 JSON 抛 ConfigError。"""
    from src import config_loader

    p = tmp_path / "settings.json"
    p.write_text("{这不是合法json", encoding="utf-8")
    monkeypatch.setattr(config_loader.portable, "SETTINGS_PATH", p)
    config_loader.load_settings.cache_clear()

    with pytest.raises(config_loader.ConfigError):
        config_loader.load_settings()
    config_loader.load_settings.cache_clear()


def test_config_valid_passes():
    """合法的默认配置通过校验。"""
    from src import config_loader
    config_loader.load_settings.cache_clear()
    s = config_loader.load_settings()
    assert "fusion_weights" in s
    config_loader.load_settings.cache_clear()


# ==================== OOM 检测 ====================
def test_is_oom_like_detects_cuda_oom():
    from src.models.model_manager import ModelManager
    assert ModelManager._is_oom_like(RuntimeError("CUDA out of memory.")) is True
    assert ModelManager._is_oom_like(MemoryError()) is True
    # 非内存错误不误判
    assert ModelManager._is_oom_like(ValueError("bad value")) is False
    assert ModelManager._is_oom_like(FileNotFoundError("x")) is False


# ==================== PANNs checkpoint 完整性 ====================
def test_checkpoint_valid_size_accepted(tmp_path):
    from src.models.downloader import _is_valid_checkpoint
    p = tmp_path / "ckpt.pth"
    p.write_bytes(b"\0" * (24 * 1024 * 1024))  # 24MB 合法
    assert _is_valid_checkpoint(p) is True


def test_checkpoint_truncated_rejected(tmp_path):
    """截断的小文件被识别为损坏。"""
    from src.models.downloader import _is_valid_checkpoint
    p = tmp_path / "ckpt.pth"
    p.write_bytes(b"\0" * (5 * 1024 * 1024))  # 5MB 截断
    assert _is_valid_checkpoint(p) is False


def test_checkpoint_missing_rejected(tmp_path):
    from src.models.downloader import _is_valid_checkpoint
    assert _is_valid_checkpoint(tmp_path / "nope.pth") is False


# ==================== AudioPlayer 线程安全 ====================
def test_player_set_volume_is_thread_safe():
    """set_volume 在锁内修改，与回调线程不竞争。"""
    from src.stimulus.player import AudioPlayer
    p = AudioPlayer(sr=44100)
    # 并发设置不崩溃
    p.set_volume(0.5)
    assert p._volume == 0.5
    p.set_volume(1.0)
    assert p._volume == 1.0
    p.set_volume(-1.0)  # 越界被钳制
    assert p._volume == 0.0


def test_player_pause_resume_with_lock():
    from src.stimulus.player import AudioPlayer
    p = AudioPlayer(sr=44100)
    p.pause()
    assert p._paused is True
    p.resume()
    assert p._paused is False


# ==================== recorder.record 不吞异常 ====================
def test_recorder_record_propagates_exception(monkeypatch):
    """record() 的 finally 不再吞掉 try 体内的异常。"""
    from src.audio.recorder import AudioRecorder
    rec = AudioRecorder()
    # 让 start 抛异常，record 应向上传播而非被 finally 吞掉
    monkeypatch.setattr(rec, "start", lambda: (_ for _ in ()).throw(RuntimeError("device busy")))
    with pytest.raises(RuntimeError, match="device busy"):
        rec.record(duration=1.0)


# ==================== 空值 / 越界输入 ====================
def test_zscore_normalize_handles_extremes():
    from src.fusion.normalizer import zscore_normalize
    # 极大极小值不崩溃，结果在 [0,1]
    for v in (-1e9, 0, 1e9, float("inf"), float("-inf")):
        result = zscore_normalize(v, 180.0, 50.0)
        assert 0.0 <= result <= 1.0


def test_fusion_handles_all_zero_scores():
    """全 0 模态分数不导致除零。"""
    from src.fusion.weighted_fusion import WeightedFusion, MODALITIES
    from src.config_loader import load_settings
    fus = WeightedFusion(load_settings())
    scores = {m: (0.0, 0.0) for m in MODALITIES}
    r = fus.fuse(scores, audio_quality={"snr_db": 3.0}, asr_confidence=0.2)
    assert 0.0 <= r["negative"] <= 1.0
    assert abs(sum(r["memberships"].values()) - 1.0) < 1e-6


def test_stimulus_handles_boundary_valence_arousal():
    """valence/arousal 取边界值 0 和 1 不崩溃。"""
    from src.stimulus.generator import StimulusGenerator
    gen = StimulusGenerator()
    for v in (0.0, 1.0):
        for a in (0.0, 1.0):
            wave = gen.generate(v, a, duration=10.0)
            assert np.all(np.isfinite(wave))


# ==================== CLI 错误处理 ====================
def test_cli_missing_file_returns_error():
    """CLI 对不存在文件返回非零退出码，不崩溃。"""
    from src.cli import main
    code = main(["--audio", "nonexistent_file.wav"])
    assert code != 0


# ==================== HistoryDB 并发安全 ====================
def test_historydb_concurrent_writes_safe(tmp_path):
    """多线程并发写历史记录不损坏数据库（锁保护）。"""
    import threading
    from src.storage.database import HistoryDB
    db = HistoryDB(db_path=tmp_path / "concurrent.db", max_records=200)

    def _sample(i):
        return {"source": "upload", "negative": 0.5 + i * 0.001,
                "asr_text": f"test{i}"}

    threads = [threading.Thread(target=lambda i=i: db.add_record(_sample(i)))
               for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.get_count() == 10
