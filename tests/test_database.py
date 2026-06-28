"""数据存储测试。"""

import csv
import json

import numpy as np
import pytest

from src.storage.database import HistoryDB
from src.storage.history import HistoryManager, HistoryFull
from src.storage import exporter


# ---------------- database ----------------
@pytest.fixture
def db(tmp_path):
    return HistoryDB(db_path=tmp_path / "test.db", max_records=5)


def _sample_record(i: int = 0) -> dict:
    return {
        "source": "upload",
        "audio_path": f"/tmp/audio_{i}.wav",
        "stimulus_path": f"/tmp/stim_{i}.wav",
        "duration": 3.5,
        "negative": 0.6 + i * 0.01,
        "valence": 0.4,
        "arousal": 0.7,
        "quadrant": "Q2",
        "asr_text": "测试文本",
        "asr_confidence": 0.9,
        "snr_db": 18.0,
        "modal_scores": {"acoustic": {"negative": 0.6, "arousal": 0.7}},
        "memberships": {"Q1": 0.1, "Q2": 0.8, "Q3": 0.05, "Q4": 0.05},
        "paralang_events": [],
        "stimulus_params": {"f0": 300, "pr": 0.5},
    }


def test_add_and_get_record(db):
    rid = db.add_record(_sample_record())
    assert rid > 0
    rec = db.get_record(rid)
    assert rec is not None
    assert rec["asr_text"] == "测试文本"
    assert rec["quadrant"] == "Q2"
    # JSON 字段被反序列化为 dict
    assert isinstance(rec["modal_scores"], dict)
    assert rec["modal_scores"]["acoustic"]["negative"] == 0.6


def test_get_count(db):
    assert db.get_count() == 0
    db.add_record(_sample_record())
    db.add_record(_sample_record(1))
    assert db.get_count() == 2


def test_get_recent_order(db):
    for i in range(3):
        db.add_record(_sample_record(i))
    recent = db.get_recent(limit=2)
    assert len(recent) == 2
    # 最近添加的在前
    assert recent[0]["id"] > recent[1]["id"]


def test_delete_record(db):
    rid = db.add_record(_sample_record())
    assert db.delete_record(rid) is True
    assert db.get_record(rid) is None
    assert db.delete_record(999) is False


def test_max_records_blocks_add(db):
    """达到 max_records(5) 后 add_record 返回 -1。"""
    for i in range(5):
        assert db.add_record(_sample_record(i)) > 0
    assert db.get_count() == 5
    assert db.add_record(_sample_record(99)) == -1
    assert db.get_count() == 5  # 未新增


def test_clear_all(db):
    for i in range(3):
        db.add_record(_sample_record(i))
    deleted = db.clear_all()
    assert deleted == 3
    assert db.get_count() == 0


def test_json_fields_roundtrip(db):
    """嵌套 dict 字段序列化/反序列化往返一致。"""
    rec = _sample_record()
    rid = db.add_record(rec)
    got = db.get_record(rid)
    assert got["memberships"] == rec["memberships"]
    assert got["stimulus_params"] == rec["stimulus_params"]


# ---------------- exporter ----------------
def test_export_json(db, tmp_path):
    for i in range(3):
        db.add_record(_sample_record(i))
    out = tmp_path / "exp.json"
    exporter.export(db.get_all(), out, fmt="json")
    with out.open(encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[0]["asr_text"] == "测试文本"


def test_export_csv_utf8_sig(db, tmp_path):
    """CSV 为 utf-8-sig（含 BOM），Excel 友好，中文可读。"""
    db.add_record(_sample_record())
    out = tmp_path / "exp.csv"
    exporter.export(db.get_all(), out, fmt="csv")
    raw = out.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
    with out.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert "测试文本" in rows[1]
    assert rows[0][0] == "id"


def test_export_invalid_format(db, tmp_path):
    db.add_record(_sample_record())
    with pytest.raises(ValueError):
        exporter.export(db.get_all(), tmp_path / "x.txt", fmt="xml")


# ---------------- history manager ----------------
def test_history_manager_save_audio(tmp_path):
    db = HistoryDB(db_path=tmp_path / "h.db", max_records=10)
    mgr = HistoryManager(db=db)
    audio = np.zeros(16000, dtype=np.float32)
    path = mgr.save_audio(audio, 16000, kind="rec")
    assert path.exists()
    assert path.name.startswith("rec_")
    path_stim = mgr.save_audio(np.zeros((16000, 2), dtype=np.float32), 44100, kind="stim")
    assert path_stim.exists()
    assert path_stim.name.startswith("stim_")


def test_history_manager_full_raises(tmp_path):
    db = HistoryDB(db_path=tmp_path / "h.db", max_records=2)
    mgr = HistoryManager(db=db)
    mgr.add(_sample_record())
    mgr.add(_sample_record(1))
    assert mgr.is_full
    with pytest.raises(HistoryFull):
        mgr.add(_sample_record(2))


def test_history_manager_export_and_clear(tmp_path):
    db = HistoryDB(db_path=tmp_path / "h.db", max_records=10)
    mgr = HistoryManager(db=db)
    mgr.add(_sample_record())
    mgr.add(_sample_record(1))
    out = tmp_path / "out.json"
    mgr.export(out, fmt="json")
    assert out.exists()
    n = mgr.clear()
    assert n == 2
    assert mgr.remaining() == 10
