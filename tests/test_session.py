"""v0.5：会话内状态平滑与滞回、会话/试次持久化与导出。"""

import csv

import pytest

from src.session.model import SessionDB
from src.session.state_tracker import StateTracker, TrackerConfig


# ---------------- StateTracker ----------------
def test_ema_smooths_and_first_sample_initializes():
    t = StateTracker(TrackerConfig(ema_alpha=0.5))
    s = t.update(0.8, 0.8)
    assert s["negative"] == 0.8 and s["arousal"] == 0.8 and s["n"] == 1
    s = t.update(0.2, 0.2)
    assert s["negative"] == pytest.approx(0.5) and s["arousal"] == pytest.approx(0.5)


def test_hysteresis_requires_consecutive_evidence():
    """锁定后单段噪声不翻转象限；连续两段一致才切换。"""
    t = StateTracker(TrackerConfig(ema_alpha=1.0, hysteresis_band=0.05, min_consecutive=2))
    assert t.update(0.8, 0.8)["locked"] is False            # 首段：临时判定，未锁定
    s = t.update(0.8, 0.8)                                   # 连续两段一致 → 锁定 Q2
    assert s["quadrant"] == "Q2" and s["locked"] is True
    assert t.update(0.2, 0.8)["quadrant"] == "Q2"           # 一段 Q1 证据：仍 Q2
    assert t.update(0.8, 0.8)["quadrant"] == "Q2"           # 回到 Q2：候选清零
    assert t.update(0.2, 0.8)["quadrant"] == "Q2"
    s = t.update(0.2, 0.8)                                   # 连续第二段 Q1 → 切换
    assert s["quadrant"] == "Q1" and s["flips"] == 1


def test_provisional_start_is_not_locked_by_first_wrong_segment():
    """会话开头判定是临时的：第一段判错、第二段不同，显示随最新候选走，不需要额外证据。"""
    t = StateTracker(TrackerConfig(ema_alpha=1.0, hysteresis_band=0.05, min_consecutive=2))
    assert t.update(0.8, 0.8)["quadrant"] == "Q2"           # 首段（可能是噪声）
    s = t.update(0.2, 0.8)                                   # 第二段 Q1：直接显示 Q1，仍未锁定
    assert s["quadrant"] == "Q1" and s["locked"] is False and s["flips"] == 0
    s = t.update(0.2, 0.8)                                   # 连续两段 Q1 → 锁定
    assert s["quadrant"] == "Q1" and s["locked"] is True


def test_deadzone_blocks_switch_near_midline():
    t = StateTracker(TrackerConfig(ema_alpha=1.0, hysteresis_band=0.05, min_consecutive=1))
    assert t.update(0.8, 0.8)["locked"] is True             # k=1：首段即锁定
    # 平滑点在死区内（|v−0.5| = 0.02）：即使候选是 Q1 也不切换
    assert t.update(0.48, 0.8)["quadrant"] == "Q2"
    assert t.update(0.48, 0.8)["stable"] is False
    assert t.update(0.3, 0.8)["quadrant"] == "Q1"           # 明确越过死区且 min_consecutive=1 → 切换


def test_tracker_state_empty_and_reset():
    t = StateTracker()
    assert t.state()["n"] == 0 and t.state()["quadrant"] is None
    t.update(0.7, 0.7)
    t.reset()
    assert t.state()["n"] == 0


def test_tracker_config_from_settings():
    cfg = TrackerConfig.from_settings({"session": {"ema_alpha": 0.3, "min_consecutive": 3},
                                       "thresholds": {"quadrant_mid_v": 0.5, "quadrant_mid_a": 0.5, "quadrant_band": 0.05}})
    assert cfg.ema_alpha == 0.3 and cfg.min_consecutive == 3 and cfg.hysteresis_band == 0.05


# ---------------- SessionDB ----------------
def _result(neg, ar, q):
    return {"negative": neg, "valence": 1 - neg, "arousal": ar, "dominant_quadrant": q,
            "uncertainty": {"negative_sd": 0.1, "arousal_sd": 0.2},
            "reliability": {"grade": "medium"}, "calibration_profile": "S01",
            "asr_text": "测试", "asr_confidence": 0.9,
            "modal_scores": {"acoustic": {"negative": neg, "arousal": ar}}, "memberships": {"Q2": 1.0}}


def test_session_trial_analysis_roundtrip_and_summary(tmp_path):
    db = SessionDB(tmp_path / "s.db")
    sid = db.create_session("S01", "焦虑（Q2）", notes="预实验", config={"ema_alpha": 0.5})
    assert db.get_session(sid)["induction_target"] == "焦虑（Q2）"
    tid, idx = db.new_trial(sid)
    assert idx == 1
    smoothed = {"negative": 0.7, "valence": 0.3, "arousal": 0.8, "quadrant": "Q2", "stable": True}
    db.add_analysis(tid, "pre", _result(0.7, 0.8, "Q2"), smoothed, audio_path="a.wav")
    db.add_analysis(tid, "pre", _result(0.75, 0.85, "Q2"), smoothed)
    db.record_stimulus(tid, {"f0": 280.0, "pr": 0.4, "harmony": "natural_harmonics"}, smoothed, 30.0)
    db.add_analysis(tid, "post", _result(0.55, 0.6, "Q2"), {**smoothed, "negative": 0.6, "arousal": 0.7})
    with pytest.raises(ValueError):
        db.add_analysis(tid, "during", _result(0.5, 0.5, "Q4"), smoothed)
    rows = db.analyses_of(tid)
    assert [r["phase"] for r in rows] == ["pre", "pre", "post"]
    assert [r["seq"] for r in rows] == [1, 2, 1]
    summ = db.trial_summary(sid)[0]
    assert summ["n_pre"] == 2 and summ["n_post"] == 1
    assert summ["pre_negative_mean"] == pytest.approx(0.725)
    assert summ["delta_negative"] == pytest.approx(0.55 - 0.725)
    assert summ["stimulus_f0"] == 280.0 and summ["stimulus_quadrant"] == "Q2"
    tid2, idx2 = db.new_trial(sid)
    assert idx2 == 2
    db.end_session(sid)
    lst = db.list_sessions()
    assert lst[0]["id"] == sid and lst[0]["n_trials"] == 2 and lst[0]["ended_at"]


def test_session_csv_exports(tmp_path):
    db = SessionDB(tmp_path / "s.db")
    sid = db.create_session(None, "中性")
    tid, _ = db.new_trial(sid)
    sm = {"negative": 0.5, "valence": 0.5, "arousal": 0.5, "quadrant": "Q4", "stable": False}
    db.add_analysis(tid, "pre", _result(0.5, 0.45, "Q4"), sm)
    db.add_analysis(tid, "post", _result(0.4, 0.5, "Q1"), sm)
    detail = db.export_session_csv(sid, tmp_path / "out" / "detail.csv")
    summary = db.export_trial_summary_csv(sid, tmp_path / "out" / "summary.csv")
    with detail.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["phase"] == "pre" and rows[1]["phase"] == "post"
    assert rows[0]["induction_target"] == "中性" and rows[0]["smoothed_quadrant"] == "Q4"
    with summary.open(encoding="utf-8-sig") as f:
        srows = list(csv.DictReader(f))
    assert len(srows) == 1 and srows[0]["trial"] == "1" and float(srows[0]["delta_negative"]) == pytest.approx(-0.1)


def test_delete_session_cascades(tmp_path):
    db = SessionDB(tmp_path / "s.db")
    sid = db.create_session("S02", "低落")
    tid, _ = db.new_trial(sid)
    db.add_analysis(tid, "pre", _result(0.8, 0.2, "Q3"), {"quadrant": "Q3"})
    db.delete_session(sid)
    assert db.get_session(sid) is None
    assert db.analyses_of(tid) == []
