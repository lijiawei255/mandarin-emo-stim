"""会话 / 试次数据模型、SQLite 持久化与 CSV 导出（v0.5）。

【结构】受试者档案（可选）→ **会话**（一次实验，带诱发目标标签与备注）→ **试次**
（trial）→ 三段：``pre`` 前测录音若干段、``stimulus`` 一次刺激、``post`` 后测录音若干段。
每段录音的分析结果连同当时的会话平滑状态一起落库；前后测对比是干预效果的证据，
这才是文档里「闭环」一词应有的含义（v0.4 之前放完刺激流程就结束了，是开环）。

【存储】``portable_data/sessions/sessions.db``（与历史记录库分开，便于整库导出/删除）。
【导出】``export_session_csv``：逐段明细；``export_trial_summary_csv``：每试次前测 / 后测的
均值与差值、刺激参数、诱发目标，直接可进统计软件。CSV 为 UTF-8-BOM，Excel 可直接打开。
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src import portable

SESSIONS_DIR: Path = portable.PORTABLE_DATA_ROOT / "sessions"
DEFAULT_DB_PATH: Path = SESSIONS_DIR / "sessions.db"
PHASES = ("pre", "post")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SessionDB:
    """会话数据库（线程安全，WAL）。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at       TEXT NOT NULL,
                    ended_at         TEXT,
                    participant      TEXT,            -- 受试者档案名（可空）
                    induction_target TEXT,            -- 实验者标注的诱发目标（如 焦虑 / Q2 / 中性）
                    notes            TEXT,
                    config_json      TEXT             -- 当时的 session 配置（平滑参数）
                );
                CREATE TABLE IF NOT EXISTS trials (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    trial_index      INTEGER NOT NULL,
                    created_at       TEXT NOT NULL,
                    stimulus_at      TEXT,
                    stimulus_json    TEXT,            -- 刺激参数 + 驱动它的平滑状态
                    UNIQUE(session_id, trial_index)
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id         INTEGER NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
                    phase            TEXT NOT NULL,   -- pre / post
                    seq              INTEGER NOT NULL, -- 该阶段内序号
                    created_at       TEXT NOT NULL,
                    audio_path       TEXT,
                    negative REAL, valence REAL, arousal REAL, quadrant TEXT,
                    smoothed_negative REAL, smoothed_valence REAL, smoothed_arousal REAL,
                    smoothed_quadrant TEXT, smoothed_stable INTEGER,
                    uncertainty_neg REAL, uncertainty_ar REAL, reliability TEXT,
                    calibration_profile TEXT, asr_text TEXT, asr_confidence REAL,
                    result_json      TEXT             -- 完整结果（模态分等）
                );
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #
    def create_session(self, participant: str | None, induction_target: str,
                       notes: str = "", config: dict[str, Any] | None = None) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (created_at, participant, induction_target, notes, config_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (_now(), participant, induction_target, notes,
                 json.dumps(config or {}, ensure_ascii=False)))
            conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))
            conn.commit()

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM trials t WHERE t.session_id = s.id) AS n_trials "
                "FROM sessions s ORDER BY s.id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    # ------------------------------------------------------------------ #
    # 试次
    # ------------------------------------------------------------------ #
    def new_trial(self, session_id: int) -> tuple[int, int]:
        """新建试次，返回 (trial_id, trial_index)。"""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(trial_index), 0) AS m FROM trials WHERE session_id = ?",
                               (session_id,)).fetchone()
            idx = int(row["m"]) + 1
            cur = conn.execute("INSERT INTO trials (session_id, trial_index, created_at) VALUES (?, ?, ?)",
                               (session_id, idx, _now()))
            conn.commit()
            return int(cur.lastrowid), idx

    def record_stimulus(self, trial_id: int, params: dict[str, Any], driving_state: dict[str, Any],
                        duration_sec: float | None = None) -> None:
        payload = {"params": params, "driving_state": driving_state, "duration_sec": duration_sec}
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE trials SET stimulus_at = ?, stimulus_json = ? WHERE id = ?",
                         (_now(), json.dumps(payload, ensure_ascii=False, default=float), trial_id))
            conn.commit()

    def add_analysis(self, trial_id: int, phase: str, result: dict[str, Any],
                     smoothed: dict[str, Any], audio_path: str | None = None) -> int:
        if phase not in PHASES:
            raise ValueError(f"phase 必须是 {PHASES}")
        unc = result.get("uncertainty") or {}
        rel = result.get("reliability") or {}
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM analyses WHERE trial_id = ? AND phase = ?",
                               (trial_id, phase)).fetchone()
            seq = int(row["m"]) + 1
            slim = {k: v for k, v in result.items()
                    if k in ("modal_scores", "modal_scores_raw", "memberships", "weights", "paralang_events",
                             "degraded_modalities", "fusion_mode", "calibration_source", "audio_quality",
                             "duration", "asr_confidence_source")}
            cur = conn.execute(
                """INSERT INTO analyses (trial_id, phase, seq, created_at, audio_path,
                       negative, valence, arousal, quadrant,
                       smoothed_negative, smoothed_valence, smoothed_arousal, smoothed_quadrant, smoothed_stable,
                       uncertainty_neg, uncertainty_ar, reliability, calibration_profile, asr_text, asr_confidence,
                       result_json)
                   VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?)""",
                (trial_id, phase, seq, _now(), audio_path,
                 result.get("negative"), result.get("valence"), result.get("arousal"), result.get("dominant_quadrant"),
                 smoothed.get("negative"), smoothed.get("valence"), smoothed.get("arousal"),
                 smoothed.get("quadrant"), 1 if smoothed.get("stable") else 0,
                 unc.get("negative_sd"), unc.get("arousal_sd"), rel.get("grade"),
                 result.get("calibration_profile"), result.get("asr_text"), result.get("asr_confidence"),
                 json.dumps(slim, ensure_ascii=False, default=float)))
            conn.commit()
            return int(cur.lastrowid)

    def trials_of(self, session_id: int) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM trials WHERE session_id = ? ORDER BY trial_index", (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def analyses_of(self, trial_id: int) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM analyses WHERE trial_id = ? ORDER BY phase DESC, seq", (trial_id,)).fetchall()
            # phase DESC 让 pre 在 post 前（'pre' > 'post' 按字母序为假，故手动排）
            out = [dict(r) for r in rows]
            return sorted(out, key=lambda r: (PHASES.index(r["phase"]), r["seq"]))

    # ------------------------------------------------------------------ #
    # 汇总与导出
    # ------------------------------------------------------------------ #
    def trial_summary(self, session_id: int) -> list[dict[str, Any]]:
        """每试次：前测 / 后测各指标均值、差值、刺激参数。"""
        sess = self.get_session(session_id) or {}
        out: list[dict[str, Any]] = []
        for t in self.trials_of(session_id):
            rows = self.analyses_of(t["id"])
            stim = json.loads(t["stimulus_json"]) if t.get("stimulus_json") else {}
            summary: dict[str, Any] = {
                "session_id": session_id, "participant": sess.get("participant"),
                "induction_target": sess.get("induction_target"), "trial": t["trial_index"],
                "n_pre": 0, "n_post": 0,
                "stimulus_quadrant": (stim.get("driving_state") or {}).get("quadrant"),
                "stimulus_f0": (stim.get("params") or {}).get("f0"),
                "stimulus_pr": (stim.get("params") or {}).get("pr"),
                "stimulus_harmony": (stim.get("params") or {}).get("harmony"),
                "stimulus_duration_sec": stim.get("duration_sec"),
            }
            for phase in PHASES:
                ph = [r for r in rows if r["phase"] == phase]
                summary[f"n_{phase}"] = len(ph)
                for k in ("negative", "valence", "arousal"):
                    vals = [r[k] for r in ph if r[k] is not None]
                    summary[f"{phase}_{k}_mean"] = sum(vals) / len(vals) if vals else None
                summary[f"{phase}_last_smoothed_quadrant"] = ph[-1]["smoothed_quadrant"] if ph else None
            for k in ("negative", "valence", "arousal"):
                a, b = summary.get(f"pre_{k}_mean"), summary.get(f"post_{k}_mean")
                summary[f"delta_{k}"] = (b - a) if (a is not None and b is not None) else None
            out.append(summary)
        return out

    def export_session_csv(self, session_id: int, path: Path) -> Path:
        """逐段明细导出（UTF-8-BOM）。"""
        sess = self.get_session(session_id) or {}
        cols = ["session_id", "participant", "induction_target", "trial", "phase", "seq", "created_at",
                "negative", "valence", "arousal", "quadrant",
                "smoothed_negative", "smoothed_valence", "smoothed_arousal", "smoothed_quadrant", "smoothed_stable",
                "uncertainty_neg", "uncertainty_ar", "reliability", "calibration_profile",
                "asr_confidence", "asr_text", "audio_path"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for t in self.trials_of(session_id):
                for r in self.analyses_of(t["id"]):
                    w.writerow({"session_id": session_id, "participant": sess.get("participant"),
                                "induction_target": sess.get("induction_target"), "trial": t["trial_index"],
                                **{c: r.get(c) for c in cols if c in r}})
        return path

    def export_trial_summary_csv(self, session_id: int, path: Path) -> Path:
        rows = self.trial_summary(session_id)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys()) if rows else ["session_id", "trial"]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path
