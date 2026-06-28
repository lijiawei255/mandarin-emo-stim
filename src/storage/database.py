"""SQLite 历史记录数据库。

单文件、零依赖、事务安全。存储每次分析的完整结果与关联的音频/刺激文件路径。
最大保存 200 条记录（由 ``settings.json -> history.max_records`` 控制），
超限时 :meth:`HistoryDB.add_record` 返回 -1 以阻止新增。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src import portable


class HistoryDB:
    """历史记录数据库封装（线程安全）。"""

    def __init__(self, db_path: Path | None = None, max_records: int = 200):
        """
        Args:
            db_path: 数据库文件路径。默认 ``portable_data/history/records.db``。
            max_records: 最大记录数，超出时阻止新增。
        """
        if db_path is None:
            db_path = portable.HISTORY_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.max_records = max_records
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # 开启 WAL 提升并发读写
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT    NOT NULL,
                    source          TEXT,            -- 'record' / 'upload'
                    audio_path      TEXT,            -- 录音 WAV 路径
                    stimulus_path   TEXT,            -- 生成的刺激 WAV 路径
                    duration        REAL,            -- 有效语音时长（秒）
                    negative        REAL,
                    valence         REAL,
                    arousal         REAL,
                    quadrant        TEXT,            -- Q1/Q2/Q3/Q4
                    asr_text        TEXT,
                    asr_confidence  REAL,
                    snr_db          REAL,
                    modal_scores    TEXT,            -- JSON 字符串
                    memberships     TEXT,            -- JSON 字符串
                    paralang_events TEXT,            -- JSON 字符串
                    stimulus_params TEXT             -- JSON 字符串
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    def get_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM records").fetchone()
            return int(row["c"])

    def add_record(self, record: dict[str, Any]) -> int:
        """新增一条记录。

        Args:
            record: 字段字典（键对应表列名；modal_scores/memberships/
                paralang_events/stimulus_params 若为 dict 会自动 JSON 序列化）。

        Returns:
            新记录 id；若已达 ``max_records`` 上限返回 -1。
        """
        with self._lock, self._connect() as conn:
            cur_count = conn.execute("SELECT COUNT(*) AS c FROM records").fetchone()["c"]
            if cur_count >= self.max_records:
                return -1

            record = dict(record)
            record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            for key in ("modal_scores", "memberships", "paralang_events", "stimulus_params"):
                val = record.get(key)
                if isinstance(val, (dict, list)):
                    record[key] = json.dumps(val, ensure_ascii=False)

            cols = ", ".join(record.keys())
            placeholders = ", ".join("?" * len(record))
            cur = conn.execute(
                f"INSERT INTO records ({cols}) VALUES ({placeholders})",
                tuple(record.values()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_record(self, record_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_record(self, record_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear_all(self) -> int:
        """清空全部记录，返回被删除的条数。"""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM records")
            conn.commit()
            return int(cur.rowcount)

    def get_all(self) -> list[dict[str, Any]]:
        """返回全部记录（按 id 升序）。用于导出。"""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM records ORDER BY id ASC").fetchall()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        # 反序列化 JSON 字段
        for key in ("modal_scores", "memberships", "paralang_events", "stimulus_params"):
            val = d.get(key)
            if isinstance(val, str) and val:
                try:
                    d[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
