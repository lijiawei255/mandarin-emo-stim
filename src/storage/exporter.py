"""历史记录导出工具（JSON / CSV）。

CSV 使用 ``utf-8-sig`` 编码（带 BOM），确保 Excel 直接打开中文不乱码。
JSON 字段（modal_scores 等）在 CSV 中以 JSON 字符串形式写入单列。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# CSV 导出时展开的顶层列（其余 JSON 字段单独成列）
_CSV_COLUMNS = [
    "id", "created_at", "source", "audio_path", "stimulus_path", "duration",
    "negative", "valence", "arousal", "quadrant", "asr_text", "asr_confidence",
    "snr_db", "modal_scores", "memberships", "paralang_events", "stimulus_params",
]


def export_json(records: list[dict[str, Any]], path: Path) -> Path:
    """导出为 JSON 文件。

    Args:
        records: 记录列表（来自 :meth:`HistoryDB.get_all`）。
        path: 输出文件路径。

    Returns:
        写入的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path


def export_csv(records: list[dict[str, Any]], path: Path) -> Path:
    """导出为 CSV 文件（utf-8-sig，Excel 友好）。

    Args:
        records: 记录列表。
        path: 输出文件路径。

    Returns:
        写入的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        for rec in records:
            row = []
            for col in _CSV_COLUMNS:
                val = rec.get(col, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)
                row.append(val)
            writer.writerow(row)
    return path


def export(records: list[dict[str, Any]], path: Path, fmt: str = "json") -> Path:
    """按格式导出。

    Args:
        records: 记录列表。
        path: 输出路径。
        fmt: ``"json"`` 或 ``"csv"``。

    Returns:
        写入的文件路径。
    """
    fmt = fmt.lower().lstrip(".")
    if fmt == "csv":
        return export_csv(records, path)
    if fmt == "json":
        return export_json(records, path)
    raise ValueError(f"不支持的导出格式: {fmt}（仅支持 json / csv）")
