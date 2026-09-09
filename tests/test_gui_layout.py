"""GUI 布局几何回归：3 分辨率 × 3 状态下无重叠 / 挤压 / 溢出 / 压扁。

使用 offscreen 平台（CI 可跑）。注意 offscreen 无 CJK 字体时中文以方块
渲染，但字宽度量仍然有效，几何检查结论不受影响；真实字体下的目视核验
见 scripts/gui_screenshot.py 与 docs/superpowers/plans/ui-visual-check.md。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.slow

from scripts.ui_geometry_check import SIZES, STATES, collect_issues, make_window  # noqa: E402


@pytest.fixture(autouse=True)
def _noop_messagebox(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("warning", "information", "critical", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None), raising=False)


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
@pytest.mark.parametrize("state", STATES)
def test_layout_has_no_geometry_issues(size, state):
    app, win = make_window(*size, state)
    try:
        issues = collect_issues(win)
        assert issues == [], "\n".join(str(i) for i in issues)
    finally:
        win.close()
