"""界面主题：唯一调色板来源。

风格：暖奶油（warm ivory）——温暖、低饱和、以陶土橙为唯一强调色。
所有控件与样式表都从 :data:`PALETTE` 取色，代码中**不得**再出现硬编码色值
（``tests/test_gui_smoke.py::test_no_hardcoded_colors_outside_theme`` 守护）。

换主题只需改本文件。``styles.qss.tmpl`` 中以 ``{{key}}`` 引用调色板键，
由 :func:`build_qss` 渲染成最终 QSS。
"""

from __future__ import annotations

from pathlib import Path

PALETTE: dict[str, str] = {
    # 底色与面板
    "bg": "#FAF9F5",          # 页面底色（ivory）
    "card": "#FFFFFF",        # 主面板
    "card_alt": "#F0EEE6",    # 次级面板 / 状态栏
    "border": "#E5E2D9",      # 细线边框
    "border_strong": "#CFCBC0",
    # 文字
    "text": "#1F1E1D",        # 正文（slate）
    "text_muted": "#6E6D68",  # 次级文字
    "text_faint": "#A5A39C",  # 占位 / 待处理
    # 强调色（陶土橙）
    "accent": "#D97757",
    "accent_dark": "#C4633F",
    "accent_soft": "#F3DED3",  # 强调色的浅底（hover / 禁用主按钮）
    "accent_text": "#A8502E",  # 强调色用于**文字**时的深化版（白底对比度 ≈ 5.3:1）
    # 语义状态色（暖调；均按 WCAG AA 在 card / card_alt 底上对比度 ≥ 4.5:1 选定）
    "success": "#56703F",
    "warning": "#8A6200",
    "error": "#B23F2E",
    # 几何
    "radius": "8px",
    "radius_sm": "6px",
    # 字体
    "font_family": '"Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "Segoe UI", sans-serif',
}

_TEMPLATE_PATH = Path(__file__).with_name("styles.qss.tmpl")


def build_qss(template_path: Path | None = None) -> str:
    """把 ``styles.qss.tmpl`` 中的 ``{{key}}`` 替换为调色板值，返回 QSS 文本。

    未定义的键会原样保留（便于测试发现遗漏）。
    """
    path = template_path or _TEMPLATE_PATH
    text = path.read_text(encoding="utf-8")
    for key, value in PALETTE.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def color(key: str) -> str:
    """按键取色（供 pyqtgraph 等需要直接传色值的场合）。"""
    return PALETTE[key]


def inline(**props: str) -> str:
    """生成内联 QSS 片段，值中可用调色板键名（如 ``color="accent"``）。

    >>> inline(color="text_muted", font_size="12px")
    'color: #6E6D68; font-size: 12px;'
    """
    parts = []
    for prop, value in props.items():
        css_prop = prop.replace("_", "-")
        parts.append(f"{css_prop}: {PALETTE.get(value, value)};")
    return " ".join(parts)
