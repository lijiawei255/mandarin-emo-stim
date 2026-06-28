"""波形可视化控件（pyqtgraph）。

浅色包豪斯风格：白色背景上绘制主色（蓝）波形线条，播放时主色进度线。
为性能对长波形做下采样（每像素 ~1 个点）。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

# 包豪斯主色
_PRIMARY = "#1F5FA8"


class WaveformView(QWidget):
    """声刺激波形展示控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._duration = 0.0

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOption("background", "#FFFFFF")
        pg.setConfigOption("foreground", "#1A1A1A")
        self.plot = pg.PlotWidget()
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.getAxis("bottom").setPen("#9A9A9A")
        self.plot.getAxis("left").setPen("#9A9A9A")
        self.plot.getAxis("bottom").setTextPen("#6A6A6A")
        self.plot.getAxis("left").setTextPen("#6A6A6A")
        self.plot.setLabel("bottom", "时间 (s)", color="#6A6A6A")
        self.plot.setLabel("left", "幅度", color="#6A6A6A")
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        layout.addWidget(self.plot)

        self.wave_item = pg.PlotCurveItem(pen=pg.mkPen(_PRIMARY, width=1))
        self.plot.addItem(self.wave_item)

        # 播放进度线（主色虚线）
        self.position_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(_PRIMARY, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.position_line)
        self.position_line.setVisible(False)

    def set_waveform(self, data: np.ndarray, sr: int) -> None:
        """设置并绘制波形（自动下采样）。

        Args:
            data: 双声道或单声道波形。展示时取左声道。
            sr: 采样率。
        """
        if data.ndim > 1:
            mono = data[:, 0]
        else:
            mono = data
        self._duration = len(mono) / sr if sr > 0 else 0.0

        # 下采样：目标约 2000 个点以保证性能
        target_points = 2000
        step = max(1, len(mono) // target_points)
        downsampled = mono[::step]
        t = np.linspace(0, self._duration, len(downsampled))

        self.wave_item.setData(t, downsampled)
        self.plot.setXRange(0, self._duration, padding=0.02)
        peak = float(np.max(np.abs(downsampled))) or 1.0
        self.plot.setYRange(-peak * 1.1, peak * 1.1, padding=0)

    def set_position(self, seconds: float) -> None:
        """更新播放进度线位置。"""
        if self._duration <= 0:
            return
        self.position_line.setVisible(True)
        self.position_line.setPos(seconds)

    def clear_position(self) -> None:
        self.position_line.setVisible(False)

    @property
    def duration(self) -> float:
        return self._duration
