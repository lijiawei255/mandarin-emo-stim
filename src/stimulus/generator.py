"""声刺激生成主控。

对外提供 :class:`StimulusGenerator`，封装「参数计算 → 波形合成」全流程。
对应项目计划文档 3.6 节与 Quick Start。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config_loader import load_settings, load_stimulus_params
from src.stimulus.strategies import compute_params
from src.stimulus.synthesizer import synthesize


class StimulusGenerator:
    """差异化声刺激生成器。"""

    def __init__(self, config: dict[str, Any] | None = None, sr: int | None = None):
        """
        Args:
            config: settings.json 全部内容。为 ``None`` 时自动加载默认配置。
            sr: 刺激采样率。为 ``None`` 时取配置中的 ``audio.stimulus_sample_rate``。
        """
        self.config = config if config is not None else load_settings()
        self.stimulus_config = load_stimulus_params()
        if sr is None:
            sr = int(self.config["audio"]["stimulus_sample_rate"])
        self.sr = sr
        # slab 默认采样率与刺激采样率保持一致
        try:
            import slab
            slab.set_default_samplerate(self.sr)
        except Exception:
            pass

    def generate(self, valence: float, arousal: float,
                 memberships: dict[str, float] | None = None,
                 duration: float | None = None) -> np.ndarray:
        """生成声刺激波形。

        Args:
            valence: 效价 [0,1]。
            arousal: 唤醒度 [0,1]。
            memberships: 四象限隶属度。为 ``None`` 时由 (valence, arousal) 计算。
            duration: 时长（秒）。为 ``None`` 时取配置默认值。

        Returns:
            ``np.ndarray`` shape=(n_samples, 2) dtype=float32。
        """
        from src.fusion.quadrant import compute_quadrant_memberships

        thr = self.config["thresholds"]
        if memberships is None:
            memberships = compute_quadrant_memberships(
                valence, arousal,
                thr["quadrant_mid_v"], thr["quadrant_mid_a"], thr["quadrant_band"],
            )

        if duration is None:
            duration = float(self.config["audio"]["stimulus_duration_sec"])
        # 约束时长到合法区间
        audio_cfg = self.config["audio"]
        duration = max(float(audio_cfg["stimulus_min_sec"]),
                       min(float(audio_cfg["stimulus_max_sec"]), duration))

        params = compute_params(valence, arousal, memberships, self.stimulus_config)
        waveform = synthesize(params, duration, self.sr, self.stimulus_config)
        return waveform

    def params_for(self, valence: float, arousal: float,
                   memberships: dict[str, float] | None = None):
        """返回计算出的声学参数（便于 UI 展示，不合成波形）。"""
        from src.fusion.quadrant import compute_quadrant_memberships

        thr = self.config["thresholds"]
        if memberships is None:
            memberships = compute_quadrant_memberships(
                valence, arousal,
                thr["quadrant_mid_v"], thr["quadrant_mid_a"], thr["quadrant_band"],
            )
        return compute_params(valence, arousal, memberships, self.stimulus_config)
