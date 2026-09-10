"""会话内情绪状态估计：指数平滑 + 象限滞回。

【为何需要】v0.1–v0.4 把每段录音当成独立快照：相邻两段之间没有任何关联，象限会因
测量噪声来回翻转，刺激随之切换。真实情绪具有**惯性**（emotional inertia，Kuppens
et al. 2010：情绪状态在数分钟尺度上自相关很高），一分钟内不会来回跳；实验中受试者
的状态由诱发任务缓慢带入，一次会话通常只有一到两种目标状态。因此会话内应把多段录音
合并成一个状态估计，并让象限判定具有滞回，多段语音一致才改变一次判定。

【做法】
1. **指数平滑**（EMA）：s_t = α·x_t + (1−α)·s_{t−1}，对 negative 与 arousal 各自平滑；
   α 默认 0.5，首段直接初始化。α 越小越平稳、越迟钝。
2. **象限滞回**：候选象限由平滑后的点决定；只有当该点落在中线两侧的「死区」之外
   （|v−0.5| 与 |a−0.5| 都 > hysteresis_band，默认 0.05）**且连续 min_consecutive 段**
   （默认 2）都指向同一个新象限时才切换；否则维持上一判定。会话开头的判定是**临时的**
   （``locked=False``）：在连续 min_consecutive 段一致之前，显示的象限随最新候选走而不
   锁定——否则第一段若判错会被滞回「锁住」，序列模拟显示这会让约三成序列始终无法纠正。
3. 输出同时给出「当前段」与「会话平滑」两套数值，界面两者都显示；**驱动刺激的是
   平滑状态**。

【边界】平滑会引入滞后：状态真变了，至少要 min_consecutive 段之后才被承认。这是
有意的取舍——在本工具的使用场景里，误报翻转的代价（刺激切换、打断受试者）高于
延迟一两段承认变化的代价。参数都在 settings.json 的 ``session`` 段，可按实验调整。
v0.5 的评测（docs/evaluation.md §0.−1）用 CSEMOTIONS 同一说话人同一情绪的连续句子
模拟会话，量化了平滑对稳定性与准确率的影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.fusion.quadrant import compute_quadrant_memberships, dominant_quadrant


@dataclass
class TrackerConfig:
    ema_alpha: float = 0.5
    hysteresis_band: float = 0.05
    min_consecutive: int = 2
    mid_v: float = 0.5
    mid_a: float = 0.5
    quadrant_band: float = 0.05

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> TrackerConfig:
        sess = settings.get("session", {}) or {}
        thr = settings.get("thresholds", {}) or {}
        return cls(
            ema_alpha=float(sess.get("ema_alpha", 0.5)),
            hysteresis_band=float(sess.get("hysteresis_band", 0.05)),
            min_consecutive=int(sess.get("min_consecutive", 2)),
            mid_v=float(thr.get("quadrant_mid_v", 0.5)),
            mid_a=float(thr.get("quadrant_mid_a", 0.5)),
            quadrant_band=float(thr.get("quadrant_band", 0.05)),
        )


@dataclass
class StateTracker:
    """会话内状态跟踪器。每次 :meth:`update` 传入一段录音的融合结果。"""

    config: TrackerConfig = field(default_factory=TrackerConfig)
    n: int = 0
    negative: float | None = None
    arousal: float | None = None
    quadrant: str | None = None
    _candidate: str | None = None
    _candidate_count: int = 0
    _flips: int = 0
    locked: bool = False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.n = 0
        self.negative = self.arousal = None
        self.quadrant = self._candidate = None
        self._candidate_count = 0
        self._flips = 0
        self.locked = False

    def update(self, negative: float, arousal: float) -> dict[str, Any]:
        """并入一段录音的 (negative, arousal)，返回当前平滑状态。"""
        a = self.config.ema_alpha
        negative = min(1.0, max(0.0, float(negative)))
        arousal = min(1.0, max(0.0, float(arousal)))
        if self.n == 0 or self.negative is None or self.arousal is None:
            self.negative, self.arousal = negative, arousal
        else:
            self.negative = a * negative + (1 - a) * self.negative
            self.arousal = a * arousal + (1 - a) * self.arousal
        self.n += 1
        self._update_quadrant()
        return self.state()

    def _update_quadrant(self) -> None:
        c = self.config
        v, ar = 1.0 - self.negative, self.arousal
        memberships = compute_quadrant_memberships(v, ar, c.mid_v, c.mid_a, c.quadrant_band)
        candidate = dominant_quadrant(memberships)
        outside_deadzone = abs(v - c.mid_v) > c.hysteresis_band and abs(ar - c.mid_a) > c.hysteresis_band
        if not self.locked:
            # 起始阶段：显示随最新候选走；连续 min_consecutive 段一致（且在死区外）后才锁定
            if candidate == self._candidate and outside_deadzone:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = candidate, (1 if outside_deadzone else 0)
            self.quadrant = candidate
            if self._candidate_count >= c.min_consecutive:
                self.locked = True
                self._candidate_count = 0
            return
        if candidate == self.quadrant or not outside_deadzone:
            # 与当前一致，或落在死区内：不累计切换
            self._candidate, self._candidate_count = self.quadrant, 0
            return
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = candidate, 1
        if self._candidate_count >= c.min_consecutive:
            self.quadrant = candidate
            self._flips += 1
            self._candidate_count = 0

    def state(self) -> dict[str, Any]:
        """当前平滑状态（n=0 时各值为 None）。"""
        if self.n == 0 or self.negative is None or self.arousal is None:
            return {"n": 0, "negative": None, "valence": None, "arousal": None,
                    "quadrant": None, "memberships": None, "stable": False, "flips": 0}
        c = self.config
        v = 1.0 - self.negative
        memberships = compute_quadrant_memberships(v, self.arousal, c.mid_v, c.mid_a, c.quadrant_band)
        return {
            "n": self.n,
            "negative": self.negative,
            "valence": v,
            "arousal": self.arousal,
            "quadrant": self.quadrant,
            "memberships": memberships,
            # stable：平滑点的主象限与滞回判定一致，且不在死区内
            "stable": (self.locked and dominant_quadrant(memberships) == self.quadrant
                       and abs(v - c.mid_v) > c.hysteresis_band and abs(self.arousal - c.mid_a) > c.hysteresis_band),
            "locked": self.locked,
            "flips": self._flips,
        }
