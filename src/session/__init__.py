"""会话模式（v0.5）：受试者 → 会话 → 试次（前测 / 刺激 / 后测）。

- :mod:`src.session.state_tracker`：会话内的情绪状态估计（指数平滑 + 象限滞回）。
- :mod:`src.session.model`：会话 / 试次 / 分析记录的数据模型、SQLite 持久化与 CSV 导出。
"""
