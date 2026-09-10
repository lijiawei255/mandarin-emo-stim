"""可学习融合：岭回归把 6 模态 × 2 的原始分线性映射到 (negative, arousal)。

【定位】手工权重（``config/settings.json`` 的 fusion_weights）是启发式；本模块提供
一个**可解释的线性替代**：negative = w_s·x + b_s，arousal = w_a·x + b_a，
x 为 12 维（各模态的 negative 与 arousal 原始分）。用带 L2 正则的最小二乘
拟合，系数保存在 ``config/learned_fusion.json``，可直接阅读每个模态的贡献。

【训练目标】评测语料只有类别标签，回归目标取各类别在 Russell 环上的参照坐标
（``EMOTION_TARGETS``），是对连续标签的粗略替代——因此学到的映射只能保证
「类别中心」的相对位置，不能当作校准好的连续量。

【默认不启用】settings.json 的 ``fusion_mode`` 为 ``"weighted"``（手工权重）。设为
``"learned"`` 后 WeightedFusion 用本模块的系数计算 negative/arousal，其余（动态
权重、软象限、不确定性）不变。学到的系数来自**表演型**语料 CSEMOTIONS 的交叉
验证（docs/evaluation.md §0），迁移到自然语音的效果未验证。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MODALITIES = ("acoustic", "prosody", "paralang", "physical", "text_llm", "text_stat")
FEATURE_NAMES = tuple(f"{m}.{ax}" for m in MODALITIES for ax in ("negative", "arousal"))

# 类别 → (negative, arousal) 参照坐标；相对位置取自 Russell (1980)，数值为启发式
EMOTION_TARGETS: dict[str, tuple[float, float]] = {
    "angry": (0.85, 0.85),
    "fearful": (0.80, 0.80),
    "sad": (0.80, 0.25),
    "neutral": (0.50, 0.45),
    "happy": (0.15, 0.75),
    "playfulness": (0.25, 0.70),
    "surprise": (0.50, 0.85),
}


def features_from_raw(raw: dict[str, dict[str, float]]) -> np.ndarray:
    """``modal_scores_raw`` → 12 维特征向量（缺失模态填 0.5）。"""
    x = []
    for m in MODALITIES:
        v = raw.get(m) or {}
        x.append(float(v.get("negative", 0.5)))
        x.append(float(v.get("arousal", 0.5)))
    return np.asarray(x, dtype=np.float64)


def fit_ridge(X: np.ndarray, Y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """岭回归（截距不正则）。X: (n, 12)，Y: (n, 2)。返回 (W: (12, 2), b: (2,))。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    ym = Y.mean(axis=0)
    Yc = Y - ym
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + l2 * np.eye(d), Xc.T @ Yc)
    b = ym - mu @ W
    return W, b


class LearnedFusion:
    """线性融合器。"""

    def __init__(self, W: np.ndarray, b: np.ndarray, meta: dict[str, Any] | None = None):
        self.W = np.asarray(W, dtype=np.float64).reshape(len(FEATURE_NAMES), 2)
        self.b = np.asarray(b, dtype=np.float64).reshape(2)
        self.meta = meta or {}

    def predict(self, raw: dict[str, dict[str, float]]) -> tuple[float, float]:
        y = features_from_raw(raw) @ self.W + self.b
        return float(np.clip(y[0], 0.0, 1.0)), float(np.clip(y[1], 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "_meta": self.meta,
            "features": list(FEATURE_NAMES),
            "coefficients": {
                "negative": {f: round(float(self.W[i, 0]), 5) for i, f in enumerate(FEATURE_NAMES)},
                "arousal": {f: round(float(self.W[i, 1]), 5) for i, f in enumerate(FEATURE_NAMES)},
            },
            "intercept": {"negative": round(float(self.b[0]), 5), "arousal": round(float(self.b[1]), 5)},
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedFusion:
        W = np.zeros((len(FEATURE_NAMES), 2))
        for i, f in enumerate(FEATURE_NAMES):
            W[i, 0] = float(data["coefficients"]["negative"].get(f, 0.0))
            W[i, 1] = float(data["coefficients"]["arousal"].get(f, 0.0))
        b = np.array([float(data["intercept"]["negative"]), float(data["intercept"]["arousal"])])
        return cls(W, b, data.get("_meta"))

    @classmethod
    def load(cls, path: Path) -> LearnedFusion | None:
        """读取系数文件；缺失或损坏返回 ``None``（调用方回退到手工权重）。"""
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None


def fit_from_rows(rows: list[dict[str, Any]], l2: float = 1.0,
                  targets: dict[str, tuple[float, float]] | None = None,
                  meta: dict[str, Any] | None = None) -> LearnedFusion:
    """由评测缓存行（含 ``modal_scores_raw`` 与 ``emotion``）拟合。"""
    targets = targets or EMOTION_TARGETS
    X, Y = [], []
    for r in rows:
        if r.get("emotion") not in targets:
            continue
        X.append(features_from_raw(r.get("modal_scores_raw") or r.get("modal_scores") or {}))
        Y.append(targets[r["emotion"]])
    if len(X) < len(FEATURE_NAMES) + 2:
        raise ValueError(f"样本过少（{len(X)}），无法拟合 {len(FEATURE_NAMES)} 维线性模型")
    W, b = fit_ridge(np.asarray(X), np.asarray(Y), l2=l2)
    m = {"n_train": len(X), "l2": l2, "targets": {k: list(v) for k, v in targets.items()}, **(meta or {})}
    return LearnedFusion(W, b, m)
