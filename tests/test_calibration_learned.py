"""v0.3：个人基线校准、可学习融合、不确定性输出。"""

import json

import numpy as np
import pytest

from src.config_loader import load_settings
from src.fusion import learned_fusion, personal_calibration
from src.fusion.weighted_fusion import MODALITIES, WeightedFusion


def _raw(s=0.5, a=0.5):
    return {m: {"negative": s, "arousal": a} for m in MODALITIES}


# ---------------- personal calibration ----------------
def test_personal_offsets_from_multiple_recordings_and_clip():
    rows = [_raw(0.6, 0.3), _raw(0.7, 0.2)]
    off = personal_calibration.compute_offsets(rows)
    assert "paralang" not in off                       # 默认排除副语言
    assert off["acoustic"] == {"negative": -0.15, "arousal": 0.25}
    big = personal_calibration.compute_offsets([_raw(0.95, 0.05)])
    assert big["prosody"] == {"negative": -0.3, "arousal": 0.3}   # 钳制 ±0.3
    assert personal_calibration.compute_offsets([]) == {}


def test_personal_save_load_clear(tmp_path):
    p = tmp_path / "user_baseline.json"
    off = personal_calibration.compute_offsets([_raw(0.6, 0.4)])
    personal_calibration.save(off, p, meta={"source": "test"})
    loaded = personal_calibration.load(p)
    assert loaded["acoustic"] == (-0.1, 0.1)
    assert json.loads(p.read_text(encoding="utf-8"))["_meta"]["source"] == "test"
    assert personal_calibration.clear(p) is True
    assert personal_calibration.load(p) == {}
    assert personal_calibration.clear(p) is False


def test_personal_offsets_override_corpus_offsets(tmp_path):
    """有个人基线时，个人偏移替代语料偏移（仅对有个人数据的模态）。"""
    cfg = json.loads(json.dumps(load_settings()))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"offsets": {
        "acoustic": {"negative": 0.05, "arousal": 0.05},
        "paralang": {"negative": 0.02, "arousal": 0.0},
    }}), encoding="utf-8")
    personal = tmp_path / "user.json"
    personal_calibration.save({"acoustic": {"negative": -0.2, "arousal": 0.1}}, personal)
    cfg["fusion_calibration"] = {"enabled": True, "path": str(corpus),
                                 "personal": True, "personal_path": str(personal)}
    fus = WeightedFusion(cfg)
    assert fus.offsets["acoustic"] == (-0.2, 0.1)   # 个人覆盖
    assert fus.offsets["paralang"] == (0.02, 0.0)   # 语料保留
    assert fus.calibration_source == "personal"
    cfg["fusion_calibration"]["personal"] = False
    assert WeightedFusion(cfg).offsets["acoustic"] == (0.05, 0.05)


# ---------------- learned fusion ----------------
def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 12))
    W_true = np.zeros((12, 2))
    W_true[0, 0] = 0.8
    W_true[3, 1] = 0.6
    Y = X @ W_true + np.array([0.1, 0.2])
    W, b = learned_fusion.fit_ridge(X, Y, l2=1e-6)
    assert np.allclose(W, W_true, atol=1e-3)
    assert np.allclose(b, [0.1, 0.2], atol=1e-3)


def test_learned_fusion_roundtrip_and_predict(tmp_path):
    rows = []
    rng = np.random.default_rng(1)
    for emo, (tn, ta) in learned_fusion.EMOTION_TARGETS.items():
        for _ in range(8):
            raw = {m: {"negative": float(np.clip(tn + rng.normal(0, 0.05), 0, 1)),
                       "arousal": float(np.clip(ta + rng.normal(0, 0.05), 0, 1))} for m in MODALITIES}
            rows.append({"emotion": emo, "modal_scores_raw": raw})
    lf = learned_fusion.fit_from_rows(rows, l2=0.1)
    p = tmp_path / "learned.json"
    lf.save(p)
    lf2 = learned_fusion.LearnedFusion.load(p)
    assert lf2 is not None
    n, a = lf2.predict(_raw(0.85, 0.85))
    assert n > 0.7 and a > 0.7
    n, a = lf2.predict(_raw(0.15, 0.75))
    assert n < 0.3
    assert learned_fusion.LearnedFusion.load(tmp_path / "missing.json") is None


def test_weighted_fusion_learned_mode_falls_back_when_missing(tmp_path):
    cfg = json.loads(json.dumps(load_settings()))
    cfg["fusion_mode"] = "learned"
    cfg["learned_fusion_path"] = str(tmp_path / "nope.json")
    fus = WeightedFusion(cfg)
    r = fus.fuse({m: (0.9, 0.9) for m in MODALITIES})
    assert r["fusion_mode"] == "weighted"     # 回退
    assert r["negative"] > 0.8


def test_weighted_fusion_learned_mode_uses_coefficients(tmp_path):
    cfg = json.loads(json.dumps(load_settings()))
    cfg["fusion_calibration"] = {"enabled": False}
    W = np.zeros((12, 2))
    W[0, 0] = 1.0      # negative 只看 acoustic.negative
    W[1, 1] = 1.0      # arousal 只看 acoustic.arousal
    lf = learned_fusion.LearnedFusion(W, np.zeros(2))
    p = tmp_path / "learned.json"
    lf.save(p)
    cfg["fusion_mode"] = "learned"
    cfg["learned_fusion_path"] = str(p)
    fus = WeightedFusion(cfg)
    scores = {m: (0.5, 0.5) for m in MODALITIES}
    scores["acoustic"] = (0.9, 0.2)
    r = fus.fuse(scores)
    assert r["fusion_mode"] == "learned"
    assert r["negative"] == pytest.approx(0.9)
    assert r["arousal"] == pytest.approx(0.2)


# ---------------- uncertainty ----------------
def test_uncertainty_reflects_modality_disagreement():
    cfg = json.loads(json.dumps(load_settings()))
    cfg["fusion_calibration"] = {"enabled": False}
    fus = WeightedFusion(cfg)
    agree = fus.fuse({m: (0.8, 0.8) for m in MODALITIES})
    scores = {m: (0.8, 0.8) for m in MODALITIES}
    scores["text_llm"] = (0.1, 0.1)
    scores["prosody"] = (0.2, 0.2)
    disagree = fus.fuse(scores)
    assert agree["uncertainty"]["negative_sd"] == pytest.approx(0.0, abs=1e-9)
    assert disagree["uncertainty"]["negative_sd"] > 0.2
    assert disagree["uncertainty"]["n_active"] == 6
    # 降级模态不计入分歧
    r = fus.fuse(scores, skip_calibration={"text_llm"})
    assert r["uncertainty"]["n_active"] == 5
