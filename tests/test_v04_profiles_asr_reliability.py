"""v0.4：受试者档案、ASR 后验置信度、可信度等级。"""

import json

import pytest

from src.config_loader import load_settings
from src.fusion import personal_calibration as pc
from src.fusion import reliability
from src.fusion.weighted_fusion import MODALITIES, WeightedFusion


def _raw(s=0.5, a=0.5):
    return {m: {"negative": s, "arousal": a} for m in MODALITIES}


# ---------------- 受试者档案 ----------------
def test_profile_save_append_average_and_list(tmp_path):
    d = tmp_path / "profiles"
    pc.save_profile("S01", [_raw(0.6, 0.4)], meta={"source": "a.wav"}, profiles_dir=d)
    pc.save_profile("S01", [_raw(0.8, 0.2)], profiles_dir=d)          # 追加 → 平均
    assert pc.list_profiles(d) == ["S01"]
    off = pc.load_profile("S01", d)
    assert off["acoustic"] == pytest.approx((-0.2, 0.2))              # mean 0.7/0.3
    info = pc.profile_info("S01", d)
    assert info["n_samples"] == 2 and info["name"] == "S01"
    pc.save_profile("S01", [_raw(0.5, 0.5)], append=False, profiles_dir=d)   # 覆盖
    assert pc.profile_info("S01", d)["n_samples"] == 1
    assert pc.load_profile("S01", d)["acoustic"] == (0.0, 0.0)


def test_profile_name_validation_and_delete(tmp_path):
    d = tmp_path / "profiles"
    a = tmp_path / "active.json"
    with pytest.raises(ValueError):
        pc.save_profile("../evil", [_raw()], profiles_dir=d)
    pc.save_profile("受试者_03", [_raw(0.6, 0.6)], profiles_dir=d)
    pc.set_active("受试者_03", a)
    assert pc.get_active(a) == "受试者_03"
    assert pc.delete_profile("受试者_03", d, a) is True
    assert pc.get_active(a) is None                                   # 删除激活档案 → 取消激活
    assert pc.list_profiles(d) == []


def test_load_active_prefers_profile_then_legacy(tmp_path):
    d = tmp_path / "profiles"
    a = tmp_path / "active.json"
    legacy = tmp_path / "user_baseline.json"
    assert pc.load_active(d, a, legacy) == ({}, None)
    pc.save(pc.compute_offsets([_raw(0.6, 0.6)]), legacy)
    off, name = pc.load_active(d, a, legacy)
    assert name == "user_baseline" and off["acoustic"] == (-0.1, -0.1)
    pc.save_profile("P1", [_raw(0.4, 0.4)], profiles_dir=d)
    pc.set_active("P1", a)
    off, name = pc.load_active(d, a, legacy)
    assert name == "P1" and off["acoustic"] == (0.1, 0.1)


def test_fusion_uses_active_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(pc, "ACTIVE_PATH", tmp_path / "active.json")
    monkeypatch.setattr(pc, "DEFAULT_PATH", tmp_path / "legacy.json")
    cfg = json.loads(json.dumps(load_settings()))
    cfg["fusion_calibration"] = {"enabled": True, "path": str(tmp_path / "corpus.json"), "personal": True}
    pc.save_profile("S02", [_raw(0.7, 0.3)])
    pc.set_active("S02")
    fus = WeightedFusion(cfg)
    assert fus.calibration_source == "personal"
    assert fus.calibration_profile == "S02"
    assert fus.offsets["acoustic"] == pytest.approx((-0.2, 0.2))
    r = fus.fuse({m: (0.7, 0.3) for m in MODALITIES})
    assert r["calibration_profile"] == "S02"
    assert r["modal_scores"]["acoustic"] == {"negative": pytest.approx(0.5), "arousal": pytest.approx(0.5)}


# ---------------- ASR 后验置信度 ----------------
def test_asr_posterior_confidence_from_hooked_decoder():
    import torch

    from src.models.asr_model import ASRModel

    class _Inner:
        sos, eos, blank_id = 1, 2, 0

        def cal_decoder_with_predictor(self, *a, **k):
            # 3 个位置：token 5 (p=0.9)、token 7 (p=0.6)、blank 0 (p=0.99，应被跳过)
            probs = torch.zeros((1, 3, 8))
            probs[0, 0, 5], probs[0, 0, 6] = 0.9, 0.1
            probs[0, 1, 7], probs[0, 1, 6] = 0.6, 0.4
            probs[0, 2, 0], probs[0, 2, 6] = 0.99, 0.01
            return torch.log(probs.clamp_min(1e-12)), torch.tensor([3])

    class _Auto:
        def __init__(self):
            self.model = _Inner()

        def generate(self, input, batch_size_s=300):
            self.model.cal_decoder_with_predictor(None)   # 模拟推理内部调用
            return [{"text": "你好吗", "timestamp": [[0, 200], [200, 400], [400, 600]]}]

    asr = ASRModel(device="cpu", model=_Auto())
    out = asr.transcribe("x.wav")
    assert out["confidence_source"] == "posterior"
    assert out["confidence"] == pytest.approx((0.9 + 0.6) / 2, abs=1e-3)
    assert len(out["token_confidences"]) == 2


def test_asr_falls_back_to_proxy_without_hook():
    from src.models.asr_model import ASRModel

    class _Auto:
        def generate(self, input, batch_size_s=300):
            return [{"text": "今天天气很好我很开心", "timestamp": []}]

    out = ASRModel(device="cpu", model=_Auto()).transcribe("x.wav")
    assert out["confidence_source"] == "proxy"
    assert 0.3 <= out["confidence"] <= 1.0


# ---------------- 可信度等级 ----------------
def test_reliability_grade_uses_empirical_ranking(tmp_path):
    """等级按各档实测准确率命名：分歧最大的三分位若准确率最高，则为 high。"""
    p = tmp_path / "unc.json"
    p.write_text(json.dumps({"cuts": [0.10, 0.20], "terciles": {
        "t0": {"accuracy": 0.48, "grade": "low"},
        "t1": {"accuracy": 0.68, "grade": "medium"},
        "t2": {"accuracy": 0.85, "grade": "high"}}}), encoding="utf-8")
    g = reliability.grade({"negative_sd": 0.05, "arousal_sd": 0.02}, path=p)
    assert g["grade"] == "low" and g["accuracy_in_bin"] == 0.48 and g["tercile"] == "t0"
    assert reliability.grade({"negative_sd": 0.05, "arousal_sd": 0.15}, path=p)["grade"] == "medium"
    assert reliability.grade({"negative_sd": 0.3, "arousal_sd": 0.0}, path=p)["grade"] == "high"
    assert reliability.grade({"negative_sd": 0.3}, path=tmp_path / "missing.json") is None
    assert reliability.grade({}, path=p) is None
