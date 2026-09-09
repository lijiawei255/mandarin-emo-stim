"""融合模块测试。"""

import pytest

from src.config_loader import load_settings
from src.fusion.normalizer import clip01, zscore_normalize
from src.fusion.quadrant import (
    compute_quadrant_memberships,
    dominant_quadrant,
)
from src.fusion.weighted_fusion import WeightedFusion


@pytest.fixture
def fusion():
    return WeightedFusion(load_settings())


def _all_half_scores():
    """6 模态全为 (0.5, 0.5) 的中性输入。"""
    from src.fusion.weighted_fusion import MODALITIES
    return {m: (0.5, 0.5) for m in MODALITIES}


# ---------------- normalizer ----------------
def test_zscore_normalize_endpoints():
    # value = mu -> 0.5
    assert abs(zscore_normalize(180.0, 180.0, 50.0) - 0.5) < 1e-9
    # value 很大（>= mu+2sigma）-> 1.0
    assert abs(zscore_normalize(280.0, 180.0, 50.0) - 1.0) < 1e-9
    # value 很小（<= mu-2sigma）-> 0.0
    assert abs(zscore_normalize(80.0, 180.0, 50.0) - 0.0) < 1e-9
    # 结果必在 [0,1]
    for v in (-1000, -10, 0, 50, 100, 500, 9999):
        assert 0.0 <= zscore_normalize(v, 180.0, 50.0) <= 1.0


def test_zscore_normalize_zero_sigma():
    assert zscore_normalize(1.0, 1.0, 0.0) == 0.5


def test_clip01():
    assert clip01(-0.5) == 0.0
    assert clip01(1.5) == 1.0
    assert clip01(0.3) == 0.3


# ---------------- quadrant ----------------
def test_quadrant_memberships_sum_to_one():
    for v in (0.1, 0.3, 0.5, 0.7, 0.9):
        for a in (0.1, 0.3, 0.5, 0.7, 0.9):
            m = compute_quadrant_memberships(v, a)
            assert abs(sum(m.values()) - 1.0) < 1e-9


def test_quadrant_dominant_correct():
    assert dominant_quadrant(compute_quadrant_memberships(0.9, 0.9)) == "Q1"
    assert dominant_quadrant(compute_quadrant_memberships(0.1, 0.9)) == "Q2"
    assert dominant_quadrant(compute_quadrant_memberships(0.1, 0.1)) == "Q3"
    assert dominant_quadrant(compute_quadrant_memberships(0.9, 0.1)) == "Q4"


# ---------------- fusion ----------------
def test_fusion_output_ranges(fusion):
    result = fusion.fuse(_all_half_scores())
    for key in ("negative", "valence", "arousal"):
        assert 0.0 <= result[key] <= 1.0
    assert abs(result["valence"] - (1 - result["negative"])) < 1e-9


def test_fusion_weights_normalized(fusion):
    result = fusion.fuse(_all_half_scores())
    for axis in ("negative", "arousal"):
        w = result["weights"][axis]
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_fusion_negative_high_when_all_negative(fusion):
    """全部模态为高负面 -> negative 接近 1。"""
    from src.fusion.weighted_fusion import MODALITIES
    scores = {m: (0.95, 0.5) for m in MODALITIES}
    result = fusion.fuse(scores, asr_confidence=0.9)
    assert result["negative"] > 0.9


def test_fusion_valence_high_when_all_positive(fusion):
    """全部模态为低负面（正面）-> negative 接近 0，valence 接近 1。"""
    from src.fusion.weighted_fusion import MODALITIES
    scores = {m: (0.05, 0.5) for m in MODALITIES}
    result = fusion.fuse(scores, asr_confidence=0.9)
    assert result["negative"] < 0.1
    assert result["valence"] > 0.9


def test_fusion_low_snr_shifts_to_text(fusion):
    """低 SNR 时，文本模态权重应高于默认（声学被压低）。"""
    default = fusion.fuse(_all_half_scores())["weights"]["negative"]
    low_snr = fusion.fuse(_all_half_scores(), audio_quality={"snr_db": 3.0})["weights"]["negative"]
    assert low_snr["text_llm"] > default["text_llm"]
    assert low_snr["acoustic"] < default["acoustic"]


def test_fusion_low_asr_shifts_to_acoustic(fusion):
    """低 ASR 置信度时，声学权重应高于默认（文本被压低）。"""
    default = fusion.fuse(_all_half_scores())["weights"]["negative"]
    low_asr = fusion.fuse(_all_half_scores(), asr_confidence=0.2)["weights"]["negative"]
    assert low_asr["acoustic"] > default["acoustic"]


def test_fusion_extreme_case_uniform(fusion):
    """SNR<5dB 且 ASR<0.3 时，所有模态权重平均分配（各 1/6）。"""
    result = fusion.fuse(
        _all_half_scores(),
        audio_quality={"snr_db": 3.0},
        asr_confidence=0.2,
    )
    w = result["weights"]["negative"]
    for m in ("acoustic", "text_llm"):
        assert abs(w[m] - 1.0 / 6) < 1e-6


def test_fusion_paralang_boost(fusion):
    """强副语言事件（confidence>0.8）提升 paralang 权重。"""
    default = fusion.fuse(_all_half_scores())["weights"]["negative"]["paralang"]
    boosted = fusion.fuse(
        _all_half_scores(),
        paralang_events=[{"event": "screaming", "confidence": 0.9}],
    )["weights"]["negative"]["paralang"]
    assert boosted > default


def test_fusion_modal_scores_present(fusion):
    result = fusion.fuse(_all_half_scores())
    assert "modal_scores" in result
    assert len(result["modal_scores"]) == 6


# ---------------- 中性校准偏移 ----------------
def test_modality_calibration_applied_and_clamped(tmp_path):
    import json

    from src.fusion.weighted_fusion import WeightedFusion
    cfg = json.loads(json.dumps(load_settings()))
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"offsets": {
        "text_llm": {"negative": -0.15, "arousal": 0.0},
        "physical": {"negative": -0.9, "arousal": 0.9},   # 超范围 → 钳制到 ±0.3
    }}), encoding="utf-8")
    cfg["fusion_calibration"] = {"enabled": True, "path": str(p)}
    fus = WeightedFusion(cfg)
    assert fus.offsets["text_llm"] == (-0.15, 0.0)
    assert fus.offsets["physical"] == (-0.3, 0.3)
    r = fus.fuse(_all_half_scores())
    assert r["modal_scores_raw"]["text_llm"]["negative"] == 0.5
    assert r["modal_scores"]["text_llm"]["negative"] == pytest.approx(0.35)
    # 禁用后不偏移
    cfg["fusion_calibration"]["enabled"] = False
    assert WeightedFusion(cfg).offsets == {}


def test_modality_calibration_missing_file_is_noop(tmp_path):
    import json

    from src.fusion.weighted_fusion import WeightedFusion
    cfg = json.loads(json.dumps(load_settings()))
    cfg["fusion_calibration"] = {"enabled": True, "path": str(tmp_path / "nope.json")}
    assert WeightedFusion(cfg).offsets == {}


# ---------------- prosody norms 加载 ----------------
def test_load_prosody_stats_prefers_file_and_falls_back(tmp_path):
    import json

    from src.fusion.normalizer import LEGACY_PROSODY_STATS, load_prosody_stats
    p = tmp_path / "prosody_norms.json"
    p.write_text(json.dumps({
        "mixed": {"mean_f0": {"mu": 201.5, "sigma": 44.0, "n": 600},
                  "hnr": {"mu": 12.0, "sigma": 0.0, "n": 600}},   # sigma=0 → 回退
        "female": {"mean_f0": {"mu": 230.0, "sigma": 30.0, "n": 300}},
    }), encoding="utf-8")
    s = load_prosody_stats(p)
    assert s["mean_f0"] == (201.5, 44.0)
    assert s["hnr"] == LEGACY_PROSODY_STATS["hnr"]          # sigma 非正回退
    assert s["jitter_local"] == LEGACY_PROSODY_STATS["jitter_local"]  # 缺失回退
    assert load_prosody_stats(p, group="female")["mean_f0"] == (230.0, 30.0)
    assert load_prosody_stats(tmp_path / "missing.json") == LEGACY_PROSODY_STATS
