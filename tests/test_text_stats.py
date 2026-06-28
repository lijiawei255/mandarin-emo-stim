"""文本统计特征测试。"""

import pytest

from src.features import text_stats


@pytest.fixture(autouse=True)
def _reload_dicts():
    text_stats._DictLoader.reload()
    yield
    text_stats._DictLoader.reload()


def test_empty_text_neutral():
    r = text_stats.analyze("")
    assert r.s == 0.5
    assert r.a == 0.5


def test_positive_text_low_negative():
    """正面文本 -> 负面分偏低（< 0.5）。"""
    r = text_stats.analyze("我今天非常开心，太棒了！")
    assert r.s < 0.5
    assert r.detail["pos_count"] > 0


def test_negative_text_high_negative():
    """负面文本 -> 负面分偏高（> 0.5）。"""
    r = text_stats.analyze("我非常痛苦，太糟糕了，我好难过")
    assert r.s > 0.5
    assert r.detail["neg_count"] > 0


def test_negation_flips_polarity():
    """否定词反转情感极性：'不开心' 应比 '开心' 更负面。"""
    r_pos = text_stats.analyze("开心")
    r_neg = text_stats.analyze("不开心")
    assert r_neg.s > r_pos.s


def test_degree_adverb_amplifies():
    """程度副词放大原始情感强度：'非常好' 的原始得分应高于 '好'。"""
    r_plain = text_stats.analyze("好")
    r_strong = text_stats.analyze("非常好")
    assert r_strong.detail["score_raw"] > r_plain.detail["score_raw"]


def test_scores_in_range():
    for text in ["好", "坏", "今天天气不错", "我很累", "中立的一句话"]:
        r = text_stats.analyze(text)
        assert 0.0 <= r.s <= 1.0
        assert 0.0 <= r.a <= 1.0


def test_exclamation_raises_arousal():
    """感叹号提高唤醒度。"""
    r_no_excl = text_stats.analyze("今天天气真好")
    r_excl = text_stats.analyze("今天天气真好！")
    assert r_excl.a >= r_no_excl.a


def test_detail_populated():
    r = text_stats.analyze("我非常喜欢这个产品")
    assert "total_words" in r.detail
    assert "pos_count" in r.detail
    assert r.detail["total_words"] > 0
