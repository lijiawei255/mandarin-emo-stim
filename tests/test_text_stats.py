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


def test_plain_statement_arousal_is_centered():
    """平铺直叙的陈述句唤醒分应接近 0.5（v0.1 恒为 ~0.1–0.2）。"""
    r = text_stats.analyze("这个城市的地铁一共有十二条线路")
    assert 0.35 <= r.a <= 0.65


def test_cvaw_lexicon_used_when_present(tmp_path, monkeypatch):
    """存在 cvaw.csv 时用维度词典估计 V-A，并写入 detail。"""
    from src import portable
    monkeypatch.setattr(portable, "DICTIONARIES_DIR", tmp_path)
    (tmp_path / "positive_words.txt").write_text("开心\n", encoding="utf-8")
    (tmp_path / "negative_words.txt").write_text("痛苦\n", encoding="utf-8")
    (tmp_path / "negation_words.txt").write_text("不\n", encoding="utf-8")
    (tmp_path / "degree_adverbs.txt").write_text("非常\t1.5\n", encoding="utf-8")
    (tmp_path / "cvaw.csv").write_text(
        "No.,Word,Valence_Mean,Valence_SD,Arousal_Mean,Arousal_SD\n"
        "1,开心,8.0,0.5,7.0,0.5\n2,痛苦,1.5,0.5,6.5,0.5\n", encoding="utf-8")
    text_stats._DictLoader.reload()
    r = text_stats.analyze("我很开心")
    assert r.detail["cvaw_hits"] == 1
    assert r.detail["cvaw_negative"] == pytest.approx(1 - 7 / 8)
    assert r.s < 0.4
    r2 = text_stats.analyze("我很痛苦")
    assert r2.s > 0.6 and r2.a > r.a - 0.2


def test_detail_populated():
    r = text_stats.analyze("我非常喜欢这个产品")
    assert "total_words" in r.detail
    assert "pos_count" in r.detail
    assert r.detail["total_words"] > 0
