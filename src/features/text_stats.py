"""文本语言学统计特征（jieba 分词 + 情感词典加权）。

【原理】基于情感词典的浅层情感分析方法（Lexicon-based Sentiment Analysis）：
对文本分词后，查情感词典判断每个词的极性（正面/负面），加权求和得整体倾向。
这是 LLM 语义分析之外的「轻量、可解释、确定性」文本支路，二者互补。

【关键语言学处理】
  - 程度副词修饰：前 2 词窗口内的程度副词（如「非常」×1.5、「极其」×2.0、
    「稍微」×0.6）放大或衰减情感强度。
  - 否定词反转：前 3 词窗口内奇数个否定词（如「不」「没」）会反转极性
    （「不开心」→负面），且反转略衰减（×0.8），模拟否定表达的弱化语气。
    偶数个否定则双重否定表肯定（「不是不开心」→正面）。
  - 这些窗口规则是中文情感分析的标准做法（参考 BosonNLP/知网词典方法）。

【唤醒度估计】综合标点（感叹/疑问句 arousal 高）、第一人称代词占比
（自我聚焦→情绪外显）、短句比例（急促→高唤醒）等浅层线索。

【词表】``resources/dictionaries/`` 为本项目自维护的常见词起始词典（不受版权限制），
用户可自行扩展。加载后缓存（见 _DictLoader）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jieba

from src import portable
from src.fusion.normalizer import clip01

# 第一人称代词（用于唤醒度估计）
FIRST_PERSON_PRONOUNS = frozenset({"我", "俺", "咱", "本人", "鄙人", "老子", "人家"})


@dataclass
class TextStatsResult:
    s: float
    a: float
    detail: dict[str, Any] = field(default_factory=dict)


class _DictLoader:
    """懒加载并缓存情感词表。"""

    _pos: set[str] | None = None
    _neg: set[str] | None = None
    _negation: set[str] | None = None
    _degree: dict[str, float] | None = None

    @classmethod
    def _load_words(cls, path: Path) -> set[str]:
        words: set[str] = set()
        if not path.exists():
            return words
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line)
        return words

    @classmethod
    def _load_degree(cls, path: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    continue
        return out

    @classmethod
    def positive(cls) -> set[str]:
        if cls._pos is None:
            cls._pos = cls._load_words(portable.DICTIONARIES_DIR / "positive_words.txt")
        return cls._pos

    @classmethod
    def negative(cls) -> set[str]:
        if cls._neg is None:
            cls._neg = cls._load_words(portable.DICTIONARIES_DIR / "negative_words.txt")
        return cls._neg

    @classmethod
    def negation(cls) -> set[str]:
        if cls._negation is None:
            cls._negation = cls._load_words(portable.DICTIONARIES_DIR / "negation_words.txt")
        return cls._negation

    @classmethod
    def degree(cls) -> dict[str, float]:
        if cls._degree is None:
            cls._degree = cls._load_degree(portable.DICTIONARIES_DIR / "degree_adverbs.txt")
        return cls._degree

    @classmethod
    def reload(cls) -> None:
        cls._pos = cls._neg = cls._negation = None
        cls._degree = None


def _split_sentences(text: str) -> list[str]:
    """按标点切句。"""
    import re
    parts = re.split(r"[。！？!？.；;\n]+", text)
    return [p for p in parts if p.strip()]


def analyze(text: str) -> TextStatsResult:
    """分析文本，返回统计情感分。

    Args:
        text: ASR 转写文本（中文）。

    Returns:
        :class:`TextStatsResult`，含 (负面分 s, 唤醒分 a, 详情)。
    """
    if not text or not text.strip():
        return TextStatsResult(s=0.5, a=0.5, detail={"reason": "empty_text"})

    words = list(jieba.lcut(text))
    words = [w for w in words if w.strip()]
    total_words = len(words)
    if total_words == 0:
        return TextStatsResult(s=0.5, a=0.5, detail={"reason": "no_words"})

    pos_set = _DictLoader.positive()
    neg_set = _DictLoader.negative()
    neg_set_words = _DictLoader.negation()
    degree_dict = _DictLoader.degree()

    score_val = 0.0
    pos_count = 0
    neg_count = 0
    for i, word in enumerate(words):
        is_pos = word in pos_set
        is_neg = word in neg_set
        if not is_pos and not is_neg:
            continue

        weight = 1.0
        # 程度副词（前 2 个词窗口）
        for j in range(max(0, i - 2), i):
            if words[j] in degree_dict:
                weight *= degree_dict[words[j]]
        # 否定词（前 3 个词窗口，奇数个反转）
        neg_count_here = sum(1 for j in range(max(0, i - 3), i) if words[j] in neg_set_words)
        if neg_count_here % 2 == 1:
            weight = -weight * 0.8  # 否定反转略衰减

        if is_pos:
            score_val += weight
            pos_count += 1
        else:
            score_val -= weight
            neg_count += 1

    # 归一化到 [0,1]：s_text_stat = 0.5 - score/(total*2)
    s_text_stat = clip01(0.5 - score_val / (total_words * 2))

    # ---- 唤醒分估计 ----
    first_person_count = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)
    first_person_ratio = first_person_count / total_words
    exclam_count = text.count("!") + text.count("！")
    question_count = text.count("?") + text.count("？")
    sentences = _split_sentences(text)
    if sentences:
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        short_sentence_ratio = 1.0 if avg_len < 8 else 0.0
    else:
        short_sentence_ratio = 0.0

    a_text_stat = clip01(
        0.3 * (exclam_count * 0.3 + question_count * 0.15) / 3.0
        + 0.3 * min(1.0, first_person_ratio * 3)
        + 0.2 * short_sentence_ratio
        + 0.2 * 0.5
    )

    detail = {
        "total_words": total_words,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "score_raw": score_val,
        "s_text_stat": s_text_stat,
        "a_text_stat": a_text_stat,
        "exclam_count": exclam_count,
        "question_count": question_count,
        "first_person_ratio": first_person_ratio,
    }
    return TextStatsResult(s=s_text_stat, a=a_text_stat, detail=detail)
