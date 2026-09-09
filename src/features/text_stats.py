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

【唤醒度估计】以 0.5 为中性基线，按感叹/疑问标点、程度副词密度、第一人称占比
向上调整，长句平铺直叙向下调整；v0.1 的公式基线为 0.1，在陈述句上几乎恒为低唤醒
（AISHELL-3 中性语音实测 0.17），v0.2 已重新居中。

【维度词典（可选）】若存在 ``resources/dictionaries/cvaw.csv``（Chinese EmoBank /
CVAW，Yu et al. 2016，**仅限学术用途，需用户自行同意其条款下载，仓库不分发**），
则用其 1–9 分的 Valence / Arousal 均值直接估计维度分，并与极性法各半融合。
文件格式：CSV，含表头 ``Word``、``Valence_Mean``、``Arousal_Mean``（CVAW 4.0 原始格式）。

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
    _cvaw: dict[str, tuple[float, float]] | None = None   # word -> (valence 1-9, arousal 1-9)
    _cvaw_loaded: bool = False

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
    def cvaw(cls) -> dict[str, tuple[float, float]]:
        """可选的 CVAW 维度词典；文件不存在时返回空 dict。"""
        if not cls._cvaw_loaded:
            cls._cvaw = cls._load_cvaw(portable.DICTIONARIES_DIR / "cvaw.csv")
            cls._cvaw_loaded = True
        return cls._cvaw or {}

    @staticmethod
    def _load_cvaw(path: Path) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        if not path.exists():
            return out
        import csv
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return out
            cols = {c.strip().lower(): c for c in reader.fieldnames}
            w = cols.get("word")
            v = cols.get("valence_mean")
            a = cols.get("arousal_mean")
            if not (w and v and a):
                return out
            for row in reader:
                try:
                    out[row[w].strip()] = (float(row[v]), float(row[a]))
                except (ValueError, KeyError, AttributeError):
                    continue
        return out

    @classmethod
    def reload(cls) -> None:
        cls._pos = cls._neg = cls._negation = None
        cls._degree = None
        cls._cvaw = None
        cls._cvaw_loaded = False


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

    # ---- 可选：CVAW 维度词典 ----
    cvaw = _DictLoader.cvaw()
    lex_hits = [cvaw[w] for w in words if w in cvaw]
    lex_negative = lex_arousal = None
    if lex_hits:
        v_mean = sum(x[0] for x in lex_hits) / len(lex_hits)
        a_mean = sum(x[1] for x in lex_hits) / len(lex_hits)
        lex_negative = clip01(1.0 - (v_mean - 1.0) / 8.0)
        lex_arousal = clip01((a_mean - 1.0) / 8.0)
        s_text_stat = clip01(0.5 * s_text_stat + 0.5 * lex_negative)

    # ---- 唤醒分估计（以 0.5 为中性基线，上下调整）----
    first_person_count = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)
    first_person_ratio = first_person_count / total_words
    exclam_count = text.count("!") + text.count("！")
    question_count = text.count("?") + text.count("？")
    degree_count = sum(1 for w in words if w in degree_dict)
    sentences = _split_sentences(text)
    avg_len = (sum(len(s) for s in sentences) / len(sentences)) if sentences else len(text)

    base = lex_arousal if lex_arousal is not None else 0.5
    a_text_stat = clip01(
        base
        + min(0.25, 0.12 * exclam_count)                 # 感叹句：激动
        + min(0.12, 0.06 * question_count)               # 疑问句：轻度激活
        + 0.15 * min(1.0, degree_count / total_words * 5)  # 程度副词密度：强调
        + 0.10 * min(1.0, first_person_ratio * 3)        # 自我聚焦：情绪外显
        - (0.08 if avg_len > 18 else 0.0)                # 长句平铺直叙：偏平静
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
        "degree_count": degree_count,
        "cvaw_hits": len(lex_hits),
        "cvaw_negative": lex_negative,
        "cvaw_arousal": lex_arousal,
    }
    return TextStatsResult(s=s_text_stat, a=a_text_stat, detail=detail)
