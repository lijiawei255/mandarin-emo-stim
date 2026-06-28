# 科研依据与方法学说明

本文档是项目的**算法原理总参考**，供二次开发者快速理解「为什么这么算」。
代码中各模块的 docstring 有更精炼的对应说明，本文是展开版。

---

## 1. 理论框架：Russell 情绪环模型

本工具的核心理论依据是 Russell (1980) 的情绪环模型（Circumplex Model of Affect），
将情绪映射到二维连续平面：

- **Valence（效价）**：正负向，从极度负面(0)到极度正面(1)。
- **Arousal（唤醒度）**：激活程度，从极度平静(0)到极度激动(1)。

两个维度组合划分四个象限，对应不同情绪簇与差异化干预策略：

| 象限 | Valence | Arousal | 情绪簇 | 干预方向 |
|------|---------|---------|--------|----------|
| Q1 | 高 | 高 | 积极/兴奋（喜悦、狂喜） | 匹配/强化 |
| Q2 | 低 | 高 | 焦虑/紧张（愤怒、恐惧、焦虑） | **降唤醒**（慢脉冲+粉噪） |
| Q3 | 低 | 低 | 低落/抑郁（悲伤、沮丧） | **提唤醒**（明亮协和+快脉冲） |
| Q4 | 高 | 低 | 放松/满足（平静、满足） | 维持/舒缓 |

> **为何选 Russell 模型而非离散情绪分类**：离散分类（如 Ekman 六情）边界硬、
> 难以表达混合情绪；Russell 的连续 V-A 平面可表达任意情绪状态，且便于映射到
> 连续声学参数（避免离散切换的突兀感）。

---

## 2. 多模态情感计算：6 模态加权融合

### 2.1 为何多模态

单一模态鲁棒性有限：声学受环境噪声/口音干扰，文本受 ASR 错误影响。本项目借鉴
多模态情感计算（Multimodal Sentiment Analysis）的可靠性原则——**声学 + 文本双通道
互补**，并按信号质量动态调整各模态信任度。

### 2.2 六模态

| 模态 | 模型/方法 | 捕捉的情感线索 | 默认权重(negative/arousal) |
|------|----------|---------------|--------------------------|
| acoustic | emotion2vec_plus_large | 深层声学情感表征（语调/音色综合） | 0.30 / 0.35 |
| prosody | praat-parselmouth | F0/节奏/HNR/Jitter/Shimmer | 0.15 / 0.25 |
| paralang | PANNs CNN10 | 笑/哭/尖叫等副语言事件 | 0.10 / 0.15 |
| physical | librosa/scipy | 响度/频谱质心/粗糙度 | 0.05 / 0.10 |
| text_llm | Qwen3-1.7B | 语义层负面情绪 | 0.30 / 0.10 |
| text_stat | jieba+词典 | 词法层情感极性 | 0.10 / 0.05 |

**权重设计依据**：负面分(negative)中文本权重高（语义直接表达负面情绪），
唤醒分(arousal)中声学权重高（唤醒主要由声学能量/节奏/语速体现）。

### 2.3 动态权重调整（鲁棒性核心）

固定权重在信号质量变化时会失效。本项目按信号质量**自适应**调整（见
`src/fusion/weighted_fusion.py`）：

1. **低 SNR（音频噪声大）**：声学模态受污染不可信 → 衰减声学权重，转移给文本模态。
2. **低 ASR 置信度（转写不可靠）**：文本模态不可信 → 反向转移回声学。
3. **极端情况（SNR<5dB 且 ASR<0.3）**：两通道都不可信 → 所有模态平均(各 1/6)。
4. **强副语言事件（如尖叫 conf>0.8）**：强情感信号 → 副语言权重 ×1.5 放大。

每次调整后**重新归一化**，确保权重和恒为 1。这一机制使整体估计对噪声/口音/
ASR 错误更鲁棒——这是相比固定权重融合的关键改进。

---

## 3. 各模态算法原理

### 3.1 emotion2vec：离散情绪 → 连续 V-A（`src/models/emotion_model.py`）

emotion2vec 基于数据2vec 自监督预训练，从原始波形提取情感相关深层表征，输出 9 类
离散情绪的 softmax 置信度。

**关键步骤**：把 9 类离散情绪投影到 Russell V-A 平面。依据心理学文献中各类情绪的
实证锚点（`config/emotion_mapping.json`）：

| 情绪 | n_base(负面) | a_base(唤醒) | 说明 |
|------|-------------|-------------|------|
| angry | 0.90 | 0.90 | 高负高唤 |
| fearful | 0.85 | 0.95 | 高负极高唤 |
| disgusted | 0.80 | 0.70 | 高负中唤 |
| sad | 0.80 | 0.20 | 高负**低**唤（与愤怒区分） |
| happy | 0.10 | 0.75 | 低负(正)高唤 |
| surprised | 0.55 | 0.90 | 中中高唤 |
| neutral | 0.50 | 0.40 | 中中 |

**聚合**：以各情绪置信度为权重做加权平均（等价于期望值），比 argmax 单类更平滑、
对模糊边界更鲁棒。

**unknown 收缩**：当 unknown(模型不确定)置信度 > 0.5，结果向中性(0.5)收缩 50%，
降低不确定样本的贡献。

### 3.2 韵律学（`src/features/prosody.py`）

韵律（prosody）是语音超段层面的声学特征，承载大量情感信息（Juslin & Laukka 2004）：

- **F0（基频）均值/范围**：高唤醒情绪 F0 更高、范围更大；低落情绪 F0 低且平。
- **语速 speech_rate**：焦虑/紧张常加快，抑郁常减慢。
- **停顿占比 pause_ratio**：犹豫、抑郁时静音段增多。
- **HNR（谐波噪声比）**：反映嗓音「干净度」。HNR 低=气声/粗糙（悲伤、紧张、压抑）。
  故负面分用 1/HNR（越低越负面）。
- **Jitter（基频微扰）**：相邻周期 F0 的微小波动。>3% 为病理/强情绪。
- **Shimmer（振幅微扰）**：相邻周期振幅波动。情绪激动/疲惫时升高。

子权重聚合公式见模块 docstring，参考统计量见 `src/fusion/normalizer.py`。

### 3.3 副语言事件（`src/models/pann_model.py`）

PANNs（Pretrained Audio Neural Networks）CNN10 在 AudioSet 上训练，能识别 527 类
声学事件。本项目关注其中与情绪强相关的副语言事件：

| 事件 | n_contrib | a_contrib | 含义 |
|------|-----------|-----------|------|
| 笑声 Laughter | -0.5 | +0.3 | 正面 |
| 哭泣 Crying | +0.7 | +0.5 | 高负中唤 |
| 尖叫 Screaming | +0.3 | +1.0 | 极端唤醒 |
| 叹息/呼吸 Sigh | +0.4 | -0.3 | 压抑/放松 |

聚合按文档公式：以置信度加权累加 n_contrib/a_contrib，再除以 (1+总置信度) 归一化。
> AudioSet 无独立「Sigh」标签，通过 Breathing 标签近似。

### 3.4 物理声学（`src/features/physical.py`）

底层物理特征（Ilie & Thompson 2006）：

- **响度 RMS**、**频谱质心**：见上表。
- **高频能量比**：过高(刺耳)/过低(沉闷)都偏负面，故用 |norm-0.5|*2。
- **SNR**：信号帧 vs 噪声帧功率比；稳态信号(CV<0.05)视为高 SNR。
- **频谱粗糙度 roughness**：基于 Sethares (1993) 与 Plomp-Levelt 不协和模型——
  相邻频率分量在 20-150Hz 拍频内产生「粗糙」不协和感（人耳对 ~70Hz 拍频最敏感）。
  取频谱显著峰对，按拍频的高斯权重加权求和。

### 3.5 文本语义（`src/models/llm_model.py`）

Qwen3-1.7B 经 few-shot prompt 输出两个 0-1 浮点数（负面分、唤醒度）。
> 1.7B 小模型对精确数值评分能力有限，故用 few-shot 示例约束输出量纲。实际情感
> 量化以 6 模态融合为主，LLM 仅作文本语义支路之一。解析失败重试一次，二次失败
> 降级为文本统计分。

### 3.6 文本统计（`src/features/text_stats.py`）

情感词典加权法（Lexicon-based）：分词后查词典判极性，加权求和。
关键处理：程度副词修饰（前 2 词窗口）、否定词反转（前 3 词窗口，奇数反转、
偶数双重否定表肯定，反转×0.8 衰减）。

---

## 4. 声刺激生成：音乐心理学映射（`src/stimulus/`）

### 4.1 设计目标

根据检测到的情绪，生成**差异化**声刺激——非统一放松音，而是针对不同象限给出
有理论依据的声学干预。

### 4.2 声学参数 → 情绪映射

依据 Juslin & Laukka 2004、Bresin & Friberg 2011、Ilie & Thompson 2006：

| 参数 | 高唤醒 | 低唤醒 | 正面 | 负面 |
|------|--------|--------|------|------|
| 脉冲率 pr | 快（激活） | 慢（平复） | — | — |
| 基频 f0 | — | — | 高（明亮） | 低（深沉） |
| 响度 | 大 | 小 | — | — |
| 频谱质心 | 高（亮） | 低（暗） | — | — |
| 起音 attack | 陡（冲击） | 缓（柔和） | — | — |
| 谐和结构 | — | — | 大三和弦（协和） | 整数泛音（紧张） |

**反直觉的干预分支**（差异化设计的精髓）：
- **Q2 焦虑**：高 arousal 反而用**慢脉冲**（引导呼吸放缓，降唤醒干预）。
- **Q3 抑郁**：低 valence 反而**提 f0**（注入明亮感/能量，激活干预）。

### 4.3 软混合

按四象限隶属度对锚点参数加权混合（避免象限边界突变），再在主象限内按
valence/arousal 连续微调（见 `strategies.py::compute_params`）。

### 4.4 波形合成（`synthesizer.py`）

谐和音叠加 → 带通塑形 → 振幅调制(AM) → ADSR 包络 → 粉噪混合 →
安全限幅(-10dBFS) → 淡入淡出 → Haas 立体声（右声道延迟 12ms 产生自然宽度，
非可闻回声）。

### 4.5 关于双耳节拍

v1 曾含双耳节拍（Binaural Beats）。v2 基于 Ingendoh et al. (2023) 系统综述移除——
该综述指出双耳节拍诱导特定脑波频段的证据不足。

### 4.6 声音安全

峰值限幅 -10 dBFS（≈70-75 dB SPL，正常交谈音量），无听力损伤风险。

---

## 5. 归一化方法（`src/fusion/normalizer.py`）

z-score 归一化：以「该特征在正常语音中的分布」为参照。

```
z = (value - mu) / sigma        # 偏离均值多少个标准差
z 截断到 [-2, 2]                 # 抑制极端离群点
norm = (z + 2) / 4              # 映射到 [0, 1]
```

参考 mu/sigma 来自中文普通话语料统计（男女混合）。偏离基准的程度即情绪强度。
> z-score 假设近似正态，对偏态特征（如 Jitter）是近似，工程上足够鲁棒。

---

## 6. 局限性（二次开发须知）

- **LLM 文本评分**：1.7B 小模型精度有限，正面文本评分可能偏高。可替换更大 LLM
  （需调整显存预算）。
- **副语言事件**：AudioSet 非专为副语言设计，叹息等通过 Breathing 近似。
- **无微调**：全部预训练模型，未在特定数据微调，个体差异可能影响精度。
- **单语**：ASR/LLM/词典均针对中文普通话。
- **科研用途**：本工具用于方法学探索，**不构成临床诊断或治疗手段**。

---

## 7. 参考文献

1. Russell, J.A. (1980). A circumplex model of affect. *JPSP*, 39(6), 1161-1178.
2. Juslin, P.N., & Laukka, P. (2004). Expression, perception, and induction of musical emotions. *JNMR*, 33(3), 217-238.
3. Bresin, R., & Friberg, A. (2011). Emotion rendering in music. *Cortex*, 47(9), 1068-1081.
4. Ilie, G., & Thompson, W.F. (2006). A comparison of acoustic cues in music and speech. *Music Perception*, 23(4), 319-330.
5. Ma, Y., et al. (2024). emotion2vec+: Advancing universal speech emotion representation. *ACL 2024 Findings*.
6. Sethares, W.A. (1993). Local consonance and the relationship between timbre and scale. *JASA*, 94(3), 1218-1228.
7. Plomp, R., & Levelt, W.J.M. (1965). Tonal consonance and critical bandwidth. *JASA*, 38(4), 548-560.
8. Ingendoh, R.M., et al. (2023). Binaural beats to entrain the brain? A systematic review. *PLOS ONE*, 18(5):e0286023.
9. Soderlund, G., et al. (2007). Listen to the noise: Noise is beneficial for cognitive performance in ADHD. *J Child Psychol Psychiatry*, 48(8), 840-847.
10. Schönwiesner, M., & Bialas, O. (2021). slab: An easy to learn Python package for psychoacoustic experiments. *JOSS*, 6(62), 3284.
