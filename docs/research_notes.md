# 科研依据与方法学说明

## 1. 理论框架

### 1.1 Russell 情绪环模型

本工具的核心理论依据是 Russell (1980) 的情绪环模型（Circumplex Model of Affect），将情绪映射到二维连续空间：

- **Valence（效价）**：正负向，从极度负面到极度正面。
- **Arousal（唤醒度）**：激活程度，从极度平静到极度激动。

两个维度的组合划分四个象限（Q1-Q4），对应不同情绪状态与差异化干预策略。

### 1.2 多模态情感计算

单一模态的情感识别鲁棒性有限。本工具采用 6 模态加权融合：

1. **声学情感**（emotion2vec）：基于自监督预训练的语音情感表征。
2. **韵律学**（parselmouth）：F0、能量、节奏等超段特征。
3. **副语言事件**（PANNs）：笑声、哭泣、尖叫等非语言事件。
4. **物理声学**（librosa）：响度、频谱质心、粗糙度等物理量。
5. **文本语义**（Qwen3）：转写文本的深层语义情感。
6. **文本统计**（jieba）：基于情感词典的浅层词法分析。

多模态融合提升了对噪声、口音、表述差异的鲁棒性。

## 2. 声刺激设计依据

### 2.1 音乐心理学映射

声刺激参数（基频、脉冲率、响度、频谱质心、谐和结构）的象限锚点基于音乐心理学实证文献（Juslin & Laukka 2004；Bresin & Friberg 2011；Ilie & Thompson 2006）：

- **高唤醒**：更快脉冲率、更大响度、更陡起音、更亮音色（高频谱质心）。
- **低唤醒**：慢脉冲、轻响度、缓起音、暗音色。
- **正面效价**：协和的大三和弦、高基频。
- **负面效价**：整数泛音结构、低基频。

### 2.2 差异化干预策略

不同于统一放松刺激，本工具按情绪象限差异化生成：

- **Q2（焦虑/高唤醒）**：慢脉冲引导呼吸放缓、粉噪背景降低唤醒。
- **Q3（抑郁/低唤醒）**：较快脉冲、明亮协和音色激活唤醒。
- **Q1（兴奋）**：高能量、快节奏、协和。
- **Q4（放松）**：中低脉冲、空灵谐和、可加粉噪。

### 2.3 关于双耳节拍

v1 曾包含双耳节拍（Binaural Beats）作为核心干预。v2 基于 Ingendoh et al. (2023) 的系统综述证据移除该机制——该综述指出双耳节拍诱导特定脑波频段的证据不足。

## 3. 归一化方法

各模态特征通过 z-score 归一化到 [0,1]：

```
z = (value - μ) / σ        # 截断到 [-2, 2]
norm = (z + 2) / 4         # 映射到 [0, 1]
```

参考均值/标准差来自中文普通话语音研究文献（见 `src/fusion/normalizer.py` 的 `PROSODY_STATS`）。

## 4. 动态权重调整

融合权重非固定，根据信号质量动态调整（详见 `src/fusion/weighted_fusion.py`）：

- **低 SNR**：降低声学模态权重，提升文本模态权重。
- **低 ASR 置信度**：降低文本模态权重。
- **强副语言事件**：提升副语言模态权重 ×1.5。
- **极端情况**（SNR<5dB 且 ASR<0.3）：所有模态平均分配。

每次调整后重新归一化，确保权重和恒为 1。

## 5. 局限性说明

- **LLM 文本评分**：1.7B 小模型对精确数值评分能力有限，采用 few-shot 约束输出，但正面文本评分仍可能偏高。实际量化以 6 模态融合为主，LLM 仅作为文本语义支路之一。
- **副语言事件**：AudioSet 标签非专为副语言设计，叹息等事件通过 Breathing 标签近似。
- **无微调**：全部使用预训练模型，未在特定数据上微调，个体差异可能影响精度。
- **科研用途**：本工具用于情感计算方法学探索，不构成临床诊断或治疗手段。

## 6. 参考文献

1. Russell, J.A. (1980). A circumplex model of affect. *JPSP*, 39(6), 1161-1178.
2. Juslin, P.N., & Laukka, P. (2004). Expression, perception, and induction of musical emotions. *JNMR*, 33(3), 217-238.
3. Bresin, R., & Friberg, A. (2011). Emotion rendering in music. *Cortex*, 47(9), 1068-1081.
4. Ilie, G., & Thompson, W.F. (2006). A comparison of acoustic cues in music and speech. *Music Perception*, 23(4), 319-330.
5. Ma, Y., et al. (2024). emotion2vec+: Advancing universal speech emotion representation. *ACL 2024 Findings*.
6. Sethares, W.A. (1993). Local consonance and the relationship between timbre and scale. *JASA*, 94(3), 1218-1228.
7. Ingendoh, R.M., et al. (2023). Binaural beats to entrain the brain? A systematic review. *PLOS ONE*, 18(5):e0286023.
8. Schönwiesner, M., & Bialas, O. (2021). slab: An easy to learn Python package for psychoacoustic experiments. *JOSS*, 6(62), 3284.
