# 科研依据与方法学说明

本文档是项目的**算法原理总参考**，供二次开发者理解「为什么这么算」以及
「哪些是有文献支撑的、哪些是本项目的工程启发式」。代码中各模块的 docstring
有精炼的对应说明，本文是展开版。

> **阅读约定**：每处方法学都标注证据等级——
> **[文献]** 有具体出处支持；**[启发式]** 本项目设定、方向与文献一致但数值
> 未经实证；**[实测]** 由本仓库 `scripts/evaluate.py` 在公开语料上测得。
> 完整的实测结果与局限性见 [evaluation.md](./evaluation.md)。

---

## 1. 理论框架：Russell 情绪环模型 [文献]

核心理论依据是 Russell (1980) 的情绪环模型（Circumplex Model of Affect），
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

### 2.1 为何多模态 [文献]

单一模态鲁棒性有限：声学受环境噪声/口音干扰，文本受 ASR 错误影响。多模态
情感计算的综述（Poria et al. 2017）系统总结了「声学 + 文本」互补融合优于单
模态的证据。本项目采用晚期融合（各模态独立打分再加权）。

### 2.2 六模态

| 模态 | 模型/方法 | 捕捉的情感线索 | 默认权重(negative/arousal) |
|------|----------|---------------|--------------------------|
| acoustic | emotion2vec_plus_large | 深层声学情感表征（语调/音色综合） | 0.30 / 0.35 |
| prosody | praat-parselmouth | F0/F0 斜率/语速/HNR/Jitter/Shimmer | 0.15 / 0.25 |
| paralang | PANNs CNN10 | 笑/哭/尖叫等副语言事件 | 0.10 / 0.15 |
| physical | librosa/scipy | 响度/频谱质心/粗糙度 | 0.05 / 0.10 |
| text_llm | Qwen3-1.7B | 语义层负面情绪 | 0.30 / 0.10 |
| text_stat | jieba+词典 | 词法层情感极性 | 0.10 / 0.05 |

**权重来源 [启发式]**：负面分中文本权重高（语义直接表达负面），唤醒分中
声学权重高（唤醒主要由声学能量/节奏/语速体现，Juslin & Laukka 2003 的元分析
支持这一方向）。**具体数值由本项目设定，未经数据学习**；各模态单独的判别力
与融合增益见 evaluation.md 的消融实验。

### 2.3 动态权重调整（鲁棒性设计）[启发式]

固定权重在信号质量变化时会失效。本项目按信号质量自适应调整（见
`src/fusion/weighted_fusion.py`）：

1. **低 SNR（音频噪声大）**：声学模态受污染不可信 → 衰减声学权重，转移给文本模态。
2. **低 ASR 置信度（转写不可靠）**：文本模态不可信 → 反向转移回声学。
3. **极端情况（SNR<5dB 且 ASR<0.3）**：两通道都不可信 → 所有模态平均(各 1/6)。
4. **强副语言事件（如尖叫 conf>0.8）**：强情感信号 → 副语言权重 ×1.5 放大。

每次调整后重新归一化，确保权重和恒为 1。

**须知的薄弱环节**：规则 2 依赖的「ASR 置信度」**不是模型输出的后验概率**，
而是由转写文本长度与重复率估计的**代理指标**（`ASRModel._estimate_confidence`）。
FunASR 1.0.25 的 `Paraformer.inference` 内部计算了 token 级 `am_scores` 但不在
返回结果中暴露，故无法低成本获得真实置信度。该代理指标只能捕捉「转写为空/
极短/大量重复」这类粗粒度失败，对语义级错误无感知。

### 2.4 中性校准（基线归一化）[实测]（v0.2）

各模态的绝对分在情绪中性语音上并不落在 0.5：v0.1 在 AISHELL-3 上实测 text_llm 负面
0.66、physical 负面 0.60 / 唤醒 0.34、text_stat 唤醒 0.17，导致 62% 的中性语音被判入
Q3。v0.2 起融合前对每个模态加常数偏移 offset = 0.5 − 中性均值（`config/modality_calibration.json`，
由 `scripts/evaluate.py neutral` 在 AISHELL-3 校准集上实测，钳制 ±0.3）。这是情感计算
中常规的说话人/语料基线归一化思路（Schuller et al. 2011 的 speaker normalization 即同类做法）；
它只平移均值，不改变各模态内部的相对排序。可在 `settings.json` 的 `fusion_calibration.enabled`
关闭；结果同时返回校准前后的模态分（`modal_scores_raw` / `modal_scores`）。

**个人基线（v0.3）[实测]**：语料级偏移只消除模型对「一般中文语音」的偏置，个人嗓音
（F0、气声、语速习惯）会让各模态在**个人**中性状态下再次偏离 0.5。用户录一段平静朗读
（默认 30 s）得到个人偏移（`src/fusion/personal_calibration.py`，保存在
`portable_data/calibration/user_baseline.json`），融合时替代语料偏移。evaluation.md §0
用 CSEMOTIONS 每位配音员的中性句模拟了这一做法（说话人级基线）。

### 2.5 可学习融合（v0.3，默认不启用）[实测]

`src/fusion/learned_fusion.py`：岭回归把 12 维原始模态分线性映射到 (negative, arousal)，
系数保存在 `config/learned_fusion.json`，每个模态的贡献可直接阅读。训练目标是各类别在
Russell 环上的参照坐标（连续标签的粗略替代），因此只能保证类别中心的相对位置。系数由
CSEMOTIONS（表演型）交叉验证与全量拟合得到，迁移到自然语音未验证；设 `settings.json`
的 `fusion_mode="learned"` 启用。

### 2.6 不确定性（v0.3）

融合结果附带活跃模态（未降级）校准后分数的加权标准差 `uncertainty.{negative_sd, arousal_sd}`。
六个模态分歧越大，绝对判断越不可靠；GUI 在象限标题后显示「模态分歧 ±」。这不是统计意义的
置信区间，只是分歧度的直观量。

### 2.7 降级行为

任一模态异常（模型推理失败、标签表缺失等）时，该模态以中性分 (0.5, 0.5)
参与融合，并在结果的 `degraded_modalities` 中列出；GUI 与 CLI 都会如实展示。
仅「加载音频」与「融合」失败会中止分析。

---

## 3. 各模态算法原理

### 3.1 emotion2vec：离散情绪 → 连续 V-A（`src/models/emotion_model.py`）

emotion2vec+ [文献: Ma et al. 2024] 基于 data2vec 自监督预训练，从原始波形
提取情感相关深层表征，输出 9 类离散情绪的 softmax 置信度。

**关键步骤**：把 9 类离散情绪投影到 Russell V-A 平面（`config/emotion_mapping.json`
的 n_base / a_base）：

| 情绪 | n_base(负面) | a_base(唤醒) | 说明 |
|------|-------------|-------------|------|
| angry | 0.90 | 0.90 | 高负高唤 |
| fearful | 0.85 | 0.95 | 高负极高唤 |
| disgusted | 0.80 | 0.70 | 高负中唤 |
| sad | 0.80 | 0.20 | 高负**低**唤（与愤怒区分） |
| happy | 0.10 | 0.75 | 低负(正)高唤 |
| surprised | 0.55 | 0.90 | 效价中性偏负、高唤 |
| neutral | 0.50 | 0.40 | 中中 |

**锚点来源 [启发式，方向有文献]**：各类情绪在 V-A 平面的**相对位置**（愤怒/
恐惧高唤醒负效价、悲伤低唤醒负效价、快乐高唤醒正效价）取自 Russell (1980)
图 2 的环形排列，并与 Warriner et al. (2013) 的 13,915 个英文词 V-A 规范中
对应情绪词的评分方向一致。**具体数值（如 0.90 / 0.95）是本项目为拉开象限
距离而设定的，不是任何实验的测量值。** 值得注意的争议：surprise 的效价在
文献中并不稳定（可正可负），本项目取 0.55 略偏负，评测时对 surprise 单独
报告、不计入严格象限准确率。

**聚合**：以各情绪置信度为权重做加权平均（等价于期望值），比 argmax 单类更
平滑、对模糊边界更鲁棒。**unknown 收缩**：unknown 置信度 > 0.5 时结果向
中性(0.5)收缩 50%。

### 3.2 韵律学（`src/features/prosody.py`）

韵律（prosody）承载大量情感信息 [文献: Juslin & Laukka 2003 对 104 项语音
情绪研究的元分析；Banse & Scherer 1996]：

- **F0 均值/范围/标准差**：高唤醒情绪 F0 更高、范围更大；低落情绪 F0 低且平。
- **F0 斜率（f0_slope，Hz/s）**：有声帧 F0 对时间的线性回归斜率。悲伤语音的
  F0 轮廓更平且下降 [Banse & Scherer 1996]，故负斜率增加负面分。
  > 早期版本用 std_f0 冒充「f0_drop」，已修正。
- **语速 speech_rate（音节/秒）**：焦虑/紧张常加快，抑郁常减慢。**优先由
  Paraformer 的字级时间戳计算**（字数 / 时间戳并集时长，中文一字一音节），
  这接近 de Jong & Wempe (2009) 以音节核计数的标准定义；无时间戳时回退到
  能量包络穿越法（粗略）。
- **停顿占比 pause_ratio**：犹豫、抑郁时静音段增多。
- **HNR（谐波噪声比）**：HNR 低=气声/粗糙（悲伤、紧张、压抑），故负面分用
  反向。聚合时**保留合法的负值帧**（噪声大于谐波的帧），仅排除 Praat 的
  -200 dB 无定义哨兵。
- **Jitter / Shimmer**：基频/振幅微扰，情绪激动或嗓音紧张时升高。v0.2 起同时
  进入**唤醒分**（各 0.10）：Banse & Scherer (1996) 报告恐惧/紧张语音的微扰升高；
  v0.1 的唤醒线索只有能量与语速，抓不住「低声屏息」型恐惧（CSEMOTIONS 上
  fearful 的唤醒均值仅 0.52）。

**z-score 参考值 [实测，见 §5]**：各指标先按 `config/prosody_norms.json`
中的 μ/σ 做 z-score 归一化到 [0,1]，再按子权重加权（子权重为 [启发式]）。

### 3.3 副语言事件（`src/models/pann_model.py`）

PANNs CNN10 [文献: Kong et al. 2020] 在 AudioSet 上训练，识别 527 类声学事件。
本项目关注与情绪强相关的副语言事件（贡献值为 [启发式]）：

| 事件 | n_contrib | a_contrib | 含义 |
|------|-----------|-----------|------|
| 笑声 Laughter | -0.5 | +0.3 | 正面 |
| 哭泣 Crying | +0.7 | +0.5 | 高负中唤 |
| 尖叫 Screaming | +0.3 | +1.0 | 极端唤醒 |
| 叹息/呼吸 Sigh / Breathing | +0.4 | -0.3 | 压抑/放松 |

聚合：以置信度加权累加，再除以 (1+总置信度) 归一化。AudioSet 无独立「Sigh」
标签，通过 Breathing 近似。标签表随仓库分发（`resources/panns/`，CC BY 4.0）。

### 3.4 物理声学（`src/features/physical.py`）

底层物理特征 [文献: Ilie & Thompson 2006 比较音乐与语音的声学线索]：

- **响度 RMS**、**频谱质心**：高唤醒 → 更响、更亮。
- **高频能量比**：过高(刺耳)/过低(沉闷)都偏负面，故用 |norm-0.5|*2。
- **SNR**：信号帧 vs 噪声帧功率比，同时用于动态权重。
- **能量动态范围 rms_dynamic_range_db（v0.3）**：帧 RMS 的 P95 与中位数之差。
  Banse & Scherer (1996) 报告愤怒/恐惧的强度变异升高；作为「平均响度」之外的唤醒线索。
- 聚合（v0.3）：**负面分只保留粗糙度**——v0.2 的消融显示含 SNR / 高频极端度的负面分与
  效价参照序**负相关**（仅保留 physical 时 ρV = −0.13），即这些项是反向噪声；SNR 仍
  提取，只用于音频质量与动态权重。唤醒分 = 0.30·响度 + 0.20·质心 + 0.15·高频比
  + 0.15·粗糙度 + 0.20·动态范围。
- **频谱粗糙度 roughness**：基于 Plomp & Levelt (1965) 与 Sethares (1993)
  的感觉不协和模型——相邻分量在 20–150 Hz 拍频内产生「粗糙」感。本实现取
  显著峰对，按拍频的高斯权重（峰值约 70 Hz）加权求和，是原模型的**简化版**
  （未按临界带宽随频率缩放）。

### 3.5 文本语义（`src/models/llm_model.py`）

Qwen3-1.7B 经 few-shot prompt 输出两个 0–1 浮点数（负面分、唤醒度）。v0.2 的
系统提示明确「无情绪的事实陈述两个值都应接近 0.5」，few-shot 加入两条中性陈述
与一条平静正面例子——v0.1 只有 4 条例子且无中性样本，模型对 AISHELL-3 中性
文本给出 0.66 的负面分，是绝对偏置的最大来源。
**首次调用 greedy 解码**（确定性，同一输入结果可复现），解析失败才低温采样
重试一次，二次失败降级为文本统计分。1.7B 小模型对精确数值评分能力有限，
few-shot 示例用于约束输出量纲；文本语义只是 6 模态之一。

### 3.6 文本统计（`src/features/text_stats.py`）

情感词典加权法（Lexicon-based）：分词后查词典判极性，加权求和；程度副词
（前 2 词窗口）放大/衰减，否定词（前 3 词窗口）奇数反转、偶数双重否定。

**方法学出处 [文献]**：中文词汇的维度情感规范见 Chinese EmoBank / CVAW
（Yu et al. 2016；Lee et al. 2022），提供 5,512 个中文词的 Valence-Arousal
九点量表评分。**本项目自建的 `resources/dictionaries/` 是极性词表（正/负），
不是 V-A 规范表**。v0.2 起：若用户把 CVAW 4.0 原始 CSV 放到
`resources/dictionaries/cvaw.csv`（仅限学术用途，需自行同意其条款，仓库不分发），
文本统计会用其 V-A 均值直接估计维度分并与极性法各半融合；否则唤醒度由标点、
程度副词密度、第一人称占比等浅层线索以 0.5 为基线上下调整 [启发式]
（v0.1 的公式基线为 0.1，陈述句恒为低唤醒，已修正）。

---

## 4. 声刺激生成：音乐心理学映射（`src/stimulus/`）

### 4.1 设计目标

根据检测到的情绪，生成**差异化**声刺激——非统一放松音，而是针对不同象限给出
有理论依据的声学干预。

### 4.2 声学参数 → 情绪映射 [文献]

依据 Juslin & Laukka (2003/2004)、Bresin & Friberg (2011)、Ilie & Thompson (2006)：

| 参数 | 高唤醒 | 低唤醒 | 正面 | 负面 |
|------|--------|--------|------|------|
| 脉冲率 pr | 快（激活） | 慢（平复） | — | — |
| 基频 f0 | — | — | 高（明亮） | 低（深沉） |
| 响度 | 大 | 小 | — | — |
| 频谱质心 | 高（亮） | 低（暗） | — | — |
| 起音 attack | 陡（冲击） | 缓（柔和） | — | — |
| 谐和结构 | — | — | 大三和弦（协和） | 整数泛音（紧张） |

**反向干预分支 [设计选择，见 §4.2.1]**：
- **Q2 焦虑**：高 arousal 反而用**慢脉冲**（0.25–1.0 Hz），意在向呼吸/心率
  节律靠拢以降唤醒。注意：只有 arousal 接近 1 时脉冲率才降到 0.25 Hz（15 次/
  分，接近静息呼吸），中等 arousal 时约 0.6 Hz，仍快于呼吸节律。
- **Q3 抑郁**：低 valence 反而**提 f0**（注入明亮感/能量，激活干预）。

### 4.2.1 与 ISO 原则的关系 [文献，对立证据]

音乐治疗中的 **ISO 原则**（Altshuler 1948）主张先用与当事人当前情绪**匹配**
的音乐建立共鸣，再逐步过渡到目标情绪。Starcke & von Georgi (2024) 的受控实验
发现：诱发悲伤后，先听悲伤歌曲再听快乐歌曲（ISO）比连续两首快乐歌曲更有效
地缓解悲伤。

本项目对 Q2/Q3 采取的是**直接反向调控**（不经匹配阶段），这是一个**有意的
设计选择**而非文献共识：其优点是单段刺激即可实施、参数映射连续；代价是缺
少 ISO 的共鸣阶段，对部分个体可能产生「被反驳」的不适。二次开发者若要实现
ISO 式渐进，可在 `strategies.py` 中先按当前 (v, a) 生成匹配段，再线性插值到
目标 (v', a') 生成第二段并交叉淡化。**本项目未对两种策略做过对照实验。**

### 4.3 软混合

按四象限隶属度对锚点参数加权混合（避免象限边界突变），再在主象限内按
valence/arousal 连续微调（`strategies.py::compute_params`）。

### 4.4 波形合成（`synthesizer.py`）

谐和音叠加 → 带通塑形 → 振幅调制(AM) → ADSR 包络 → 粉噪混合 → **RMS 响度
归一** → **数字峰值限幅** → 淡入淡出 → Haas 立体声（右声道延迟 12 ms，低于
回声感知阈值，产生宽度感 [文献: Haas 1951]）。

**粉噪的作用 [启发式，证据有限]**：粉噪（1/f 频谱）在 Q2/Q4 中用于**能量
遮蔽与频谱填充**，使纯音刺激不显单薄。早期版本曾引用 Söderlund et al. (2007)
支持「粉噪有平复作用」——该引用**不成立**：那项研究考察的是**白噪**对 ADHD
儿童**认知表现**的随机共振效应，与放松无关。目前关于色噪声对唤醒的直接证据
有限：已有瞳孔测量研究未发现不同色噪声对持续唤醒有差异性影响。因此本项目
**不主张粉噪具有生理平复效应**。

### 4.5 关于双耳节拍

v1 曾含双耳节拍（Binaural Beats）。v2 基于 Ingendoh et al. (2023) 系统综述移除——
该综述指出双耳节拍诱导特定脑波频段的证据不足。

### 4.6 声音安全

生成音频的**数字峰值限幅**为 `settings.stimulus.max_peak_dbfs`（默认 -10 dBFS），
响度按 RMS 归一到 [-30, -10] dBFS。**dBFS 是数字满刻度相对值，实际声压级
(SPL) 完全取决于播放设备与系统音量，本软件无法保证任何 SPL 数值**。使用者
应从低音量起听并逐步调到舒适水平；如需严格声压控制，请用声级计校准播放链路。
早期文档中「≈70–75 dB SPL、无听力损伤风险」的表述不成立，已删除。

---

## 5. 归一化方法（`src/fusion/normalizer.py`）

z-score 归一化：以「该特征在正常语音中的分布」为参照。

```
z = (value - mu) / sigma        # 偏离均值多少个标准差
z 截断到 [-2, 2]                 # 抑制极端离群点
norm = (z + 2) / 4              # 映射到 [0, 1]
```

**参考 μ/σ 的来源 [实测]**：`config/prosody_norms.json` 由 `scripts/evaluate.py
calibrate` 在 **AISHELL-3**（Apache-2.0，218 位普通话说话人的情绪中性朗读语料）
上分层抽样实测得到，含男女混合与分性别统计、样本量与计算日期。文件缺失时
回退到代码内的启发式常数（早期版本的凭空设定值，仅作兜底）。
> z-score 假设近似正态，对偏态特征（如 Jitter）是近似。

---

## 6. 局限性（二次开发须知）

- **未在自然情绪语料上验证**：evaluation.md 的情绪判别验证使用 CSEMOTIONS
  （专业配音员的**表演型**情绪，录音棚音质），通常比自然情绪更夸张，会
  **高估**真实场景表现；且无法检验噪声鲁棒性设计（SNR 规则不会触发）。
- **ASR 置信度是代理指标**（§2.3），动态权重的「低 ASR 置信度」分支很少被
  正确触发。
- **权重与锚点未经学习**：融合权重、V-A 锚点、副语言贡献、子权重均为启发式。
- **LLM 文本评分**：1.7B 小模型精度有限，正面文本评分可能偏高。
- **副语言事件**：AudioSet 非专为副语言设计，叹息等通过 Breathing 近似。
- **无微调**：全部预训练模型，未在特定数据微调，个体差异可能影响精度。
- **单语**：ASR/LLM/词典均针对中文普通话。
- **干预策略未做对照实验**（§4.2.1）。
- **科研用途**：本工具用于方法学探索，**不构成临床诊断或治疗手段**。

---

## 7. 参考文献

1. Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
2. Juslin, P.N., & Laukka, P. (2003). Communication of emotions in vocal expression and music performance: Different channels, same code? *Psychological Bulletin*, 129(5), 770–814.
3. Juslin, P.N., & Laukka, P. (2004). Expression, perception, and induction of musical emotions. *Journal of New Music Research*, 33(3), 217–238.
4. Banse, R., & Scherer, K.R. (1996). Acoustic profiles in vocal emotion expression. *Journal of Personality and Social Psychology*, 70(3), 614–636.
5. Bresin, R., & Friberg, A. (2011). Emotion rendering in music: Range and characteristic values of seven musical variables. *Cortex*, 47(9), 1068–1081.
6. Ilie, G., & Thompson, W.F. (2006). A comparison of acoustic cues in music and speech for three dimensions of affect. *Music Perception*, 23(4), 319–330.
7. Ma, Z., et al. (2024). emotion2vec: Self-supervised pre-training for speech emotion representation. *Findings of ACL 2024*.
8. Kong, Q., et al. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM TASLP*, 28, 2880–2894.
9. Plomp, R., & Levelt, W.J.M. (1965). Tonal consonance and critical bandwidth. *JASA*, 38(4), 548–560.
10. Sethares, W.A. (1993). Local consonance and the relationship between timbre and scale. *JASA*, 94(3), 1218–1228.
11. Haas, H. (1951). Über den Einfluss eines Einfachechos auf die Hörsamkeit von Sprache. *Acustica*, 1, 49–58.
12. Ingendoh, R.M., Posny, E.S., & Heine, A. (2023). Binaural beats to entrain the brain? A systematic review of the effects of binaural beat stimulation on brain oscillatory activity. *PLOS ONE*, 18(5), e0286023.
13. Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45, 1191–1207.
14. Yu, L.-C., Lee, L.-H., Hao, S., et al. (2016). Building Chinese affective resources in valence-arousal dimensions. *NAACL-HLT 2016*, 540–545.
15. Lee, L.-H., Li, J.-H., & Yu, L.-C. (2022). Chinese EmoBank: Building valence-arousal resources for dimensional sentiment analysis. *ACM TALLIP*, 21(4), 1–18.
16. Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A review of affective computing: From unimodal analysis to multimodal fusion. *Information Fusion*, 37, 98–125.
17. de Jong, N.H., & Wempe, T. (2009). Praat script to detect syllable nuclei and measure speech rate automatically. *Behavior Research Methods*, 41, 385–390.
18. Starcke, K., & von Georgi, R. (2024). Music listening according to the iso principle modulates affective state. *Musicae Scientiae*, 28(3).
19. Söderlund, G., Sikström, S., & Smart, A. (2007). Listen to the noise: Noise is beneficial for cognitive performance in ADHD. *Journal of Child Psychology and Psychiatry*, 48(8), 840–847. ——**仅作为被更正的错误引用保留说明，不支持本项目任何主张。**
20. Shi, Y., et al. (2021). AISHELL-3: A multi-speaker Mandarin TTS corpus. *Interspeech 2021*.
21. AIDC-AI (2025). CSEMOTIONS: A Mandarin emotional speech dataset. Hugging Face `AIDC-AI/CSEMOTIONS`（Apache-2.0；NOTICE 声明含 HLTSingapore ESD 衍生内容）。
22. Schönwiesner, M., & Bialas, O. (2021). slab: An easy to learn Python package for psychoacoustic experiments. *JOSS*, 6(62), 3284.
23. Schuller, B., Batliner, A., Steidl, S., & Seppi, D. (2011). Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge. *Speech Communication*, 53(9–10), 1062–1087.
