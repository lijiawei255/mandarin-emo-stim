# 验证报告（v0.1.0）

本报告如实记录本仓库方法学在公开语料上的三层验证结果，**包括不达预期的项**。
复跑命令见文末。方法学的证据等级标注见 [research_notes.md](./research_notes.md)。

> **先说结论的边界**：本项目的输出**未经自然情绪语料验证**。第 2 层使用的是专业配音员的
> 表演型情绪，通常比自然情绪更夸张，会**高估**真实场景表现；且为录音棚音质，噪声鲁棒性
> 设计（SNR 动态权重）在此不会被触发。请把下面的数字理解为「方法在理想条件下的方向是否
> 正确」，而不是「在真实用户语音上有多准」。

---

## 1. 数据集与许可

| 语料 | 用途 | 许可 | 获取方式 |
|------|------|------|----------|
| **AISHELL-3**（Shi et al. 2021；OpenSLR 93） | 第 1 层：韵律基准实测、字错率、中性语音输出分布 | **Apache-2.0**，允许商用 | HF 镜像 `shenyunhang/AISHELL-3`，按需下载 `test/` 子集 |
| **CSEMOTIONS**（AIDC-AI 2025） | 第 2 层：情绪判别验证 | **Apache-2.0** + NOTICE | HF `AIDC-AI/CSEMOTIONS` parquet |

**CSEMOTIONS 渊源的完整披露**：其 `NOTICE` 文件声明该数据集 *incorporates* 第三方数据集
Emotional-Speech-Data（ESD，HLTSingapore，MIT 许可），并附「This database can only be used for
research purpose」。本项目的使用完全落在该限制内：仅用于科研方法学验证、非商业；**不转分发任何
音频**（评测脚本运行时按需下载到已被 gitignore 的 `portable_data/eval/`，仓库只提交脚本与结果表）；
在此署名 AIDC-AI 与 HLTSingapore。

排除的候选语料及理由：ESD 官网要求提交许可表单而其仓库 LICENSE 为 MIT（自相矛盾，排除歧义）；
EmotionTalk（BAAI）为 auto-gated 且 CC-BY-NC-SA；RAVDESS 为英语且 NC；MAGICDATA 等 OpenSLR
中文语料为 CC-BY-NC-ND。

---

## 2. 方法

### 第 1 层：AISHELL-3（零许可风险）

- **抽样**：测试集 249 位说话人中，每性别随机抽 10 位（seed 20260910），每人取编号前 40 句中的
  30 句，共 600 句（女 300 / 男 300）。只下载被抽中的文件。
- **calibrate**：对每句以 `src.features.prosody.extract` 提取 9 个韵律指标（语速 = 转写字数 /
  音频时长，含首尾静音，故略低估），计算 mixed / female / male 三组 μ、σ，写入
  `config/prosody_norms.json`。**这一层不是测准确率，而是把此前凭空设定的 z-score 参考值换成
  实测值。**
- **neutral**：对其中 200 句跑完整管线，报告 Paraformer 字错率（CER，去标点后按字计算）、
  ASR 置信度代理、SNR，以及 Negative / Valence / Arousal 的分布与象限直方图。中性朗读语音的
  理想输出应接近 (0.5, 0.5)；若系统性偏离，即为标定偏差。

### 第 2 层：CSEMOTIONS

- **抽样**：7 类情绪 × 10 位说话人，每单元蓄水池抽样 6 句（seed 20260910），共 420 句。
- **参照象限**：happy / playfulness → Q1，angry / fearful → Q2，sad → Q3。**neutral 与 surprise
  不计入严格象限准确率**：neutral 无象限归属（应落在中心附近，单独报告其分布）；surprise 的效价
  在文献中本就不稳定（可正可负）。
- **指标**：各情绪 V-A 均值与标准差；8 组成对方向检验（如 arousal: sad < angry）；Valence /
  Arousal 与 Russell 环形参照序的 Spearman ρ；象限混淆矩阵与准确率（随机基线 0.25）；
  6 模态消融（去除单一 / 仅保留单一，其余置中性 0.5）；动态权重开/关（关 = SNR 15 dB、
  ASR 0.8、无副语言事件的固定权重）。消融与开关在缓存的模态分数上离线重算，不重跑模型。

### 第 3 层：合成受控信号（`tests/test_scientific_behavior.py`，进 CI）

程序化生成已知参数的信号，断言各模块的响应**方向**：F0 升高 / 语速加快 → 唤醒升高；HNR 降低 /
拍频粗糙度升高 → 负面升高；象限角点归属与软过渡连续；Q2 高唤醒脉冲更慢、Q3 低效价基频更高；
粉噪仅 Q2/Q4；跨象限边界参数连续。这一层不需要任何外部数据。

---

## 3. 结果

<!-- RESULTS -->

---

## 4. 解读

<!-- INTERPRETATION -->

---

## 5. 复跑

```bash
pip install -r requirements-eval.txt
python scripts/evaluate.py calibrate --per-gender 300
python scripts/evaluate.py neutral --limit 200
python scripts/evaluate.py emotion --per-cell 6
python scripts/evaluate.py report          # 表格 → portable_data/eval/results/report_tables.md
pytest tests/test_scientific_behavior.py   # 第 3 层
```

环境：Windows 11，RTX 4000 Ada（12 GB），CUDA 12.1，torch 2.3.1，本仓库 v0.1.0。
评测脚本默认从官方 Hub 下载（`HF_ENDPOINT=https://huggingface.co`；hf-mirror 对这两个数据集
返回 308 跳转会使 `huggingface_hub` 元数据校验失败）。
