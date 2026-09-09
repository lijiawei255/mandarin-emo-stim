# Mandarin-EmoStim 仓库审核与改进方案

日期：2026-09-10
目标：使本仓库成为一个**可被他人参考的、科学依据经过如实标注的**闭环可听声情绪监测与个性化调控开源科研代码库。

---

## 0. 审核基线（本次实测）

| 项目 | 结果 |
|------|------|
| 默认单元测试（非 GPU/非 slow） | 92 passed, 31 skipped |
| GUI 冒烟（`-m slow`, offscreen） | 17 passed |
| 模型 + 端到端（`-m gpu`, RTX 4000 Ada 12GB） | 14 passed（约 4 分 21 秒） |
| 真实窗口渲染（1800×1055 / 1600×900） | 中文渲染正常，无重叠/截断 |

代码规模：源码约 4200 行 + 测试约 1600 行 + 文档约 550 行。

结论：**工程完成度高、健壮性处理扎实**；主要短板在「科学性论据的严谨度」与「开源仓库的可参考性配套」。

---

## 1. 科学性问题清单与处置

### 1.1 P0 —— 整条链路从未在标注数据上验证

**现状**：融合权重、9 类情绪→V-A 锚点、韵律 z-score 参考量、PANNs 事件贡献值、韵律/物理子权重，全部是手工设定的启发式常数。README 却宣称输出「可解释、可复现的量化情绪指标」。对一个供他人参考的科研仓库，这是最严重的问题——读者无从判断这些数字的意义。

**数据集合规性筛查结论**（本次逐一核实许可条款）：

| 语料 | 许可 | 门禁/表单 | 语言 | 情绪标注 | 采用 |
|------|------|-----------|------|----------|------|
| **AISHELL-3**（OpenSLR 93） | **Apache-2.0**，允许商用 | 无 | 普通话，218 人，85h | 无（情绪中性朗读） | ✅ 采用 |
| **CSEMOTIONS**（AIDC-AI） | **Apache-2.0** + NOTICE | 无 | 普通话，10 人，10.24h | 7 类 | ✅ 采用 |
| ESD（HLTSingapore） | 仓库内 LICENSE 为 MIT，官网却称需提交许可表单（**自相矛盾**） | 官网要求表单 | 中/英 | 5 类 | ❌ 排除（歧义） |
| EmotionTalk（BAAI） | CC-BY-NC-SA-4.0 | **auto-gated** | 普通话 | 7 类 | ❌ 排除（门禁 + NC 与本仓库 Apache-2.0 不一致） |
| RAVDESS | CC-BY-NC-SA-4.0 | 无 | **英语** | 8 类 | ❌ 排除（非中文 + NC 条款） |
| MAGICDATA 等 OpenSLR 中文语料 | CC-BY-NC-ND-4.0 | 无 | 普通话 | 无 | ❌ 排除（ND 条款） |

**关于 CSEMOTIONS 渊源的完整披露**（须写入 `docs/evaluation.md` 与 README）：
CSEMOTIONS 由 AIDC-AI（阿里国际数字商业）以 Apache-2.0 发布，其 `NOTICE` 文件声明该数据集**incorporates 第三方数据集 Emotional-Speech-Data (ESD, HLTSingapore)，以 MIT 许可，并附「This database can only be used for research purpose」**。本项目的使用完全落在该限制之内：
- 用途为科研方法学验证，非商业用途；
- **不转分发任何音频**，评测脚本在运行时按需下载，仓库只提交脚本与结果表；
- 在文档中完整署名 AIDC-AI 与 HLTSingapore 并复制其 NOTICE 要点。

**三层验证设计**（新增 `scripts/evaluate.py`，子命令式）：

#### 第 1 层：`calibrate` —— 用 AISHELL-3 实测普通话韵律基准（零许可风险）

这一层**不是测准确率，而是修掉一个真实的科学缺陷**。`src/fusion/normalizer.py` 的 `PROSODY_STATS`（8 个特征的 μ/σ）目前是**凭空设定**的常数，文档却称「来自中文普通话语料统计」。

- 在 AISHELL-3 上做**性别均衡的分层抽样**（默认每性别 300 句，可 `--limit` 调整），实测 `mean_f0 / std_f0 / f0_range / speech_rate / pause_ratio / hnr / jitter_local / shimmer_local` 的 μ 与 σ。
- 产出 `config/prosody_norms.json`（含样本量、抽样方式、语料版本、计算日期），`normalizer.py` 改为从该文件读取，并保留旧常数作为文件缺失时的回退。
- 同时产出**男女分组统计**，供二次开发者按需选用（现有单一混合 μ/σ 对男女声都不准）。
- **副产物 A**：用 AISHELL-3 的标注转写测 Paraformer 的**真实字错率（CER）**，为「ASR 置信度是启发式代理」这一局限性提供量化背景。
- **副产物 B**：在情绪中性朗读语音上跑完整管线，检验输出是否**居中**。若 Negative 的均值系统性偏离 0.5，即为**标定偏差**，如实报告并给出修正建议。

#### 第 2 层：`emotion` —— 用 CSEMOTIONS 做情绪判别验证

- 7 类情绪 → 参照象限映射：happy/playfulness → Q1；angry/fearful → Q2；sad → Q3。
  **neutral 与 surprise 单独处理**：neutral 无明确象限归属（应落在中心附近，单独报告其 V-A 分布）；surprise 的效价在文献中本就有争议（可正可负），单独报告不计入严格象限准确率。这一处理方式在文档中明确说明理由，不做掩饰。
- 指标：
  1. 各情绪类别的 Valence / Arousal 均值与标准差，检验与 Russell 模型的**方向一致性**（例如 sad 的 arousal 应显著低于 angry）；
  2. 四象限混淆矩阵与准确率；
  3. V-A 与类别参照值的 Spearman 相关；
  4. **6 模态消融**：逐一屏蔽每个模态，量化其对判别力的贡献，检验「多模态融合优于单模态」这一核心主张**是否真的成立**；
  5. 动态权重开/关的对比（CSEMOTIONS 为录音棚级音质，SNR 规则不会被触发，此点须如实说明）。
- 默认分层抽样（每情绪每说话人若干句，`--limit` 控制），完整跑法一并记录。

#### 第 3 层：`tests/test_scientific_behavior.py` —— 合成受控信号的模块级验证（进 CI，无需任何外部数据）

程序化生成已知参数的信号，断言各模块的**响应方向与单调性**：F0 升高 → arousal 升高；语速加快 → arousal 升高；HNR 降低 → negative 升高；频谱粗糙度升高 → negative 升高；Q2/Q3 干预分支的脉冲率与基频方向正确。

**产出**：`docs/evaluation.md`，含完整结果表、可复跑命令、数据集署名与许可声明，以及**诚实的解读——包括不达预期的项**。README 增设「验证状态」小节链接过去。

**必须写明的局限性**：CSEMOTIONS 为专业配音员的**表演型情绪**，通常比自然情绪更夸张，会**高估**真实场景表现；且为录音棚音质，无法检验噪声鲁棒性设计。这两点在文档中明确标注。

### 1.2 P1 —— 引用与实际主张不符

| 位置 | 问题 | 处置 |
|------|------|------|
| `synthesizer.py` / `research_notes.md` §4.2、参考文献 [9] | 粉噪「平复作用」引用 Söderlund et al. 2007。该研究实为**白噪**对**ADHD 儿童认知表现**的随机共振效应，不支持「粉噪令人放松」 | 删除该因果主张。改为：粉噪用于**能量遮蔽与频谱填充**，并如实标注「其放松效应在文献中证据有限，近期瞳孔测量研究未发现不同色噪声对唤醒的差异性影响」 |
| `emotion_mapping.json` 的 n_base/a_base | 文档称「依据心理学文献中的实证锚点」，但无任何具体出处 | 补 Russell (1980) 情绪环坐标 + Warriner et al. (2013) ANEW 扩展；无法一一对应的，如实标注为「基于文献方向性的启发式默认值，非实证测量值」 |
| `normalizer.py` PROSODY_STATS | 文档称「来自中文普通话语料统计」，实为凭空设定的估计值 | 由 §1.1 第 1 层**从 AISHELL-3 实测替换**，并记录样本量与抽样方式；额外提供男女分组统计 |
| `text_stats.py` 词典法 | 无 V-A 维度依据 | 补引 **Chinese EmoBank / CVAW (Yu et al. 2016)** —— 中文词汇的 V-A 九点量表规范，是本模块方法学的正统出处；并说明本项目自建词表为极性词表而非 V-A 规范表 |
| Q2 慢脉冲 / Q3 提亮的「反向干预」 | 与音乐治疗 **ISO 原则**（先匹配当前情绪再引导）方向相反，文档未讨论 | 新增 `research_notes.md` §4.2.1，说明这是一个**明确的设计选择**，并引 Starcke & von Georgi (2024) 的 ISO 原则实验结果作为对立证据，把「本工具采用直接调控而非 ISO 渐进引导」列为可被质疑与替换的设计点 |

### 1.3 P1 —— dBFS ≠ SPL 的安全声明

**现状**：README、GUI「关于」框、`research_notes.md` 均称「峰值 -10 dBFS ≈ 70–75 dB SPL，无听力损伤风险」。

**问题**：dBFS 是数字满刻度相对值，实际声压完全取决于播放设备与系统音量。此声明**不成立**，且属于安全承诺，必须修正。

**处置**：全部改为：
> 生成音频的数字峰值限幅为 -10 dBFS。**实际声压级取决于你的播放设备与系统音量，本工具无法保证。** 首次使用请在低音量下起听并逐步调整到舒适水平；如需严格的声压控制，请使用声级计对播放链路做校准。

### 1.4 P2 —— 方法学实现的可改进项

1. **ASR 置信度是文本长度启发式**，却驱动整个动态权重机制。已查证 FunASR 1.0.25 的 `Paraformer.inference` 计算了 `am_scores` 但**不在返回结果中暴露**（`funasr/models/paraformer/model.py:548`），因此无法低成本换成真实后验概率。处置：保留启发式，但在代码与文档中**明确标注为代理指标**，并在 `research_notes.md` 局限性一节写明这是动态权重机制的薄弱环节。
2. **语速估计**用能量包络穿越均值计数，而 Paraformer 已返回字级时间戳。处置：改用时间戳计算音节率（`字数 / 有效语音时长`），保留能量法作为无时间戳时的回退。
3. **`f0_drop` 实际用的是 `std_f0`**，与命名和文档描述不符。处置：改为真正的 F0 时间斜率（对有声帧 F0 序列做线性回归取斜率）。
4. **LLM 首次调用 temperature=0.1 采样**，破坏可复现性。处置：首次即 greedy（`do_sample=False`），失败时才改用低温采样重试。
5. **HNR 过滤 `> 0`** 丢弃了合法的负 HNR 帧（噪声大于谐波的帧，恰恰是情绪相关信号）。处置：改为过滤 parselmouth 的无定义哨兵值。
6. **合成器响度处理与文档不符**：`synthesizer.py:131` 按**峰值**归一却注释为 RMS，且随后统一乘 0.7，导致 -10 dBFS 限幅**永不触发**，实际峰值恒为 -13 dBFS。处置：改为真正的 RMS 归一 + 峰值限幅，使限幅逻辑真实生效，注释与行为一致。

---

## 2. 代码质量问题与处置

| 严重度 | 问题 | 处置 |
|--------|------|------|
| 高 | **管线降级承诺不成立**：非关键步骤失败仅记日志继续，但 `_step_fuse` / `_finalize` 会因缺 key 直接 `KeyError`（`pipeline.py:157-195`） | 为 6 模态分数设中性默认值（0.5），任一模态失败时降级而非崩溃，并在结果中返回 `degraded_modalities` 列表，GUI/CLI 如实展示 |
| 高 | **Ctrl+C 修复可能无效**：`app.py:81` 的 `QTimer` 未 `connect` 任何 Python 槽，Qt 在 C++ 层处理超时，不会唤醒 Python 信号处理 | 连接一个空 lambda 槽；补一个可验证该行为的测试 |
| 中 | **PANNs 标签读 `~/panns_data/`**（`pann_model.py:77`），违反「数据全在 portable_data」的承诺；缺失时**静默**返回中性分，用户无从察觉副语言支路已失效 | 改为读 `portable_data/models/panns/`，随下载器一并获取；缺失时记 WARNING 并在结果中标记该模态降级 |
| 中 | 单卡仍包 `torch.nn.DataParallel`（`pann_model.py:67`），无收益且使 `state_dict` 键名复杂化 | 移除 |
| 中 | **死代码与过时描述**：`ModelLoadWorker` 及 `_on_model_progress`/`_on_models_loaded` 已无调用方；`main_window` docstring 仍称模型在工作线程加载；面板对象名仍是 `RedPanel`/`BluePanel`/`YellowPanel`；`user_guide.md` 仍描述红蓝黄黑构成主义界面 | 删除死代码，重命名为语义化对象名（`InputPanel`/`MetricsPanel`/`ModalPanel`/`StimulusPanel`），同步全部文档 |
| 低 | 无「科学行为」回归测试 | 补：已知情绪特征向量应落入预期象限；Q2/Q3 干预分支的方向性断言 |

---

## 3. 开源仓库规范

**已达标**：双语 README、Apache-2.0、测试夹具来源与协议标注、隐私数据 gitignore、详尽中文 docstring、CONTRIBUTING 含合规要点。

**补齐清单**：
1. `.github/workflows/ci.yml` —— Ubuntu + Windows × Python 3.10，跑非 GPU 测试 + `ruff` 静态检查。
2. `pyproject.toml` —— 项目元数据、版本号单一来源（0.1.0）、ruff/pytest 配置（`pytest.ini` 段合并进来，含 markers）。
3. `CITATION.cff` —— 使他人可正确引用本仓库。
4. `.github/ISSUE_TEMPLATE/`（bug / feature）+ `PULL_REQUEST_TEMPLATE.md` + `CODE_OF_CONDUCT.md`。
5. `CHANGELOG.md` 整理：`Unreleased` → `[0.1.0] - 2026-09-10`。
6. `.gitignore` 精简：移除与本项目无关的模板段（Django/Flask/Scrapy/Marimo/Streamlit 等），保留 Python + 本项目专有规则。
7. `scripts/lit_search.py` 加入 `.gitignore`（用户已确认不入库）。

---

## 4. 仓库装饰

- **README 徽章**：License / Python 3.10 / Platform Windows / CI status / Release / Tests passing。
- **界面截图**：真实渲染的新 UI 截图放入 `docs/images/`，README 顶部展示。
- **架构图**：Mermaid 流程图（音频 → 6 模态 → 融合 → 象限 → 声刺激），直接在 README 渲染。
- **Release v0.1.0**：打 tag、写 Release Notes、附界面截图与四象限声刺激样例 WAV。
- GitHub 仓库 topics 与 description 更新。

---

## 5. UI 风格替换：Light Bauhaus → 暖奶油（warm ivory）

**配色**（用户已确认为暖奶油色系）：

| 角色 | 色值 |
|------|------|
| 页面底色 | `#FAF9F5` (ivory) |
| 卡片/面板 | `#FFFFFF` / 次级面板 `#F0EEE6` |
| 边框 | `#E5E2D9` |
| 正文 | `#1F1E1D` (slate) |
| 次级文字 | `#6E6D68` |
| 强调 / 主操作 | `#D97757` (clay) |
| 成功 / 警告 / 错误 | `#788C5D` / `#B8860B` / `#BC4C3C`（暖调，与整体协调） |
| 圆角 | 8px（替代当前 0px 直角） |

**实现要点**：新增 `src/gui/theme.py` 作为**唯一调色板来源**，`styles.qss` 由模板生成或引用同一组常量；清除 5 个控件文件与 `main_window.py` 中全部内联硬编码颜色。这样后续换主题只改一处。

---

## 6. UI 视觉验证（三层）

1. **几何自动检查**（进 pytest，CI 可跑）：新增 `scripts/ui_geometry_check.py` + `tests/test_gui_layout.py`。遍历全部可见控件，检测：
   - 同级控件矩形相交（重叠）；
   - `QLabel`/`QPushButton` 的 `sizeHint().width()` 超过实际分配宽度（文字被挤压/截断）；
   - 子控件溢出父控件边界；
   - 控件高度小于其最小高度需求。
   覆盖 **3 种分辨率**（1280×720 / 1440×900 / 1920×1080）× **3 种状态**（空闲 / 分析完成 / 加载浮层显示中）。
2. **真实窗口截图目视检查**：`scripts/gui_screenshot.py` 扩展为覆盖上述 9 种组合（本次已验证真实渲染下中文字体正常，offscreen 模式会退化为方块，故必须用真实平台渲染）。我逐张检查。
3. **视觉模型交叉验证**：用 `qwen3-vl-plus` 对每张截图做独立判读（是否有重叠、截断、错位、对比度不足），作为第二意见。

---

## 7. 执行顺序

| 阶段 | 内容 | 可独立验证 |
|------|------|-----------|
| A | 代码质量修复（§2）+ 方法学实现改进（§1.4） | 现有测试全绿 + 新增回归测试 |
| B | UI 主题替换（§5）+ 视觉验证三层（§6） | 几何检查通过 + 截图目视 + 视觉模型 |
| C | 文档科学性修正（§1.2、§1.3） | 人工复核 |
| D | 三层验证（§1.1）：AISHELL-3 基准实测 → CSEMOTIONS 情绪验证 → 合成信号模块测试；产出 `docs/evaluation.md` | 评测脚本可复跑，第 3 层进 CI |
| E | 仓库规范（§3）+ 装饰（§4）+ Release | CI 绿 + Release 页面 |

阶段 A/B/C 相互独立，D 依赖 A（修复后的管线才值得评测），E 最后。

---

## 8. 明确不做（YAGNI）

- 不重训或微调任何模型。
- 不改动 6 模态的整体架构与融合公式结构（只修实现 bug 与如实标注依据）。
- 不做 macOS / Linux 适配（README 已如实声明仅 Windows+NVIDIA 验证）。
- 不引入新的重型依赖（评测所需的 `datasets` 库仅作为可选依赖，写入 `requirements-eval.txt`，不进主依赖）。
- 不转分发任何数据集音频。
