# 更新日志

本项目版本变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.5.0] - 2026-09-10

会话模式：让使用流程符合真实受试者的数据逻辑（情绪有惯性、不会瞬间切换；干预效果要看前后测）。
情绪判别方法学未变。

### 新功能
- **会话 → 试次（前测 / 刺激 / 后测）**：右栏新增会话面板，「开始会话」记录受试者档案与实验者标注的
  诱发目标；每段录音归入当前试次的当前阶段；生成刺激后自动进入后测；「新试次」「切到前测/后测」；
  「结束并导出」写出逐段明细与试次汇总（前后测均值与差值、刺激参数）两个 CSV。
  数据库 `portable_data/sessions/sessions.db`（`src/session/model.py`）。
- **会话内状态估计**：对 negative / arousal 做指数平滑（α 0.5），象限判定带滞回（死区 0.05、连续 2 段
  一致才切换；会话开头为临时判定，锁定前随最新候选走）。**驱动刺激的是平滑状态**而非最后一段。
  参数在 `settings.json` 的 `session` 段（`src/session/state_tracker.py`）。
- 不开会话时行为与 v0.4 相同（每段独立判定）。

### 评测
- `scripts/evaluate.py sequence`：用 CSEMOTIONS 同说话人同情绪的 6 句序列离线模拟会话。默认设置下
  象限翻转率 55% → 0%，末段准确率 0.735 → 0.755，全位置准确率不变；「从未正确」序列 6% → 14%
  （若不做临时起始判定则为 31%）。详见 `docs/evaluation.md` §0.−1。

### 文档
- README 的「闭环」改为「测量闭环」：v0.5 能观察刺激前后的变化，但跨试次如何调整刺激由实验者决定，
  没有自动策略；research_notes 新增 §2.7（情绪惯性、前后测协议）；用户手册新增会话流程。

### 界面
- 会话面板放在多模态分解下方的空余区域；波形区设最小高度，避免小屏下被上方面板挤压。

## [0.4.0] - 2026-09-10

补上 0.3.0 报告列为待办的两项（ASR 真实置信度、不确定性校准），并把个人基线改为可预存的受试者档案。
情绪判别方法学未变，交叉验证数字与 0.3.0 一致（`docs/evaluation.md` §0.0）。

### 新功能
- **受试者档案**：个人基线按受试者保存（`portable_data/calibration/profiles/<name>.json`），同一档案多次
  录音取平均；输入区新增「受试者档案」下拉框，菜单改为「为受试者录制基线 / 从文件添加 / 删除档案」，
  CLI `--profile` / `--calibrate-user` / `--list-profiles`。基线可在受试者平静时预录、实验当天选用。
  历史记录新增校准来源、档案名与不确定性字段（旧库幂等增列）。
- **ASR token 后验置信度**：在 FunASR Paraformer 内部方法上包一层暂存 decoder 输出，置信度取保留 token 的
  最大后验均值（`confidence_source=posterior`，无钩子时回退代理指标）。与逐句 CER 的 Spearman ρ ≈ −0.32
  （代理指标仅 +0.03）；`asr_confidence_threshold` 改为 0.85。
- **可信度等级**：`config/uncertainty_thresholds.json` 由交叉验证留出预测生成（分歧度三分位的实测象限
  准确率），GUI 与 CLI 显示「可信度 X（该档实测准确率 y）」。**实测方向与直觉相反**：分歧度与判对正相关
  （ρ +0.35，最一致档 0.48 vs 最分歧档 0.85），因为「都接近中性」也算一致但只是证据弱；等级因此按实测
  准确率排序命名，不假设方向。

### 评测
- `neutral` / `emotion` 报告后验置信度分布及其与 CER 的相关；`emotion` 写出可信度阈值文件；
  `docs/evaluation/` 归档 v0.3 汇总。

## [0.3.0] - 2026-09-10

不依赖新数据的方法学与可用性改进；评测协议升级为性别均衡的 5 折说话人交叉验证（`docs/evaluation.md` §0.1）。

### 新功能
- **个人基线校准**：GUI「文件 → 个人基线校准（录 30 秒平静朗读）/ 从音频文件设置 / 清除」，CLI
  `--calibrate-user` / `--clear-user-calibration`。个人偏移替代语料偏移（`src/fusion/personal_calibration.py`，
  `portable_data/calibration/user_baseline.json`）。交叉验证中「说话人级中性基线」是折间最稳的配置
  （0.680 ± 0.014，效价 ρ 0.82）。
- **可学习融合（可选）**：岭回归把 12 维原始模态分映射到 (negative, arousal)，系数可读
  （`config/learned_fusion.json`，`src/fusion/learned_fusion.py`）；`settings.json` 设 `fusion_mode="learned"` 启用，
  文件缺失回退手工权重。交叉验证 0.715 ± 0.033（效价 ρ 0.76）。默认不启用：训练数据为表演型语料。
- **不确定性输出**：融合结果附带活跃模态的加权标准差 `uncertainty`，GUI 象限标题下显示「模态分歧」与校准来源，
  CLI 同步打印。

### 方法学
- **物理声学不再提供效价信息**：实测粗糙度随唤醒升高（angry 0.66 > happy 0.56 > sad 0.50 > neutral 0.42）
  而与效价无关，负面分恒 0.5；粗糙度按 AISHELL-3 中性中位数对数居中后进入唤醒分。
- **唤醒加入能量动态范围**（帧 RMS P95 − 中位数）；在 CSEMOTIONS 上各情绪几乎无差异，未带来改善。
- 降级模态不施加校准偏移（v0.2 补丁并入）。

### 评测
- 性别均衡 5 折说话人 CV：校准关 / 默认 / 训练折选权重 / 岭回归 / 说话人级基线；`neutral` 改为先写偏移再
  离线重算输出，避免旧偏移污染中性分布。
- 结果：默认配置 0.670 ± 0.076（v0.2 单次 test 0.63）；语料级校准在物理模态修正后已无净收益
  （0.674 关 vs 0.670 开）；唤醒度 ρ 0.34 仍未改善。

### 仓库
- `.pre-commit-config.yaml`（ruff --fix、大文件拦截）；`docs/evaluation/` 归档 v0.1 / v0.2 汇总。

## [0.2.0] - 2026-09-10

针对 0.1.0 验证报告的不利结论做的方法学修订，全部评测重跑；情绪语料按说话人划分 dev / test，
权重只在 dev 上选择。结果对照见 `docs/evaluation.md` §0。

### 方法学
- **中性校准（基线归一化）**：融合前对每个模态加常数偏移（`config/modality_calibration.json`，
  由 `scripts/evaluate.py neutral` 在 AISHELL-3 中性语音上实测 offset = 0.5 − 原始均值，钳制 ±0.3）。
  `settings.json` 的 `fusion_calibration.enabled` 可关闭；结果新增 `modal_scores_raw`。
- **LLM 提示重校**：系统提示明确「无情绪陈述 ≈ 0.5」，few-shot 加入中性陈述与平静正面例子；
  对中性文本的负面分由 0.66 降到 0.52。
- **韵律唤醒加入 jitter / shimmer**（各 0.10，其余权重重新归一），依据 Banse & Scherer 1996。
- **文本统计唤醒以 0.5 为基线**（旧公式基线 0.1，陈述句恒为低唤醒）；可选接入 CVAW 维度词典
  （`resources/dictionaries/cvaw.csv`，用户自行获取，仓库不分发）。
- **物理声学负面分去掉固定项** 0.3·0.5，权重和为 1。

### 结果（CSEMOTIONS，n=294 计入象限）
- 象限准确率 0.52 → **0.68**（test 说话人 0.63）；效价 ρ 0.69 → **0.79**；方向检验 8/8。
- 多模态融合 0.68 **高于** emotion2vec 单模态 0.61（0.1.0 为 0.52 < 0.55）；去掉 LLM 支路由
  「提升」变为「下降」。
- 中性语音 negative μ 0.55 → **0.51**，判入 Q3 的比例 62% → 43%。
- **未改善**：唤醒度 ρ 0.31（test 0.24），sad 多数判入 Q2；动态权重仍未被检验。

### 评测脚本
- `neutral` 写出中性校准文件；`emotion` 报告校准开/关、dev/test、权重网格；消融改在原始分上重算。

## [0.1.0] - 2026-09-10

首个公开版本。在功能完成的基础上完成了一轮**科学性审核**：把每处方法学按证据等级
（[文献] / [启发式] / [实测]）如实标注，修正引用错配与不成立的安全声明，用公开语料
实测替换凭空设定的参数，并补齐开源仓库配套。

### 科学性修正

- **韵律 z-score 基准改为实测**：`config/prosody_norms.json` 由 `scripts/evaluate.py calibrate`
  在 AISHELL-3（Apache-2.0，情绪中性普通话朗读）上分层抽样测得，含男女分组；旧常数
  降为兜底并如实标为「凭空设定」。
- **三层验证**（`docs/evaluation.md`）：AISHELL-3 基准实测 + 字错率 + 中性语音输出分布；
  CSEMOTIONS（Apache-2.0）情绪判别的象限混淆矩阵、V-A 方向一致性、Spearman、6 模态消融、
  动态权重开关；合成受控信号的模块级方向性测试进 CI。结果**如实报告，含不达预期项**。
- **引用错配更正**：粉噪「平复作用」所引 Söderlund 2007 为白噪对 ADHD 儿童认知的研究，
  不支持该主张，已删除并改述为能量遮蔽/频谱填充。
- **安全声明更正**：删除「-10 dBFS ≈ 70–75 dB SPL，无听力损伤风险」——dBFS 与声压级
  无固定关系，实际声压取决于播放设备；改为数字限幅声明 + 低音量起听建议。
- **与 ISO 原则的关系**：新增讨论，说明 Q2/Q3 的反向干预是有意设计选择，并引 Starcke &
  von Georgi (2024) 作为对立证据；未做对照实验。
- V-A 锚点、副语言贡献、子权重等标注为启发式；文本统计补 Chinese EmoBank/CVAW 出处；
  ASR 置信度写明为文本长度代理指标（FunASR 不暴露后验概率）。

### 方法学实现改进

- 语速优先由 Paraformer 字级时间戳计算音节率（`vad.syllable_rate`），能量法仅作回退。
- `f0_drop` 改为真实的 F0 时间斜率（此前用 `std_f0` 冒充）。
- HNR 聚合保留合法负值帧，仅排除 Praat 的 -200 无定义哨兵。
- LLM 首次调用 greedy 解码（可复现），解析失败才低温采样重试。
- 合成器响度按 RMS 归一（此前按峰值归一却注释为 RMS），峰值限幅读取
  `settings.stimulus.max_peak_dbfs` 并真正生效（此前统一 ×0.7 使限幅永不触发）。
- **Q3 基频映射方向修正**：代码为「valence 越低 f0 越低」，与文档所述「提亮激活」相反。
- **四象限软混合真正实现**：此前隶属度解包后未使用，只取主象限分支，跨 0.5 时参数跳变。

### 修复

- 管线非关键模态失败时降级为中性分并列入 `degraded_modalities`（此前会在融合步骤 KeyError）。
- Ctrl+C 保活定时器未连接 Python 槽，信号处理器永不运行；已修正并加测试。
- PANNs 标签表随仓库分发（`resources/panns/`，AudioSet 元数据 CC BY 4.0），不再读用户主目录；
  缺失时显式降级而非静默中性分；移除单卡上无意义的 `DataParallel`。
- 移除 GUI 死代码（`ModelLoadWorker` 等），面板对象名语义化。

### 界面

- 新主题：ivory 底色 + 单一陶土橙强调色，`src/gui/theme.py` 为唯一调色板来源，文字色按
  WCAG AA（≥4.5:1）选定。
- 布局几何自动检查（`scripts/ui_geometry_check.py` + `tests/test_gui_layout.py`）：3 分辨率 ×
  3 状态无重叠/挤压/溢出；真实渲染截图见 `docs/images/ui/`，视觉核验记录见
  `docs/superpowers/plans/ui-visual-check.md`。

### 仓库

- `pyproject.toml`（版本单一来源、ruff、pytest）、GitHub Actions CI（Ubuntu + Windows）、
  `CITATION.cff`、`CODE_OF_CONDUCT.md`、Issue/PR 模板、`.gitignore` 精简、ruff 清零。
- README 徽章、界面截图、架构图、验证状态小节。

### 早期开发记录（合并自 Unreleased）

- 实时录音、端到端分析管线、4 个预训练模型（GPU NF4/FP16）、6 模态特征、差异化声刺激、
  SQLite 历史记录、便携模式、无头 CLI、双语 README、健壮性处理（OOM 降级、信号处理、
  worker 中断、配置校验）、模型加载改为主线程分阶段（修复 CUDA 跨线程段错误）、
  录音时长判定修复。
- 依赖修正：slab 1.8.2、panns-inference 0.1.1、transformers 4.51.3、emotion2vec v2.0.5、
  Qwen/Qwen3-1.7B、自行实现 PANNs Cnn10。

[0.5.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.5.0
[0.4.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.4.0
[0.3.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.3.0
[0.2.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.2.0
[0.1.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.1.0
