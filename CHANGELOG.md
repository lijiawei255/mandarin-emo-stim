# 更新日志

本项目版本变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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

[0.1.0]: https://github.com/lijiawei255/mandarin-emo-stim/releases/tag/v0.1.0
