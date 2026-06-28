# 更新日志

本项目版本变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] - 2026-06-28

### UI 重构：构成主义 → 浅色包豪斯（Light Bauhaus）

将可视化界面从五色构成主义（红/蓝/黄/黑厚边框色块）重构为浅色包豪斯风格：

**设计原则落地**
- **功能优先**：去除冗余装饰（厚色块、3px 黑框），保留核心控件与信息层级；改用 1px 细线分区卡片。
- **几何网格**：所有面板边距对齐到 8px 网格系统（ContentsMargins 统一 16/20px）。
- **高对比度**：浅色背景（白 `#FFFFFF` / 浅灰 `#F5F5F7`）配深色文字（`#1A1A1A`），确保可读性。
- **有限色彩**：主色仅 1 种（Bauhaus 蓝 `#1F5FA8`）+ 中性灰阶；状态色语义化区分（成功 `#2E7D32` 绿 / 警告 `#B8860B` 琥珀 / 错误 `#C62828` 红）。

**重构范围**
- `src/gui/styles.qss`：全面重写（全局浅色基底、按钮 hover 主色、进度条/滑块主色填充、菜单栏浅色细线、状态块浅灰、加载浮层浅色半透明遮罩）。
- `src/gui/widgets/`：metric_bar（主色细进度条）、modal_bars（深色标签+主色条）、status_block（QFrame 浅灰水平状态条，语义着色）、waveform_view（白底蓝线）、loading_overlay（浅色遮罩+语义色）全部浅色化。
- `src/gui/main_window.py`：去除内联硬编码颜色，改用 QSS 属性（`role`）驱动状态色；面板边距网格对齐；生成按钮主色强调。

**端到端回归测试**
- 视觉验证（大屏 1440×900 + 小屏 1280×720）：无文字重叠/截断/错位，中文渲染正常，五分区布局合理。
- 模拟用户操作流：加载浮层阶段切换/分析结果驱动UI/指标更新/波形加载/状态色语义 —— 全部响应正确。
- 自动化测试：17 项 GUI 测试 + 全量 91 项测试通过。

### 健壮性提升（异常场景不崩溃 + 状态可回滚 + 明确错误提示）

**用户中断与强制退出**
- SIGINT(Ctrl+C)/SIGTERM 优雅退出：注册信号处理器 + 200ms 定时器唤醒 Python 信号检查（解决模型加载期间 Ctrl+C 无响应）+ sys.excepthook 未捕获异常兜底。
- 工作线程中断支持：ModelLoadWorker/AnalysisWorker 在各阶段轮询 `isInterruptionRequested()`，中断时干净退出并发出 `interrupted` 信号（不再 QThread destroyed 崩溃）。
- closeEvent 安全退出：对所有运行中的 worker 调用 quit+wait，确保资源释放。

**并发与竞争条件**
- 分析期间禁用 record/upload 按钮 + 防重复 AnalysisWorker（避免孤儿线程崩溃与状态污染）。
- AudioPlayer 的 set_volume/pause/resume 加锁（修复音频回调线程的读写竞争）。
- recorder.record() 的 finally 不再吞异常（移除 finally 内 return）。
- HistoryDB 多线程并发写安全（已有锁保护，补充测试覆盖）。

**资源异常**
- OOM 自动降级 CPU：load_all 捕获 cuda.OutOfMemoryError/RuntimeError(MemoryError) 后自动切换 CPU 重试（fallback_to_cpu 不再是死代码）。
- GUI 文件 I/O 全保护：save_wav/sf.write/history.export 全部 try/except + 友好提示，磁盘满/锁文件不崩溃。
- PANNs checkpoint 完整性校验：修正大小阈值（30MB→[20,30]MB 合法区间），截断/损坏文件自动重下，下载后校验。

**异常输入与边界条件**
- 配置 schema 校验：load_settings 验证 settings.json 必需键，缺失时抛 ConfigError（清晰错误而非启动期 KeyError）。
- z-score 归一化处理 inf/极端值；融合全 0 分数不除零；刺激边界 valence/arousal(0/1) 不崩溃。

### 修复
- **GUI 启动崩溃**：`recorder.py` 与 `player.py` 顶层 `import sounddevice` 导致依赖缺失时 GUI 无法 import 启动。改为延迟导入（按需加载），缺失时录音/播放静默降级并友好提示，GUI 仍能正常启动。

### 文档（算法原理）
- **代码 docstring**：为全部核心算法模块补充心理学/声学原理说明（emotion_model 离散→V-A投影、prosody 韵律情感线索、weighted_fusion 动态权重鲁棒性、quadrant 软判定、strategies 音乐心理学映射、synthesizer 心理声学信号处理、physical Sethares粗糙度、normalizer z-score、text_stats 词典加权）。
- **docs/research_notes.md**：重写为完整算法原理总参考（Russell模型/6模态/动态权重/各模态算法/声刺激映射/归一化/局限性/10篇文献）。
- **README**（中英）：新增「算法原理」章节，指向 research_notes。

### 已新增（核心功能）
- **实时录音**：GUI「开始录音」按钮接通完整流程（开始/停止/实时计时/设备选择/到上限自动停止/录音→落盘→分析衔接）。
- **端到端分析管线**：音频 → VAD/ASR → 6 模态特征 → 加权融合 → 象限判定 → 标准化输出。
- **4 个预训练模型**（GPU NF4/FP16）：Paraformer-large ASR、emotion2vec_plus_large、PANNs CNN10（自行实现架构）、Qwen3-1.7B LLM。
- **6 模态特征提取**：声学情感、韵律（parselmouth）、副语言事件（PANNs）、物理声学（librosa）、文本语义（Qwen3 few-shot）、文本统计（jieba + 内置情感词表）。
- **差异化声刺激生成**：基于 Russell 四象限锚点的连续声学参数映射，软混合，安全限幅 -10dBFS，Haas 立体声。
- **构成主义风格 GUI**（PySide6 + pyqtgraph）：五块面布局、五色严格限定、直角 3px 黑框、多线程不阻塞。
- **SQLite 历史记录**：200 条上限、JSON/CSV 导出（UTF-8-BOM 兼容 Excel）、事务安全。
- **便携模式**：所有运行时数据集中于 `portable_data/`，国内镜像（hf-mirror.com）支持。
- **无头 CLI**：`python main.py --headless audio.wav` 或 `python -m src.cli`。
- **双语 README**（中文默认 / 英文可选）。
- **完整文档**：用户手册、开发者指南、科研依据、FAQ。

### 测试
- 92 项自动化测试全通过（单元 + GPU 模型 + 端到端 + GUI 冒烟）。
- 测试夹具音频（Wikimedia Commons，Public Domain 中文普通话）。

### 依赖版本（相对计划文档的修正）
- `slab`: 0.2.3 → 1.8.2（0.2.3 不存在）
- `panns-inference`: 0.1.3 → 0.1.1（0.1.3 不存在）
- `transformers`: 4.44.0 → 4.51.3（4.44 不支持 qwen3 架构）
- 重采样器：kaiser_best → soxr_hq（避免额外 resampy 依赖）

### 模型/配置修正
- emotion2vec revision: v2.0.0 → v2.0.5（有效版本）
- LLM 模型名: Qwen3-1.7B-Instruct → Qwen/Qwen3-1.7B（前者不存在）
- PANNs: 自行实现 Cnn10 架构（panns_inference 仅含 Cnn14）
- LLM prompt: 改用 few-shot + enable_thinking=False（提升小模型一致性）

### 性能（RTX 4000 Ada 12GB 实测）
- 模型加载：~21s
- 单次完整分析：~2.5s
- 显存占用：~2.8GB（allocated）/ 12GB total
