# 更新日志

本项目版本变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] - 2026-06-28

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
