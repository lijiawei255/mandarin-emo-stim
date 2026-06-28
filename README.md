<div align="center">

# Mandarin-EmoStim

**全离线中文普通话语音情感分析与个性化声刺激生成桌面科研工具**

[English](./README.en.md) | **中文**

</div>

---

## 简介

Mandarin-EmoStim 实现了一条完整的本地闭环：**说话 → 情绪量化 → 差异化声刺激**。

只需一个麦克风，工具会：

1. 采集中文普通话语音；
2. 通过 4 条声学支路 + 2 条文本支路（共 6 模态）的预训练模型提取特征；
3. 多特征加权融合，输出可解释、可复现的量化情绪指标（Negative / Valence / Arousal / 情绪象限）；
4. 依据 Russell 情绪环模型，生成**差异化**的个性化可听声刺激（WAV / 实时播放）。

> ⚠️ 本工具为科研探索用途，不构成医疗建议或治疗手段。特殊人群（癫痫史、严重心脏病、重度抑郁症正在接受治疗者）建议在专业人员指导下使用。

## 特性

- **全离线**：所有模型与运行时数据下载完成后无需任何网络连接。
- **多模态融合**：声学情感（emotion2vec）、韵律学（parselmouth）、副语言事件（PANNs）、物理声学（librosa）、文本语义（Qwen3）、文本统计（jieba）。
- **差异化声刺激**：基于四象限锚点的连续声学参数映射，软混合避免硬切换突兀。
- **构成主义风格 GUI**：PySide6 + pyqtgraph，几何块面、强对比纯色。
- **绿色便携**：所有数据集中在项目目录下的 `portable_data/`，删除即清除，不污染系统。
- **开源合规**：Apache License 2.0，与全部上游模型/依赖协议兼容。

## 硬件要求（当前版本：Windows + NVIDIA GPU）

| 项目 | 最低 | 推荐 |
|------|------|------|
| 显卡 | 6GB 显存 NVIDIA（CUDA） | 8GB+ 显存 |
| 内存 | 8GB | 16GB |
| 系统 | Windows 10/11 | Windows 11 |

> 当前版本仅针对 **Windows + NVIDIA 显卡**测试通过。纯 CPU 模式与 Apple Silicon 的代码路径已设计，但未在本版本验证。

## 快速开始

### 1. 创建环境

```bash
conda create -n mandarin-emo-stim python=3.10.14 -y
conda activate mandarin-emo-stim
```

### 2. 安装依赖

> Python **必须** 3.10.x（3.11/3.12 可能与 bitsandbytes 不兼容）。

```bash
# PyTorch（NVIDIA / CUDA 12.1）
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements.txt
```

**Windows 前置**：需要 [ffmpeg](https://ffmpeg.org/download.html)（加入 PATH）；若 scipy/parselmouth 编译失败，安装 [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（勾选 "Desktop development with C++"）。

**国内用户**建议配置 pip 镜像加速：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 首次运行

```bash
python main.py
```

首次启动会：硬件检测 → 检查并下载模型（约 6GB）→ 加载 → 进入就绪状态。

> 国内网络环境下，模型自动从国内源下载（Paraformer/emotion2vec 走 ModelScope，Qwen3 走 hf-mirror.com，PANNs 走 Zenodo）。

## 无头运行（不启动 GUI）

```bash
python -m src.stimulus.cli --audio path/to/test.wav
```

详见开发者文档 `docs/developer_guide.md`。

## 项目结构

```
mandarin-emo-stim/
├── config/            # 配置文件（融合权重、情绪映射、刺激参数）
├── src/
│   ├── audio/         # 录音/加载/VAD
│   ├── models/        # 4 个预训练模型封装 + 下载器 + 管理器
│   ├── features/      # 韵律/物理/文本统计特征
│   ├── fusion/        # 归一化 + 加权融合 + 象限判定
│   ├── stimulus/      # 声刺激参数映射 + 合成 + 播放
│   ├── storage/       # SQLite 历史记录 + 导出
│   └── gui/           # 构成主义风格界面
├── resources/         # 字体/图标/情感词表
├── tests/             # pytest 测试
├── docs/              # 文档
└── portable_data/     # 运行时生成，gitignore 排除（模型/录音/日志）
```

## 声音安全声明

- 生成刺激音频的峰值响度限制为 **-10 dBFS**（约合 70–75 dB SPL，正常交谈音量），无听力损伤风险。
- 建议使用耳机以获得最佳体验（非必须，扬声器同样安全）。

## 开源协议

[Apache License 2.0](./LICENSE)。所有上游模型与依赖协议兼容。

## 致谢

本项目使用了以下开源成果：emotion2vec、Paraformer（FunASR）、PANNs、Qwen3、praat-parselmouth、librosa、slab、jieba、PySide6、pyqtgraph 等。详细参考文献见 `docs/research_notes.md`。
