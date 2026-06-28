# 开发者指南

## 1. 开发环境搭建

```bash
conda create -n mandarin-emo-stim python=3.10.14 -y
conda activate mandarin-emo-stim
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

详见 [README.md](../README.md)。

## 2. 项目架构

```
src/
├── portable.py          # 便携数据重定向 + 国内镜像(HF_ENDPOINT)
├── config_loader.py     # 配置统一加载(缓存)
├── pipeline.py          # 端到端分析编排器
├── cli.py               # 无头CLI入口
├── audio/               # 录音/加载/VAD
├── models/              # 4个预训练模型封装 + 管理器 + 下载器
├── features/            # 韵律/物理/文本统计特征
├── fusion/              # 归一化 + 加权融合 + 象限判定
├── stimulus/            # 声刺激参数映射 + 合成 + 播放
├── storage/             # SQLite历史 + 导出
└── gui/                 # 构成主义PySide6界面
```

## 3. 数据流

```
音频 → loader(重采样16k单声道)
     → ASR(Paraformer, 内置VAD) → asr_text + timestamp
     → 6模态并行:
         ├ emotion2vec  → s_acoustic, a_acoustic
         ├ parselmouth  → s_prosody, a_prosody
         ├ PANNs CNN10  → s_paralang, a_paralang + events
         ├ librosa      → s_physical, a_physical + snr_db
         ├ Qwen3 LLM    → s_text_llm, a_text_llm
         └ jieba词表    → s_text_stat, a_text_stat
     → WeightedFusion(动态权重: 低SNR/低ASR/强副语言调整)
     → Quadrant(软判定隶属度)
     → StimulusGenerator(象限锚点 + 连续映射 → 波形合成)
     → 播放/保存 + 存入历史
```

## 4. 关键扩展点

### 4.1 调整融合权重

编辑 `config/settings.json` 的 `fusion_weights`（科研人员可在「设置 → 高级」修改）。权重和必须为 1。

### 4.2 扩展情感词表

`resources/dictionaries/` 下的词表为纯文本（每行一个词），可直接追加。程度副词格式为 `词<TAB>权重`。

```python
from src.features.text_stats import _DictLoader
_DictLoader.reload()  # 改完词表后重载
```

### 4.3 替换/降级模型

- `config/settings.json` 的 `models.emotion_backoff` 指定 emotion2vec 降级模型。
- `models.asr_device` 可设为 `"cpu"` 以把 ASR 卸到 CPU（显存紧张时）。
- `ModelManager.fallback_to_cpu()` 在 OOM 时一键切 CPU。

### 4.4 新增声刺激策略

在 `src/stimulus/strategies.py` 的 `compute_params` 中修改象限锚点或连续映射公式，参数定义见 `config/stimulus_params.json`。

## 5. 测试

```bash
pytest                          # 单元测试(默认跳过GPU/slow)
pytest -m "gpu"                 # 模型加载与推理(需GPU+已下载模型)
pytest -m "slow"                # GUI冒烟等耗时测试
pytest -m "gpu or slow"         # 全部集成测试
pytest -q                       # 全部(含集成,默认跳过标记项)
```

测试夹具音频：`tests/fixtures/mandarin_sample.wav`（Wikimedia Commons，Public Domain）。

## 6. 便携模式与开源合规

- **所有运行时数据**（模型、录音、历史、日志）位于 `portable_data/`，已被 gitignore 排除。
- **不要**在代码中硬编码绝对路径，统一用 `src.portable` 的常量。
- **不要**提交模型文件、个人录音或任何隐私数据。
- 测试夹具必须标注来源与协议（CC / Public Domain）。

## 7. 依赖版本说明

计划文档与实际安装存在以下偏差（已锁定于 requirements.txt）：

| 包 | 计划文档 | 实际 | 原因 |
|----|---------|------|------|
| slab | 0.2.3 | 1.8.2 | 0.2.3 不存在；1.8.2 提供所需 API |
| panns-inference | 0.1.3 | 0.1.1 | 0.1.3 不存在，0.1.1 为最高版 |
| transformers | 4.44.0 | 4.51.3 | 4.44 不支持 qwen3 架构 |
| 模型 revision | v2.0.0 | v2.0.5 | emotion2vec 有效版本 |
| Qwen3 模型名 | Qwen3-1.7B-Instruct | Qwen/Qwen3-1.7B | 前者不存在 |
| PANNs Cnn10 | panns_inference 自带 | 自行实现 | panns_inference 仅含 Cnn14 |
