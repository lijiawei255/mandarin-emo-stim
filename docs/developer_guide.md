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
└── gui/                 # 暖奶油主题 PySide6 界面（theme.py 为唯一调色板）
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

编辑 `config/settings.json` 的 `fusion_weights`。权重和必须为 1。这些权重是启发式默认值（未经学习），
改动后用 `scripts/evaluate.py emotion` 验证效果。

### 4.1.1 韵律基准值

`config/prosody_norms.json` 由 `scripts/evaluate.py calibrate` 在 AISHELL-3 上实测生成（含 mixed / female /
male 三组）。`src/fusion/normalizer.load_prosody_stats(group=...)` 可按性别选用；文件缺失时回退到
`LEGACY_PROSODY_STATS`（早期凭空设定值，仅兜底）。

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

`src/stimulus/strategies.py` 中每个象限的连续映射拆为 `_pulse_rate / _base_freq / _noise_ratio`，
`compute_params` 按四象限隶属度对它们加权（软混合）；谐和结构取主象限锚点。参数区间见
`config/stimulus_params.json`。改动后 `tests/test_scientific_behavior.py` 的方向性与连续性断言必须仍通过，
并在 `docs/research_notes.md` 标注证据等级。若要实现 ISO 原则式的「先匹配再引导」，见 research_notes §4.2.1。

### 4.5 界面主题

`src/gui/theme.py` 的 `PALETTE` 是唯一调色板来源，`styles.qss.tmpl` 用双花括号占位符引用，
`build_qss()` 渲染；控件内联样式通过 `theme.inline(...)` 生成。`tests/test_gui_smoke.py` 守护 `src/gui`
下不得出现写死的十六进制色值。改完主题后跑 `pytest -m slow`（布局几何）并用
`python scripts/gui_screenshot.py` 在真实平台截图目视核验（offscreen 无中文字体）。

## 5. 测试与静态检查

```bash
ruff check .                    # 静态检查（CI 第一道门）
pytest                          # 单元 + 科学行为回归（默认跳过 GPU/slow）
pytest -m slow                  # GUI 冒烟 + 布局几何检查（offscreen 可跑）
pytest -m gpu                   # 模型加载与端到端（需 GPU + 已下载模型）
```

| 文件 | 覆盖 |
|------|------|
| `tests/test_scientific_behavior.py` | 合成受控信号：各模块对 F0/语速/HNR/粗糙度的响应方向、象限角点与软过渡、Q2/Q3 干预分支方向、跨象限参数连续性。**改动方法学前先看它** |
| `tests/test_pipeline_degradation.py` | 任一模态失败时降级为中性分并列入 `degraded_modalities`，不崩溃 |
| `tests/test_gui_layout.py` | 3 分辨率 × 3 状态下无重叠/挤压/溢出/压扁（`scripts/ui_geometry_check.py`） |
| `tests/test_robustness.py` | 配置校验、OOM 判定、checkpoint 校验、SIGINT 定时器、PANNs 标签降级、LLM greedy、并发写库 |

测试夹具音频：`tests/fixtures/mandarin_sample.wav`（Wikimedia Commons，Public Domain）。

### 5.1 三层验证评测（`scripts/evaluate.py`）

```bash
pip install -r requirements-eval.txt
python scripts/evaluate.py calibrate   # AISHELL-3 → config/prosody_norms.json
python scripts/evaluate.py neutral     # AISHELL-3 → CER + 中性语音输出分布
python scripts/evaluate.py emotion     # CSEMOTIONS → 象限 / 方向 / 消融 / 动态权重
python scripts/evaluate.py report      # 汇总 Markdown 表格 → portable_data/eval/results/
```

数据运行时按需下载到 `portable_data/eval/`（gitignore），仓库不分发音频。每条语音的模态分数
缓存为 JSON，消融与动态权重开关在缓存上离线重算。结果与解读见 `docs/evaluation.md`。
**改动融合权重 / 锚点 / 特征后请重跑 `emotion` 并更新 evaluation.md。**

## 6. 健壮性设计（异常处理约定）

项目遵循「任何异常场景下程序不崩溃，状态可回滚，向用户输出明确错误信息，并记录详细日志」的原则。二次开发请遵循：

### 6.1 中断与退出
- **Ctrl+C / SIGTERM**：`app.py::_install_crash_handlers` 注册信号处理器，优雅退出（触发 closeEvent 清理）。靠 200ms `QTimer` 唤醒 Python 检查信号（否则 Qt 事件循环不处理 SIGINT）。
- **未捕获异常**：`sys.excepthook` 兜底，弹窗 + `logger.critical` 记录。
- **工作线程中断**：所有 `QThread` worker 在各阶段轮询 `isInterruptionRequested()`，中断时 `raise InterruptedError` 干净退出并发出 `interrupted` 信号。**新增 worker 时务必检查中断。**
- **closeEvent**：对运行中的 worker 调 `quit()` + `wait(timeout)`，否则 `QThread: Destroyed while running` 崩溃。

### 6.2 并发
- **防重复触发**：耗时操作（分析）启动时禁用相关按钮，结束时（成功/失败/中断三路）恢复。`_start_analysis` 检查 `isRunning()` 拒绝并发。
- **音频回调线程安全**：`AudioPlayer` 的 `set_volume`/`pause`/`resume`/`stop` 都在 `self._lock` 内修改状态（音频回调线程会读这些字段）。**新增共享状态务必加锁。**
- **worker 强引用**：所有 worker 存为 `self.xxx_worker`（防 GC 导致线程被销毁）。

### 6.3 资源异常
- **OOM 降级**：`ModelManager.load_all` 捕获 `cuda.OutOfMemoryError`/`RuntimeError`(memory) 后 `_switch_device_to_cpu()` 重试。`_is_oom_like` 判定可降级异常。
- **文件 I/O**：GUI 线程的所有 `sf.write`/`save_wav`/`export` 用 `try/except OSError` 包裹，失败弹窗提示（磁盘满/锁文件），不崩溃。
- **模型下载**：`downloader.py` 校验文件完整性（PANNs checkpoint 大小窗口 [20,30]MB），截断/损坏自动重下；3 次指数退避重试。

### 6.4 异常输入
- **配置校验**：`config_loader.load_settings` 验证 `settings.json` 必需键（见 `_REQUIRED_SETTINGS`），缺失抛 `ConfigError`。**新增配置项时同步更新该校验表。**
- **数值边界**：`zscore_normalize` 处理 inf/极端值；`WeightedFusion._normalize` 防除零；刺激生成钳制 valence/arousal 到 [0,1]。
- **空音频**：`recorder.stop()` 返回空数组时上层判空提示；`prosody/physical` 对静音/极短音频返回中性默认值。
- **模态降级**：`AnalysisPipeline` 中任一模态步骤异常（模型推理失败、PANNs 标签缺失等）→ 该模态置 (0.5, 0.5)
  并记入 `degraded_modalities`；仅「加载音频」与「融合」失败会抛出。GUI 警告标签与 CLI 都会列出降级模态。


## 7. 便携模式与开源合规

- **所有运行时数据**（模型、录音、历史、日志）位于 `portable_data/`，已被 gitignore 排除。
- **不要**在代码中硬编码绝对路径，统一用 `src.portable` 的常量。
- **不要**提交模型文件、个人录音或任何隐私数据。
- 测试夹具必须标注来源与协议（CC / Public Domain）。

## 8. 依赖版本说明

计划文档与实际安装存在以下偏差（已锁定于 requirements.txt）：

| 包 | 计划文档 | 实际 | 原因 |
|----|---------|------|------|
| slab | 0.2.3 | 1.8.2 | 0.2.3 不存在；1.8.2 提供所需 API |
| panns-inference | 0.1.3 | 0.1.1 | 0.1.3 不存在，0.1.1 为最高版 |
| transformers | 4.44.0 | 4.51.3 | 4.44 不支持 qwen3 架构 |
| 模型 revision | v2.0.0 | v2.0.5 | emotion2vec 有效版本 |
| Qwen3 模型名 | Qwen3-1.7B-Instruct | Qwen/Qwen3-1.7B | 前者不存在 |
| PANNs Cnn10 | panns_inference 自带 | 自行实现 | panns_inference 仅含 Cnn14 |
