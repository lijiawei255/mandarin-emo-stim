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

`tests/test_robustness.py` 覆盖异常场景：配置缺失键、OOM 检测、checkpoint 截断、播放器线程安全、CLI 错误、并发 DB 写、边界值输入。

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
