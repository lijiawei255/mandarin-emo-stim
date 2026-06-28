# 常见问题排查

## 运行异常与崩溃

### Q: 程序崩溃了怎么办？日志在哪？

A: 所有异常都会记录到 `portable_data/logs/app_YYYY-MM-DD.log`（按天滚动）。程序内置多重保护，绝大多数异常场景不会崩溃，而是弹窗提示并回滚状态：
- **Ctrl+C / 关闭窗口**：优雅退出（停止录音/播放/工作线程后退出），不会留下脏状态。
- **显存不足（OOM）**：自动降级到 CPU 推理（状态栏显示「CPU」），继续运行。
- **磁盘满 / 文件锁**：保存/导出失败时弹窗提示，不影响已分析的结果。
- **配置文件损坏**：启动时报明确的 ConfigError（指出缺失哪个键），而非晦涩的 KeyError。
- **未捕获异常**：sys.excepthook 兜底，弹窗 + 记录日志，避免静默崩溃。

### Q: 模型全部加载成功后程序闪退（无报错）？

A: 若曾遇到此问题（日志显示「LLM 模型就绪」后进程直接消失、无 Traceback），原因是模型加载在后台 QThread 线程中执行，CUDA 上下文跨线程冲突触发 C 层段错误。**已修复**：模型加载改为主线程分阶段加载，所有 CUDA 操作在主线程完成，避免跨线程崩溃。若仍出现类似闪退，请把 `portable_data/logs/` 最新日志反馈。

### Q: 加载模型时按 Ctrl+C 没反应？

A: 已修复。模型加载期间按 Ctrl+C 会触发优雅退出（最多等 200ms 响应）。若仍无响应，可能是模型下载卡在网络，可直接关闭终端窗口，程序会清理资源。

### Q: 模型加载到一半关闭窗口会崩溃吗？

A: 不会。关闭窗口时，程序会请求所有工作线程中断并等待退出（closeEvent），不会出现「QThread destroyed」崩溃。已加载的模型会被正确卸载释放显存。

## 安装与环境

### Q: 安装 bitsandbytes 报错？

A: bitsandbytes 仅支持 NVIDIA GPU + Windows/Linux。Apple Silicon 不支持，请改用 GGUF 版 Qwen3。Windows 下需 Python 3.10.x，3.11/3.12 可能不兼容。

### Q: scipy / praat-parselmouth 编译失败？

A: Windows 下安装 [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选 "Desktop development with C++"。

### Q: ffmpeg 未找到？

A: 从 https://ffmpeg.org/download.html 下载并加入 PATH，或 `conda install -c conda-forge ffmpeg`。

### Q: torch.cuda.is_available() 返回 False？

A:
1. 确认安装的是 CUDA 版 torch（`--index-url https://download.pytorch.org/whl/cu121`），而非 CPU 版。
2. 更新 NVIDIA 显卡驱动。
3. 显存需 ≥ 6GB。

## 模型下载

### Q: 模型下载失败 / 速度慢？

A:
- 国内网络下，ModelScope（Paraformer/emotion2vec）与 hf-mirror.com（Qwen3）应可直连。
- 若 HuggingFace 镜像仍不可达，确认 `HF_ENDPOINT=https://hf-mirror.com` 已设置（`src/portable.py` 会自动设置）。
- PANNs checkpoint 从 Zenodo 下载，偶有连接问题，下载器会自动重试 3 次。

### Q: 首次启动卡在「下载模型」很久？

A: 模型总计约 6GB（Qwen3 ~3.5GB、emotion2vec ~1.8GB、Paraformer ~0.9GB、PANNs ~0.03GB）。下载完成后即可完全离线运行。可先运行 `python scripts/download_models.py` 预下载。

## 运行

### Q: 显存不足（OOM）？

A:
- 关闭其他占用显存的程序（浏览器硬件加速、其他 GPU 应用）。
- 把 ASR 卸到 CPU：编辑 `config/settings.json` 的 `models.asr_device` 设为 `"cpu"`。
- 使用 emotion2vec 降级模型：`models.emotion_model` 改为 `iic/emotion2vec_plus_base`。

### Q: ASR 转写结果为空？

A:
- 检查音频是否包含有效人声（非纯静音/噪音）。
- 检查音频采样率（建议 16kHz；loader 会自动重采样）。
- 录音环境嘈杂（SNR<10dB）时界面会显示警告。

### Q: 生成的刺激声音太小/太大？

A: 刺激峰值固定限幅 -10dBFS（安全上限）。可用界面底部音量滑块调节。响度本身由唤醒度映射（高唤醒→更响）。

### Q: LLM 文本评分感觉不准？

A: 1.7B 小模型对精确数值评分能力有限。本工具以 6 模态加权融合为主，LLM 仅占文本语义支路的一部分（权重 0.30 negative / 0.10 arousal）。如需更准的文本情感，可替换为更大的 LLM（需相应调整显存预算）。

## 数据

### Q: 历史记录存在哪里？如何清除？

A: 全部位于 `portable_data/history/`（SQLite 数据库 + 音频文件）。删除该目录即清除。或在界面「文件 → 导出历史记录」后清空。

### Q: 如何在 Excel 中查看导出的 CSV？

A: 导出为 CSV 时已使用 UTF-8-BOM 编码，Excel 直接双击打开中文不乱码。

### Q: 录音功能可用吗？

A: 可用。点击左侧红色块「开始录音」即开始采集（界面实时显示已录制秒数，上限 60 秒），再次点击或点「停止/重置」结束，录制的音频会自动落盘并触发完整分析。从下拉框可选择麦克风设备。

注意事项：
- Windows 需在「系统设置 → 隐私 → 麦克风」中允许应用访问麦克风。
- 若「设备」下拉框显示「（无法枚举设备）」，通常是系统未授权麦克风访问或无可用输入设备。
- 录音环境嘈杂（SNR<10dB）时界面会显示黄色警告，结果可能不准确。
