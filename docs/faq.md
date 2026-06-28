# 常见问题排查

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
