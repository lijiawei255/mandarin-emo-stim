# 测试夹具音频来源说明

## mandarin_sample.wav

- **来源**: Wikimedia Commons
  - 文件页: https://commons.wikimedia.org/wiki/File:Zh_dialect_Mandarin_sample.ogg
  - 直接下载: https://upload.wikimedia.org/wikipedia/commons/c/ca/Zh_dialect_Mandarin_sample.ogg
- **协议**: **Public Domain（公有领域）** —— 可自由使用、修改、分发，无需署名。
- **内容**: 中文普通话语音样本。
- **原始格式**: Ogg Vorbis（约 33 KB）。
- **本仓库处理**: 使用 ffmpeg 转换为 16 kHz 单声道 16-bit PCM WAV（约 166 KB），便于模型直接读取。
  ```
  ffmpeg -i mandarin_sample.ogg -ar 16000 -ac 1 mandarin_sample.wav
  ```
- **用途**: 作为 ASR、emotion2vec、韵律特征等模型推理的测试夹具，随仓库分发。

> 若需替换为更长或带特定情绪的语音，请将新音频（CC / Public Domain）放入本目录，
> 并在此文件补充来源与协议说明。
