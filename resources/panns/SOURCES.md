# PANNs 资源来源说明

## class_labels_indices.csv

- **来源**: AudioSet 本体标签表（Google Research），随 PANNs 官方仓库分发：
  https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv
- **协议**: AudioSet 本体与元数据为 **CC BY 4.0**（https://research.google.com/audioset/）。
- **内容**: 527 类声学事件的 `index,mid,display_name` 映射，PANNs 输出维度顺序与此一致。
- **随仓库分发原因**: 文件仅 14 KB，且 GitHub raw 在部分网络环境不可达；
  打包后可离线运行。`src/models/pann_model.py` 优先读取本目录副本。
