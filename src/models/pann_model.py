"""PANNs CNN10 副语言事件检测封装。

检测笑声/哭泣/尖叫/叹息/清嗓子/咳嗽等副语言事件，并按 n_contrib/a_contrib
聚合为 (s_paralang, a_paralang) 与检测到的事件列表。

checkpoint 来自 Zenodo（Cnn10_mAP=0.380.pth），存放于 ``portable_data/models/panns/``。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src import portable
from src.config_loader import load_settings
from src.fusion.normalizer import clip01

logger = logging.getLogger("mandarin_emo_stim.panns")

# 目标副语言事件与 AudioSet 标签映射（标签名 -> n_contrib, a_contrib）
TARGET_EVENTS = {
    "Laughter": {"name_zh": "笑声", "n_contrib": -0.5, "a_contrib": 0.3},
    "Crying, sobbing": {"name_zh": "哭泣", "n_contrib": 0.7, "a_contrib": 0.5},
    "Screaming": {"name_zh": "尖叫", "n_contrib": 0.3, "a_contrib": 1.0},
    "Sigh": {"name_zh": "叹息", "n_contrib": 0.4, "a_contrib": -0.3},
    "Breathing": {"name_zh": "呼吸", "n_contrib": 0.4, "a_contrib": -0.3},
    "Throat clearing": {"name_zh": "清嗓子", "n_contrib": 0.2, "a_contrib": 0.2},
    "Cough": {"name_zh": "咳嗽", "n_contrib": 0.0, "a_contrib": 0.0},
}


class PANNModel:
    """PANNs CNN10 副语言事件检测封装。"""

    def __init__(self, device: str = "cuda", model: Any = None):
        self.device = device
        self.threshold = float(load_settings()["thresholds"]["paralang_confidence_threshold"])
        if model is not None:
            self.model = model
            self.labels = self._load_labels()
        else:
            self._load()
        logger.info("PANNs 模型就绪（device=%s）", self.device)

    def _load(self) -> None:
        import torch
        from panns_inference.pytorch_utils import move_data_to_device
        from src.models.panns_cnn10 import Cnn10  # 本地实现的 Cnn10 架构

        # 确保 checkpoint 就位
        ckpt = portable.PANNS_DIR / "Cnn10_mAP=0.380.pth"
        if not ckpt.exists():
            from src.models.downloader import download_panns_checkpoint
            download_panns_checkpoint()

        # panns_inference.AudioTagging 默认用 Cnn14，与 Cnn10 checkpoint 不匹配，
        # 因此直接构建 Cnn10 模型并加载 Cnn10 checkpoint。
        model = Cnn10(sample_rate=32000, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        checkpoint = torch.load(str(ckpt), map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if "cuda" in str(self.device):
            model.to(self.device)
            model = torch.nn.DataParallel(model)
        self.model = model
        self._move_to_device = move_data_to_device
        self.labels = self._load_labels()

    @staticmethod
    def _load_labels() -> list[str]:
        """加载 AudioSet 标签列表。"""
        import csv
        from pathlib import Path
        csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
        if not csv_path.exists():
            return []
        labels = []
        with csv_path.open("r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 3:
                    labels.append(row[2])
        return labels

    def detect(self, y: np.ndarray, sr: int = 32000) -> dict[str, Any]:
        """检测音频中的副语言事件。

        Args:
            y: 音频波形。若采样率非 32000，需调用方先重采样。
            sr: 采样率（PANNs 期望 32000）。

        Returns:
            ``{"events": [{"label", "name_zh", "confidence"}], "s_paralang", "a_paralang"}``
        """
        if len(y) == 0 or not self.labels:
            return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5}

        # PANNs 期望 (batch, samples)
        clip = y.astype(np.float32)
        try:
            import torch
            clip_t = self._move_to_device(clip[None, :], self.device)
            with torch.no_grad():
                output = self.model(clip_t, None)
            clipwise_output = output["clipwise_output"].data.cpu().numpy()
        except Exception as e:  # noqa: BLE001
            logger.warning("PANNs 推理失败: %s", e)
            return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5}

        scores = np.asarray(clipwise_output[0])

        # 收集目标事件
        detected = []
        for label, meta in TARGET_EVENTS.items():
            if label in self.labels:
                idx = self.labels.index(label)
                conf = float(scores[idx])
                if conf >= self.threshold:
                    detected.append({
                        "label": label,
                        "name_zh": meta["name_zh"],
                        "confidence": conf,
                        "n_contrib": meta["n_contrib"],
                        "a_contrib": meta["a_contrib"],
                    })

        s_paralang, a_paralang = self._aggregate(detected)
        return {"events": detected, "s_paralang": s_paralang, "a_paralang": a_paralang}

    @staticmethod
    def _aggregate(events: list[dict[str, Any]]) -> tuple[float, float]:
        """按文档 3.2 节（3）公式聚合。"""
        s = 0.5
        a = 0.5
        total_conf = 0.0
        for ev in events:
            conf = ev["confidence"]
            s += conf * ev["n_contrib"]
            a += conf * ev["a_contrib"]
            total_conf += conf
        if total_conf > 0:
            s = s / (1.0 + total_conf)
            a = a / (1.0 + total_conf)
        return clip01(s), clip01(a)
