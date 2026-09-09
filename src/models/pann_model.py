"""PANNs CNN10 副语言事件检测封装。

检测笑声/哭泣/尖叫/叹息/清嗓子/咳嗽等副语言事件，并按 n_contrib/a_contrib
聚合为 (s_paralang, a_paralang) 与检测到的事件列表。

checkpoint 来自 Zenodo（Cnn10_mAP=0.380.pth），AudioSet 标签表来自
audioset_tagging_cnn 仓库，二者均存放于 ``portable_data/models/panns/``
（绿色便携：不读写用户主目录）。标签表缺失时本模态显式降级
（``detect()`` 返回 ``degraded=True`` 并记 WARNING），不再静默返回中性分。
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

        # 确保 checkpoint 与标签表就位
        ckpt = portable.PANNS_DIR / "Cnn10_mAP=0.380.pth"
        if not ckpt.exists():
            from src.models.downloader import download_panns_checkpoint
            download_panns_checkpoint()
        if not (portable.PANNS_DIR / "class_labels_indices.csv").exists():
            from src.models.downloader import download_panns_labels
            try:
                download_panns_labels()
            except Exception as e:  # noqa: BLE001
                logger.warning("PANNs 标签表下载失败，将尝试回退目录: %s", e)

        # panns_inference.AudioTagging 默认用 Cnn14，与 Cnn10 checkpoint 不匹配，
        # 因此直接构建 Cnn10 模型并加载 Cnn10 checkpoint。
        model = Cnn10(sample_rate=32000, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        checkpoint = torch.load(str(ckpt), map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if "cuda" in str(self.device):
            model.to(self.device)
        self.model = model
        self._move_to_device = move_data_to_device
        self.labels = self._load_labels()

    @staticmethod
    def _load_labels() -> list[str]:
        """加载 AudioSet 标签列表。

        优先 ``portable_data/models/panns/class_labels_indices.csv``；
        兼容回退 ``~/panns_data/``（panns_inference 的旧默认位置）。
        """
        import csv
        from pathlib import Path
        candidates = [
            portable.PANNS_DIR / "class_labels_indices.csv",
            Path.home() / "panns_data" / "class_labels_indices.csv",
        ]
        csv_path = next((c for c in candidates if c.exists()), None)
        if csv_path is None:
            logger.warning("PANNs 标签表缺失（%s），副语言模态将降级", candidates[0])
            return []
        labels = []
        with csv_path.open("r", encoding="utf-8") as f:
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
            ``{"events": [{"label", "name_zh", "confidence"}], "s_paralang",
            "a_paralang", "degraded"}``。``degraded=True`` 表示标签表缺失或
            推理失败，本模态以中性分参与融合。
        """
        if not self.labels:
            logger.warning("PANNs 标签表不可用，副语言模态降级为中性分")
            return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5, "degraded": True}
        if len(y) == 0:
            return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5, "degraded": False}

        # PANNs 期望 (batch, samples)
        clip = y.astype(np.float32)
        try:
            import torch
            clip_t = self._move_to_device(clip[None, :], self.device)
            with torch.no_grad():
                output = self.model(clip_t, None)
            clipwise_output = output["clipwise_output"].data.cpu().numpy()
        except Exception as e:  # noqa: BLE001
            logger.warning("PANNs 推理失败，副语言模态降级: %s", e)
            return {"events": [], "s_paralang": 0.5, "a_paralang": 0.5, "degraded": True}

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
        return {"events": detected, "s_paralang": s_paralang, "a_paralang": a_paralang,
                "degraded": False}

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
