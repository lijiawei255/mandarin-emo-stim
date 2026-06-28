"""Cnn10 声学事件分类模型架构。

对应 PANNs (Pretrained Audio Neural Networks) 的 Cnn10 配置，
加载 ``Cnn10_mAP=0.380.pth`` checkpoint。

架构（由 checkpoint 结构反推确认）：
    - Spectrogram + LogmelFilterBank 前端（来自 torchlibrosa）
    - bn0 归一化
    - 4 个 ConvBlock（通道 64 -> 128 -> 256 -> 512）
    - fc1 (512->512) + fc_audioset (512->527)

注：``panns_inference`` 包仅提供 Cnn14 实现，故此处自行实现 Cnn10，
复用其 ``ConvBlock`` 与 ``torchlibrosa`` 前端组件，确保与 checkpoint 完全匹配。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from panns_inference.models import ConvBlock, init_bn


def init_layer(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Cnn10(nn.Module):
    """PANNs Cnn10 模型。"""

    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int, classes_num: int):
        super().__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window="hann", center=True, pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        # Cnn10: 4 个 ConvBlock，通道 64->128->256->512
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input_signal, _):
        # input_signal: (batch, data_length)
        x = self.spectrogram_extractor(input_signal)   # (B, 1, T, freq)
        x = self.logmel_extractor(x)                   # (B, 1, T, mel)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}
