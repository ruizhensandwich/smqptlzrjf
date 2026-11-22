"""Metric discriminator: predicts the (scaled) PESQ of an enhanced magnitude spectrum.

Source: https://github.com/yxlu-0102/MP-SENet  (models/discriminator.py, utils.py; MIT
License, Copyright (c) 2023 Yexin Lu), which follows the discriminator of CMGAN
(https://github.com/ruizhecao96/CMGAN, MIT License) and the MetricGAN idea
(Fu et al., ICML 2019). License text: see se/arch/mpsenet.py.

Used only while training (se/trainers.py); it is not part of the submitted weights.
Given the clean and an estimated compressed magnitude spectrum [B, F, T] it
outputs a score in (0, 1) that is trained to equal (PESQ - 1) / 3.5 of the
estimate (1 for clean against clean). The generator is then trained to make
this score 1, which gives it a differentiable stand-in for PESQ.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim: int = 16, in_channel: int = 2):
        super().__init__()
        layers, channels = [], in_channel
        for width in (dim, dim * 2, dim * 4, dim * 8):
            layers += [spectral_norm(nn.Conv2d(channels, width, (4, 4), (2, 2), (1, 1), bias=False)),
                       nn.InstanceNorm2d(width, affine=True), nn.PReLU(width)]
            channels = width
        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            spectral_norm(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            spectral_norm(nn.Linear(dim * 4, 1)), LearnableSigmoid1d(1))

    def forward(self, reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.stack((reference, estimate), dim=1))


DISCRIMINATORS = {"metric": MetricDiscriminator}
