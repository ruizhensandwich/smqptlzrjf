"""Model registry.

Every model maps a noisy waveform batch [B, T] to an enhanced one with
`enhance(noisy) -> [B, T]` (same length). Spectral-mask models also expose
`spectrum(wave)` and `forward(noisy) -> estimated complex spectrum`, which the
spectral losses use. Register a new model in MODELS; its constructor keyword
arguments are the config's "model_args".
"""
from __future__ import annotations

import torch
from torch import nn


class MaskCNN(nn.Module):
    """The course notebook's example: 3 Conv1d layers over log|STFT| -> sigmoid mask.

    Frequency bins are the Conv1d channels, so each layer sees all bins of a few
    neighbouring frames. The mask multiplies the complex noisy spectrum (noisy
    phase kept). Parameter names match the notebook, so its weights load as is.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128,
                 hidden_channels=(64, 64), kernel_sizes=(5, 3, 1)):
        super().__init__()
        if not (n_fft > 0 and 0 < hop <= n_fft // 2):
            raise ValueError("need n_fft > 0 and 0 < hop <= n_fft // 2")
        if len(hidden_channels) != 2 or len(kernel_sizes) != 3 or \
                any(k <= 0 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("need 2 hidden channel counts and 3 positive odd kernel sizes")
        self.n_fft, self.hop = n_fft, hop
        bins = n_fft // 2 + 1
        c1, c2 = hidden_channels
        k1, k2, k3 = kernel_sizes
        self.mask_net = nn.Sequential(
            nn.Conv1d(bins, c1, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(),
            nn.Conv1d(c2, bins, kernel_size=k3, padding=k3 // 2),
            nn.Sigmoid(),
        )
        self.register_buffer("window", torch.hann_window(n_fft))

    def spectrum(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                          return_complex=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        spec = self.spectrum(noisy)                       # [B, bins, frames]
        mask = self.mask_net(torch.log1p(spec.abs()))
        return mask * spec

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        return torch.istft(self(noisy), n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=noisy.shape[-1])


MODELS: dict[str, type[nn.Module]] = {
    "mask_cnn": MaskCNN,
}


def build_model(name: str, model_args: dict | None = None) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](**(model_args or {}))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
