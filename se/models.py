"""Model registry.

Every model maps a noisy waveform batch [B, T] to an enhanced one with
`enhance(noisy) -> [B, T]` (same length). Spectral-mask models also expose
`spectrum(wave)` and `forward(noisy) -> estimated complex spectrum`, which the
spectral losses use. Register a new model in MODELS; its constructor keyword
arguments are the config's "model_args". Models taken from open-source projects live
in se/arch/, one file per model, with their source and license at the top.
"""
from __future__ import annotations

import torch
from torch import nn

from se.arch.mpsenet import MPSENet


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


class Ensemble(nn.Module):
    """Several models in one weight file; the output is the weighted mean of their outputs.

    members: [{"model": name, "model_args": {...}, "weight": w}, ...]. The weights are
    fixed numbers stored in the checkpoint's model_args (the same for every input file) and
    are used as given divided by their sum. Each member runs on the input on its own,
    exactly as it would alone.
    """

    def __init__(self, members: list[dict]):
        super().__init__()
        if len(members) < 2 or any(m["model"] == "ensemble" or float(m["weight"]) <= 0
                                   for m in members):
            raise ValueError("an ensemble needs two or more members that are not ensembles, "
                             "each with a weight > 0")
        self.members = nn.ModuleList(build_model(m["model"], m.get("model_args"))
                                     for m in members)
        total = sum(float(m["weight"]) for m in members)
        self.weights = [float(m["weight"]) / total for m in members]

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        out = None
        for w, member in zip(self.weights, self.members):
            y = w * member.enhance(noisy)
            out = y if out is None else out + y
        return out


def _xlstm_senet(**model_args) -> nn.Module:
    from se.arch.xlstm_senet import XLSTMSENet      # imported on use: needs the einops package
    return XLSTMSENet(**model_args)


MODELS: dict[str, type[nn.Module]] = {
    "mask_cnn": MaskCNN,
    "mpsenet": MPSENet,          # se/arch/mpsenet.py (MP-SENet, MIT License)
    "xlstm_senet": _xlstm_senet,  # se/arch/xlstm_senet/ (xLSTM-SENet, MIT License)
    "ensemble": Ensemble,        # weighted mean of member models (tools/make_ensemble.py)
}


def build_model(name: str, model_args: dict | None = None) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](**(model_args or {}))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
