"""SEMamba generator: the MP-SENet layout with bidirectional Mamba blocks along time and frequency.

Source: https://github.com/RoyChao19477/SEMamba, commit
750ff5d8f0e112557ca039c268cdcc51e267f342, `models/` (MIT License, see LICENSE in this folder;
the code is based on MP-SENet and CMGAN, both MIT).
Paper:  Chao, Cheng, Lin, Lu, Tsao et al., "An Investigation of Incorporating Mamba for Speech
Enhancement", IEEE SLT 2024.

The files next to this one are the source's:
  generator.py, codec_module.py, lsigmoid.py, stfts.py     unchanged
  mamba_block.py    two patches marked `# [patch]`: Mamba, Block and RMSNorm come from
                    mamba_torch.py instead of the mamba_ssm package, and mamba_ssm's
                    weight initialisation is not applied
  mamba_torch.py    written here: Mamba in plain PyTorch (see its docstring), because mamba_ssm
                    needs CUDA to install and the submission must run on a CPU
Weights: the source's `ckpts/vd.pth` ("trained on both the VCTK-Demand and DNS-2020 corpora",
README), converted with tools/promote.py --state-dict ... --model semamba; the values are not
changed. Configuration = the source's ckpts/config.yaml (the defaults below).

This file adds the harness interface `enhance(noisy[B, T]) -> [B, T]` (se/models.py), the
same steps as the source's inference.py: scale the input to unit RMS, STFT with compressed
magnitude, generator, inverse STFT, scale back. Added for robustness (as for the other ported
models): an input that is next to silent is returned as it is, and an input longer than
`chunk_seconds` is processed in overlapping pieces.

Needs the package `einops`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .generator import SEMamba
from .stfts import mag_phase_istft, mag_phase_stft

SILENCE_RMS = 1e-5


class SEMambaDenoiser(nn.Module):
    def __init__(self, n_fft: int = 400, hop: int = 100, win: int = 400,
                 compress_factor: float = 0.3, hid_feature: int = 64, num_tfmamba: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 4, norm_epsilon: float = 1e-5,
                 beta: float = 2.0, chunk_seconds: float = 12.0, overlap_seconds: float = 1.0,
                 sample_rate: int = 16000):
        super().__init__()
        self.n_fft, self.hop, self.win, self.compress = n_fft, hop, win, compress_factor
        self.chunk = int(chunk_seconds * sample_rate)
        self.overlap = int(overlap_seconds * sample_rate)
        self.net = SEMamba({
            "stft_cfg": {"sampling_rate": sample_rate, "n_fft": n_fft, "hop_size": hop,
                         "win_size": win},
            "model_cfg": {"hid_feature": hid_feature, "compress_factor": compress_factor,
                          "num_tfmamba": num_tfmamba, "d_state": d_state, "d_conv": d_conv,
                          "expand": expand, "norm_epsilon": norm_epsilon, "beta": beta,
                          "input_channel": 2, "output_channel": 1}})

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] at its own level -> [B, T]."""
        n = x.shape[-1]
        x = F.pad(x, (0, (-n) % self.hop))           # the inverse STFT returns a multiple of hop
        scale = torch.sqrt(x.shape[-1] / x.pow(2).sum(-1, keepdim=True).clamp_min(1e-10))
        mag, pha, _ = mag_phase_stft(x * scale, self.n_fft, self.hop, self.win, self.compress)
        mag, pha, _ = self.net(mag, pha)
        y = mag_phase_istft(mag, pha, self.n_fft, self.hop, self.win, self.compress) / scale
        return y[..., :n]

    def _enhance_one(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        if n == 0 or float(x.double().pow(2).mean().sqrt()) < SILENCE_RMS:
            return x.clone()
        if n <= self.chunk:
            return self._run(x)
        window = torch.hann_window(self.chunk, periodic=False, device=x.device, dtype=x.dtype) + 1e-3
        out, weight = torch.zeros_like(x), torch.zeros_like(x)
        for start in range(0, n, self.chunk - self.overlap):
            seg = x[..., start:start + self.chunk]
            w = window[:seg.shape[-1]]
            out[..., start:start + seg.shape[-1]] += self._run(seg) * w
            weight[..., start:start + seg.shape[-1]] += w
            if start + self.chunk >= n:
                break
        return out / weight.clamp_min(1e-6)

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._enhance_one(x[None]) for x in noisy])
