"""xLSTM-SENet generator: the MP-SENet layout with xLSTM (mLSTM) blocks instead of transformers.

Source: https://github.com/NikolaiKyhne/xLSTM-SENet, commit
e195b64319a96f8289f98910b82cdd1f10f81d93, `models/` (MIT License, see LICENSE in this folder).
Paper:  Kühne, Østergaard, Jensen, Tan, "xLSTM-SENet: xLSTM for Single-Channel Speech
Enhancement", Interspeech 2025.

The files next to this one are the source's, unchanged except one import:
  generator.py, codec_module.py, xlstm_block.py, vision_lstm.py, vision_lstm_util.py,
  lsigmoid.py, stfts.py
  mamba_init.py   `_init_weights` of mamba_ssm (Apache-2.0, LICENSE_mamba), copied so that
                  xlstm_block.py does not need the mamba_ssm package (its only patched line).
No pretrained weights are used or loaded: the model starts from random weights and a
checkpoint of this repository is the only thing it reads.

This file adds the harness interface `enhance(noisy[B, T]) -> [B, T]` (se/models.py), the
same steps as the source's inference.py: scale the input to unit RMS, STFT with compressed
magnitude, generator, inverse STFT, scale back. Added for robustness: an input that is next
to silent is returned as it is, and an input longer than 4 s is processed in overlapping
4 s pieces (memory grows fast with length).

Needs the package `einops`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .generator import xLSTMSENet
from .stfts import mag_phase_istft, mag_phase_stft

SILENCE_RMS = 1e-5


class XLSTMSENet(nn.Module):
    def __init__(self, n_fft: int = 400, hop: int = 100, win: int = 400,
                 compress_factor: float = 0.3, hid_feature: int = 64, num_tfxlstm: int = 4,
                 embed_dim: int = 64, expansion_factor: int = 4, beta: float = 2.0,
                 chunk_seconds: float = 4.0, overlap_seconds: float = 1.0,
                 sample_rate: int = 16000):
        super().__init__()
        self.n_fft, self.hop, self.win, self.compress = n_fft, hop, win, compress_factor
        self.chunk = int(chunk_seconds * sample_rate)
        self.overlap = int(overlap_seconds * sample_rate)
        self.net = xLSTMSENet({
            "stft_cfg": {"sampling_rate": sample_rate, "n_fft": n_fft, "hop_size": hop,
                         "win_size": win},
            "model_cfg": {"hid_feature": hid_feature, "compress_factor": compress_factor,
                          "num_tfxlstm": num_tfxlstm, "embed_dim": embed_dim,
                          "expansion_factor": expansion_factor, "beta": beta,
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
