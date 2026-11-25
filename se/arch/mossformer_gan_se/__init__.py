"""MossFormerGAN_SE_16K: the 16 kHz speech enhancement model of ClearerVoice-Studio.

Source: https://github.com/modelscope/ClearerVoice-Studio, commit
6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61, `clearvoice/clearvoice/models/mossformer_gan_se/`
(Apache License 2.0, LICENSE in this folder). Weights: https://huggingface.co/alibabasglab/
MossFormerGAN_SE_16K revision dcb1b20e1e44d435090072ab9b223c792ea1f624, `last_best_checkpoint.pt`
(Apache-2.0; "trained on large scale datasets including open-sourced and private data"; the paper
arXiv 2506.19398 describes DNS-4 and internal speech, DNS-4 / WHAM! / internal noise, synthetic
room responses on 30 % of the mixtures, SNR 0-15 dB). Converted to this repository's checkpoint
format with tools/promote.py (generator weights only; the optimizer state is dropped).

Files next to this one, as in the source except four marked lines (`# [patch]`):
  generator.py, mossformer.py, conv_module.py, fsmn.py, se_layer.py, get_layer_from_string.py
    - the training-only discriminator is not copied (its import is removed);
    - `packaging` is replaced by torch's own version class;
    - mossformer.py imports the vendored rotary embedding below.
  rotary_embedding_torch.py   rotary-embedding-torch 0.9.2 by Phil Wang (MIT, LICENSE_rotary),
                              copied so the result does not depend on that package's version.

This file adds the harness interface `enhance(noisy[B, T]) -> [B, T]` with the steps of the
source's `decode_one_audio_mossformergan_se_16k` (clearvoice/utils/decode_batch.py):
scale the input to unit RMS (over the whole input), pad to a multiple of the hop with the start
of the input, STFT (Hamming 400 / hop 100 / FFT 400, centred), magnitude^0.3 keeping the phase,
network, magnitude^(1/0.3), inverse STFT, undo the scale. Inputs longer than 10 s are processed
in 10 s windows with a 7.5 s stride, keeping the middle of each window, as in the source.
Added: an input that is next to silent is returned unchanged.

Needs the package `einops`.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .generator import SyncANet

SILENCE_RMS = 1e-5


class MossFormerGANSE16K(nn.Module):
    def __init__(self, win_len: int = 400, hop: int = 100, fft_len: int = 400,
                 compress: float = 0.3, window_seconds: float = 10.0, sample_rate: int = 16000):
        super().__init__()
        self.win_len, self.hop, self.fft_len, self.compress = win_len, hop, fft_len, compress
        self.window = int(window_seconds * sample_rate)
        self.net = SyncANet(num_channel=64, num_features=fft_len // 2 + 1)

    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        win = torch.hamming_window(self.win_len, periodic=True, device=x.device)
        return torch.stft(x, self.fft_len, self.hop, self.win_len, window=win, center=True,
                          onesided=True, return_complex=True)

    def _run(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """[B, T] at its own level, scale [B] (the whole input's 1 / RMS) -> [B, T]."""
        n = x.shape[-1]
        pad = int(math.ceil(n / self.hop)) * self.hop - n
        x = torch.cat([x, x[:, :pad]], dim=-1) * scale[:, None]
        spec = self._spec(x)
        mag, phase = spec.abs() ** self.compress, spec.angle()
        inp = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], 1)   # [B, 2, F, T]
        real, imag = self.net(inp.permute(0, 1, 3, 2))                          # [B, 1, T, F] each
        out = torch.complex(real, imag).permute(0, 1, 3, 2).squeeze(1)           # [B, F, T]
        out = torch.polar(out.abs() ** (1.0 / self.compress), out.angle())
        win = torch.hamming_window(self.win_len, periodic=True, device=x.device)
        y = torch.istft(out, self.fft_len, self.hop, self.win_len, window=win, center=True,
                        onesided=True)
        return y[:, :n] / scale[:, None]

    def _enhance_one(self, x: torch.Tensor) -> torch.Tensor:
        """[1, T] -> [1, T]."""
        n = x.shape[-1]
        if n == 0 or float(x.double().pow(2).mean().sqrt()) < SILENCE_RMS:
            return x.clone()
        scale = torch.sqrt(n / x.pow(2).sum(-1))
        if n <= self.window:
            return self._run(x, scale)
        stride = int(self.window * 0.75)
        give = (self.window - stride) // 2
        out = torch.zeros_like(x)
        i = 0
        while i + self.window <= n:
            y = self._run(x[:, i:i + self.window], scale)
            if i == 0:
                out[:, :self.window - give] = y[:, :-give]
            else:
                out[:, i + give:i + self.window - give] = y[:, give:-give]
            i += stride
        if i < n:
            start = i - give
            y = self._run(x[:, start:], scale)
            out[:, i:] = y[:, give:]
        return out

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._enhance_one(x[None]) for x in noisy])
