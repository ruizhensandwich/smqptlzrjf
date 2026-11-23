"""MP-SENet generator: parallel magnitude-mask and phase decoders over a compressed STFT.

Source: https://github.com/yxlu-0102/MP-SENet  (models/model.py, models/transformer.py,
utils.py, dataset.py; MIT License, Copyright (c) 2023 Yexin Lu)
Paper:  Lu, Ai, Ling, "MP-SENet: A Speech Enhancement Model with Parallel Denoising of
Magnitude and Phase Spectra", Interspeech 2023.

    MIT License

    Copyright (c) 2023 Yexin Lu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The network (dense encoder, time/frequency transformer blocks, mask decoder,
phase decoder) follows the source layer for layer, so the source's published
checkpoints (best_ckpt/g_best_dns, g_best_vb) load key for key; see
`source_axes` below for what they need. Changes made for this repository:

  * one class with the harness interface `enhance(noisy[B, T]) -> [B, T]`; the
    STFT, the RMS normalisation of the input (dataset.py / inference.py in the
    source) and the inverse STFT are inside the model, settings are constructor
    arguments instead of a config object;
  * `batch_first=True` in the attention and the GRU. The source feeds
    [batch, sequence, channels] tensors to sequence-first modules; its README
    recommends this fix for the attention when retraining. With
    `source_axes=True` the blocks instead see the axes the source's own
    checkpoints were trained on: there the module called time_transformer
    runs along frequency and freq_transformer along time (for one file at a
    time, which is how the source runs inference; in the source's training
    batches the sequence also ran across the files of the batch, which is not
    reproduced);
  * imaginary parts that are zero by construction (DC and Nyquist bins, first
    frame) are set to exactly zero before the phase is taken, see analysis().
    Otherwise float32 rounding noise decides between +pi and -pi there and the
    output changes with the machine or the input level;
  * any input length (zero padding to a whole number of hops, long inputs in
    overlapping chunks) so that denoise.py never fails on an odd file.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class FFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.linear(F.leaky_relu(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):                                   # [N, L, C]
        xt = self.norm1(x)
        x = x + self.attention(xt, xt, xt, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return self.norm3(x)


class TSTransformerBlock(nn.Module):
    """Attention along time (each frequency on its own), then along frequency.

    source_axes=True: time_transformer along frequency first, then freq_transformer
    along time, which is what the source's published weights were trained to do.
    """

    def __init__(self, channels: int, n_heads: int = 4, source_axes: bool = False):
        super().__init__()
        self.source_axes = source_axes
        self.time_transformer = TransformerBlock(channels, n_heads)
        self.freq_transformer = TransformerBlock(channels, n_heads)

    def forward(self, x):                                   # [B, C, T, F]
        b, c, t, f = x.shape
        if self.source_axes:
            x = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
            x = self.time_transformer(x) + x
            x = x.view(b, t, f, c).permute(0, 2, 1, 3).reshape(b * f, t, c)
            x = self.freq_transformer(x) + x
            return x.view(b, f, t, c).permute(0, 3, 2, 1)
        x = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).reshape(b * t, f, c)
        x = self.freq_transformer(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel upsampling along frequency by r."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int = 1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size)
        self.r = r

    def forward(self, x):
        out = self.conv(self.pad1(x))
        b, c, h, w = out.shape
        out = out.view(b, self.r, c // self.r, h, w).permute(0, 2, 3, 4, 1)
        return out.reshape(b, c // self.r, h, -1)


class DenseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size=(2, 3), depth: int = 4):
        super().__init__()
        self.dense_block = nn.ModuleList(
            nn.Sequential(
                nn.ConstantPad2d((1, 1, 2 ** i, 0), value=0.0),
                nn.Conv2d(channels * (i + 1), channels, kernel_size, dilation=(2 ** i, 1)),
                nn.InstanceNorm2d(channels, affine=True),
                nn.PReLU(channels))
            for i in range(depth))

    def forward(self, x):
        skip = x
        for layer in self.dense_block:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, channels: int, in_channel: int):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1)),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))
        self.dense_block = DenseBlock(channels)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))

    def forward(self, x):                                   # [B, 2, T, F] -> [B, C, T, F // 2]
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))


class MaskDecoder(nn.Module):
    def __init__(self, channels: int, bins: int, beta: float):
        super().__init__()
        self.dense_block = DenseBlock(channels)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels),
            nn.Conv2d(channels, 1, (1, 2)))
        self.lsigmoid = LearnableSigmoid2d(bins, beta=beta)

    def forward(self, x):
        x = self.mask_conv(self.dense_block(x))
        return self.lsigmoid(x.permute(0, 3, 2, 1).squeeze(-1))        # [B, F, T]


class PhaseDecoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dense_block = DenseBlock(channels)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))
        self.phase_conv_r = nn.Conv2d(channels, 1, (1, 2))
        self.phase_conv_i = nn.Conv2d(channels, 1, (1, 2))

    def forward(self, x):
        x = self.phase_conv(self.dense_block(x))
        x = torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return x.permute(0, 3, 2, 1).squeeze(-1)                       # [B, F, T]


class MPSENet(nn.Module):
    """Noisy waveform -> enhanced waveform; forward() works on compressed spectra.

    forward(noisy_mag, noisy_pha) -> (mag, pha, com): compressed magnitude
    (|S|^compress_factor) times a mask in (0, beta), a newly estimated phase,
    and both as a complex spectrum [B, F, T, 2]. The training loss
    (se/losses.py, mpsenet) uses analysis()/forward()/synthesis() directly.
    """

    def __init__(self, n_fft: int = 400, hop: int = 100, win: int = 400, dense_channel: int = 64,
                 num_tsblocks: int = 4, n_heads: int = 4, beta: float = 2.0,
                 compress_factor: float = 0.3, chunk_seconds: float = 6.0,
                 overlap_seconds: float = 1.0, sample_rate: int = 16000,
                 source_axes: bool = False):
        super().__init__()
        if not (0 < hop <= win <= n_fft and n_fft % 2 == 0 and dense_channel % n_heads == 0):
            raise ValueError("need 0 < hop <= win <= n_fft (even) and dense_channel % n_heads == 0")
        if not (0 < compress_factor <= 1 and 0 < overlap_seconds < chunk_seconds / 2):
            raise ValueError("need 0 < compress_factor <= 1 and 0 < overlap < chunk / 2")
        self.n_fft, self.hop, self.win, self.compress_factor = n_fft, hop, win, compress_factor
        self.chunk = int(chunk_seconds * sample_rate)
        self.overlap = int(overlap_seconds * sample_rate)
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.TSTransformer = nn.ModuleList(
            TSTransformerBlock(dense_channel, n_heads, source_axes) for _ in range(num_tsblocks))
        self.mask_decoder = MaskDecoder(dense_channel, n_fft // 2 + 1, beta)
        self.phase_decoder = PhaseDecoder(dense_channel)
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    # ---- spectra -----------------------------------------------------------
    def analysis(self, wave: torch.Tensor):
        """[B, T] -> compressed magnitude [B, F, frames], phase, complex [B, F, frames, 2]."""
        spec = torch.view_as_real(torch.stft(
            wave, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
            center=True, pad_mode="reflect", normalized=False, return_complex=True))
        re, im = spec[..., 0], spec[..., 1]
        # Not in the source: the DC and Nyquist bins of a real signal are real, and so is the
        # whole first frame (centred on sample 0 of the reflect-padded signal, so symmetric).
        # float32 FFT leaves rounding noise in those imaginary parts and its sign flips the
        # phase between +pi and -pi at random; with exact zeros the same input always gives
        # the same phase, on any machine.
        real_only = torch.zeros(im.shape[-2:], dtype=torch.bool, device=im.device)
        real_only[0], real_only[-1], real_only[:, 0] = True, True, True
        im = im.masked_fill(real_only, 0.0)
        mag = torch.sqrt(re.pow(2) + im.pow(2) + 1e-9)
        pha = torch.atan2(im + 1e-10, re + 1e-5)
        mag = torch.pow(mag, self.compress_factor)
        return mag, pha, torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    def synthesis(self, mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        mag = torch.pow(mag, 1.0 / self.compress_factor)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return torch.istft(com, self.n_fft, hop_length=self.hop, win_length=self.win,
                           window=self.window, center=True)

    @staticmethod
    def norm_factor(noisy: torch.Tensor) -> torch.Tensor:
        """Per-file gain that brings the noisy input to RMS 1 (undone after enhancement)."""
        power = noisy.pow(2).mean(dim=-1, keepdim=True)
        return torch.rsqrt(power.clamp_min(1e-10))

    def forward(self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor):
        x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)   # [B, 2, T, F]
        x = self.dense_encoder(x)
        for block in self.TSTransformer:
            x = block(x)
        mag = noisy_mag * self.mask_decoder(x)
        pha = self.phase_decoder(x)
        return mag, pha, torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    # ---- waveform in, waveform out ------------------------------------------
    def _run(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        padded = max(self.n_fft, math.ceil(n / self.hop) * self.hop)
        mag, pha, _ = self.analysis(F.pad(x, (0, padded - n)))
        mag, pha, _ = self(mag, pha)
        return self.synthesis(mag, pha)[..., :n]

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        n = noisy.shape[-1]
        scale = self.norm_factor(noisy)
        x = noisy * scale
        if n <= self.chunk:
            return self._run(x) / scale
        # long input: overlapping chunks, linear cross-fade (attention memory grows with T^2)
        out = torch.zeros_like(x)
        fade = torch.linspace(0.0, 1.0, self.overlap, device=x.device, dtype=x.dtype)
        for start in range(0, n - self.overlap, self.chunk - self.overlap):
            seg = x[..., start:start + self.chunk]
            w = torch.ones(seg.shape[-1], device=x.device, dtype=x.dtype)
            if start > 0:
                w[:self.overlap] = fade
            if start + self.chunk < n:
                w[-self.overlap:] = 1.0 - fade
            out[..., start:start + seg.shape[-1]] += self._run(seg) * w
        return out / scale
