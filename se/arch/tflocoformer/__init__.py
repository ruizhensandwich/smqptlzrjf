"""TF-Locoformer: a time-frequency dual-path transformer with convolutional local modeling.

Source: https://github.com/merlresearch/tf-locoformer (standalone/, Apache-2.0, see LICENSE.md
in this folder). Paper: Saijo, Wichern, Germain, Pan, Le Roux, "TF-Locoformer: Transformer
with Local Modeling by Convolution for Speech Separation and Enhancement", IWAENC 2024.

tflocoformer_separator.py is the source's file. This file and that one reached this
repository through a teammate's package (noise_reduction_deep_v8.zip), not from the source
repository directly, so "unchanged" has not been checked against the source yet.

This file adds the harness interface `enhance(noisy[B, T]) -> [B, T]` (se/models.py) with the
settings of the source's DNS 2020 model: scale the input to unit standard deviation,
STFT 256/128, separator, inverse STFT, scale back. An input longer than 3 s is processed in
overlapping 3 s pieces.

Output level (`level`, added 2026-11-27). This DNS model returns the speech about 3.5 dB
quieter than it is in the input (it was trained with a scale-invariant loss): the gain that
brings its output to the clean reference has a median of 1.48 on the course train set (1st to
99th percentile 1.43 to 1.67). PESQ ignores level, SNR and a mix with other models do not.
`level="proj"` multiplies each file's output by the least-squares gain towards the NOISY input,
g = <noisy, y> / <y, y> (if y = c * speech and the noise is uncorrelated with the speech, this
is 1 / c; no clean reference is used), limited to [level_lo, level_hi] because the estimate
drifts upwards when much noise is left (-5 dB: +5 % in the median, up to 2.4 in the hard set),
and `level_const` when the output is silent. Validation SNR 9.12 -> 20.66 dB, PESQ unchanged
(results/mix_study/report.md). The default "none" is the model as published.

Needs the packages `rotary-embedding-torch` (and its dependency `beartype`) and `packaging`.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .tflocoformer_separator import TFLocoformerSeparator


class LocoDenoiser(nn.Module):
    def __init__(self, train_last_blocks=0, level="none", level_const=1.484, level_lo=1.35,
                 level_hi=1.65):
        super().__init__()
        if level not in ("none", "const", "proj"):
            raise ValueError("level must be 'none', 'const' or 'proj'")
        self.level, self.level_const = level, float(level_const)
        self.level_lo, self.level_hi = float(level_lo), float(level_hi)
        self.separator = TFLocoformerSeparator(
            num_spk=1, n_layers=6, emb_dim=128, norm_type="rmsgroupnorm", num_groups=4,
            tf_order="ft", n_heads=4, flash_attention=False,
            ffn_type=["swiglu_conv1d", "swiglu_conv1d"], ffn_hidden_dim=[384, 384],
            conv1d_kernel=4, conv1d_shift=1, dropout=0., eps=1e-5)
        if not 0 <= train_last_blocks <= len(self.separator.blocks):
            raise ValueError("train_last_blocks must be between zero and the number of blocks")
        self.separator.requires_grad_(False)
        self.separator.deconv.requires_grad_(True)
        if train_last_blocks:
            for block in self.separator.blocks[-train_last_blocks:]:
                block.requires_grad_(True)
        self.register_buffer("window", torch.hann_window(256), persistent=False)

    def _enhance(self, mixture):
        # Official conv/deconv run in float32; avoid mixed input/weight types under AMP.
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            return self._enhance_float(mixture.float())

    def _enhance_float(self, mixture):
        n = mixture.shape[-1]
        scale = mixture.std(dim=-1, keepdim=True).nan_to_num(0.).clamp_min(1e-8)
        x = F.pad(mixture/scale, (0, max(0, 256-n)))
        spectrum = torch.stft(x, 256, 128, window=self.window, return_complex=True)
        pred = self.separator(spectrum.transpose(1, 2))[:, 0].transpose(1, 2)
        recovered = torch.istft(pred, 256, 128, window=self.window, length=x.size(-1))
        return recovered[..., :n] * scale

    def enhance(self, mixture):
        out = self._enhance_any_length(mixture)
        if self.level == "none":
            return out
        if self.level == "const":
            return out * self.level_const
        x, y = mixture.double(), out.double()
        energy = (y * y).sum(-1)
        gain = (x * y).sum(-1) / energy.clamp_min(1e-30)
        gain = gain.nan_to_num(self.level_const).clamp(self.level_lo, self.level_hi)
        gain = torch.where(energy > 1e-10 * y.shape[-1], gain, torch.full_like(gain, self.level_const))
        return out * gain[..., None].to(out.dtype)

    def _enhance_any_length(self, mixture):
        # Bound compute for arbitrary-length input. Three-second course clips take one pass.
        window, overlap = 48000, 8000
        n = mixture.shape[-1]
        if n <= window:
            return self._enhance(mixture)
        out = torch.zeros_like(mixture)
        step = window - overlap
        fade = torch.linspace(0., 1., overlap, device=mixture.device, dtype=mixture.dtype)
        for start in range(0, n-overlap, step):
            segment = mixture[..., start:start+window]
            weight = segment.new_ones(segment.size(-1))
            if start:
                weight[:overlap] = fade
            if start + window < n:
                weight[-overlap:] = 1. - fade
            out[..., start:start+segment.size(-1)] += self._enhance(segment) * weight
        return out

