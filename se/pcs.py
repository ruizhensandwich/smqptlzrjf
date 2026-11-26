"""Perceptual Contrast Stretching (PCS) on a 400-point STFT magnitude.

Method: Chao, Yu, Fu, Lu, Tsao, "Perceptual Contrast Stretching on Target Feature for Speech
Enhancement", Interspeech 2022: the log1p magnitude of each frequency band is multiplied by a
weight between 1.0 and 1.4 (largest where PESQ's band importance is largest), which raises
the contrast between loud and quiet time-frequency bins there.
Band weights: the PCS400 table of https://github.com/RoyChao19477/SEMamba (models/pcs400.py,
commit 750ff5d, MIT License, see se/arch/semamba/LICENSE), 201 bins of a 400-point STFT.

What differs from the source's post-processing (cal_pcs), and why (measured on four
validation sets, results/mix_study/pcs_report.md):
  - strength: the weights are 1 + strength * (w - 1). At strength 1 (the source) PESQ rises
    less than at 0.5 and many more files get worse; 0.5 is what the submission uses.
  - level: log1p depends on the signal's level, so a quiet copy of a file would be stretched
    less. The spectrum is scaled as if the waveform peaked at 1, stretched, and scaled back.
  - output level: the source sets the peak of the result to 1. Here the result gets the
    least-squares gain towards the unstretched output, so the level stays what it was.
  - it is applied to the magnitude that is being inverted anyway (Hann window, one inverse
    STFT) instead of to a finished waveform through a second (Hamming) STFT; the scores are
    the same within 0.002 PESQ.
PCS trades SNR for PESQ (about -3 to -8 dB at strength 0.5): the waveform moves away from the
clean one while the perceptual score rises.
"""
from __future__ import annotations

import torch

PCS400 = torch.ones(201, dtype=torch.float64)
PCS400[3:5] = 1.070175439
PCS400[5:8] = 1.182456140
PCS400[8:10] = 1.287719298
PCS400[10:110] = 1.4
PCS400[110:130] = 1.322807018
PCS400[130:160] = 1.238596491
PCS400[160:190] = 1.161403509
PCS400[190:201] = 1.077192982

EPS = 1e-8          # a peak or an energy below this counts as silence: no division by it


def pcs_magnitude(mag: torch.Tensor, strength: float) -> torch.Tensor:
    """mag [..., 201, frames] -> expm1((1 + strength * (w - 1)) * log1p(mag))."""
    w = PCS400.to(device=mag.device, dtype=mag.dtype)[:, None]
    return torch.expm1((1 + strength * (w - 1)) * torch.log1p(mag))


def pcs_stretch(mag: torch.Tensor, phase: torch.Tensor, y0: torch.Tensor, strength: float,
                window: torch.Tensor, hop: int = 100, peak: float = 1.0) -> torch.Tensor:
    """Waveform [B, T] of the stretched spectrum at the level of y0 (the waveform of mag, phase).

    A silent y0 comes back unchanged.
    """
    top = y0.abs().amax(-1)
    ok = top > EPS
    c = torch.where(ok, peak / top.clamp_min(EPS), torch.ones_like(top))[:, None, None]
    y1 = torch.istft(torch.polar(pcs_magnitude(mag * c, strength) / c, phase), window.numel(), hop,
                     window=window, length=y0.shape[-1])
    e = (y1 * y1).sum(-1)
    g = torch.where(e > EPS, (y0 * y1).sum(-1) / e.clamp_min(EPS), torch.ones_like(e))
    return torch.where(ok[:, None], y1 * g[:, None], y0)
