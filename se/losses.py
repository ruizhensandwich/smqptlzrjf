"""Training losses: loss(model, noisy, clean) -> scalar tensor.

Register new losses in LOSSES; the config's "loss" picks one.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def log_mag_l1(model, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Course notebook: L1 between log(1+|S_hat|) and log(1+|S|) (spectral-mask models)."""
    estimate = model(noisy)
    target = model.spectrum(clean)
    return nn.functional.l1_loss(torch.log1p(estimate.abs()), torch.log1p(target.abs()))


# --------------------------------------------------------------------------
# MP-SENet (source: https://github.com/yxlu-0102/MP-SENet train.py and models/model.py, MIT
# License, Copyright (c) 2023 Yexin Lu; see se/arch/mpsenet.py). Weights as in its train.py.
MPSENET_WEIGHTS = {"mag": 0.9, "pha": 0.3, "com": 0.1, "stft": 0.1, "time": 0.2}
METRIC_WEIGHT = 0.05        # generator's metric-discriminator term (se/trainers.py)


def anti_wrapping(x: torch.Tensor) -> torch.Tensor:
    """|x| folded into [0, pi]: phase differences that differ by 2 pi cost the same."""
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def phase_loss(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Instantaneous phase + group delay (difference along frequency) + instantaneous
    angular frequency (difference along time), each anti-wrapped. Phases are [B, F, T]."""
    ip = anti_wrapping(target - estimate).mean()
    gd = anti_wrapping(torch.diff(target, dim=1) - torch.diff(estimate, dim=1)).mean()
    iaf = anti_wrapping(torch.diff(target, dim=2) - torch.diff(estimate, dim=2)).mean()
    return ip + gd + iaf


def mpsenet_terms(model, noisy: torch.Tensor, clean: torch.Tensor):
    """Loss terms of a magnitude+phase model (analysis / forward / synthesis interface).

    Returns (terms, aux): terms are the unweighted losses, aux holds what the
    metric discriminator needs (normalised clean and enhanced waveforms and
    their compressed magnitudes; the enhanced magnitude is recomputed from the
    waveform, i.e. what a listener or PESQ actually gets).
    """
    scale = model.norm_factor(noisy)
    noisy, clean = noisy * scale, clean * scale
    noisy_mag, noisy_pha, _ = model.analysis(noisy)
    mag, pha, com = model(noisy_mag, noisy_pha)
    audio = model.synthesis(mag, pha)
    clean = clean[..., :audio.shape[-1]]
    clean_mag, clean_pha, clean_com = model.analysis(clean)
    mag_hat, _, com_hat = model.analysis(audio)
    terms = {"mag": F.mse_loss(clean_mag, mag),                 # compressed magnitude
             "pha": phase_loss(clean_pha, pha),                 # anti-wrapping phase
             "com": F.mse_loss(clean_com, com) * 2,             # complex spectrum
             "stft": F.mse_loss(com, com_hat) * 2,              # consistency: STFT(iSTFT(S)) = S
             "time": F.l1_loss(clean, audio)}                   # waveform
    return terms, {"clean": clean, "audio": audio, "clean_mag": clean_mag, "mag_hat": mag_hat}


def weighted(terms: dict) -> torch.Tensor:
    return sum(MPSENET_WEIGHTS[k] * v for k, v in terms.items())


def mpsenet(model, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """MP-SENet generator loss without the metric-discriminator term."""
    return weighted(mpsenet_terms(model, noisy, clean)[0])


LOSSES = {
    "log_mag_l1": log_mag_l1,
    "mpsenet": mpsenet,
}
