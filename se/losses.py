"""Training losses: loss(model, noisy, clean) -> scalar tensor.

Register new losses in LOSSES; the config's "loss" picks one.
"""
from __future__ import annotations

import torch
from torch import nn


def log_mag_l1(model, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Course notebook: L1 between log(1+|S_hat|) and log(1+|S|) (spectral-mask models)."""
    estimate = model(noisy)
    target = model.spectrum(clean)
    return nn.functional.l1_loss(torch.log1p(estimate.abs()), torch.log1p(target.abs()))


LOSSES = {
    "log_mag_l1": log_mag_l1,
}
