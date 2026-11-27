"""Exponential moving average of a model's weights while it trains (train.ema_decay).

    average = decay * average + (1 - decay) * weights      after every optimizer step

The average only watches: it is never used for a gradient step and draws no random
numbers, so a run trains exactly the same with or without it. train.py scores it next
to the trained weights and writes it as epoch_NNNN_ema.pt / best_ema.pt.
"""
from __future__ import annotations

import torch


class WeightAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("ema decay must be within (0, 1)")
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:                                   # counters and the like: take the latest
                self.state[k].copy_(v)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "state": self.state}

    def load_state_dict(self, data: dict) -> None:
        self.decay = data["decay"]
        for k, v in data["state"].items():
            self.state[k].copy_(v)
