"""One training epoch, with or without a metric discriminator (used by train.py).

    Trainer            loss(model, noisy, clean) and one optimizer (the course example)
    MetricGANTrainer   MP-SENet recipe: the generator loss plus a discriminator that
                       learns to predict PESQ and is then used as a loss for the generator

Both keep their optimizer / scheduler (and discriminator) state for --resume.
The optimizer and the learning-rate schedule come from the config's "train"
settings: optimizer adam | adamw, scheduler plateau (halve when validation
PESQ stalls) | exponential (lr *= lr_gamma every epoch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from se.arch.metric_discriminator import DISCRIMINATORS
from se.ema import WeightAverage
from se.losses import LOSSES, METRIC_WEIGHT, MPSENET_WEIGHTS, mpsenet_terms, weighted
from se.metrics import pesq_pair

ROOT = Path(__file__).resolve().parent.parent


def make_optimizer(params, t: dict):
    cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[t["optimizer"]]
    return cls(params, lr=t["lr"], betas=tuple(t["adam_betas"]), weight_decay=t["weight_decay"])


def make_scheduler(optimizer, t: dict):
    if t["scheduler"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=t["lr_gamma"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t["lr_factor"], patience=t["lr_patience"],
        threshold=t["min_delta"], threshold_mode="abs", min_lr=t["min_lr"])


class Trainer:
    def __init__(self, model, cfg: dict, device, pool=None):
        self.model, self.device, self.t = model, device, cfg["train"]
        self.loss_fn = LOSSES[cfg["loss"]]
        self.optimizer = make_optimizer(model.parameters(), self.t)
        self.scheduler = make_scheduler(self.optimizer, self.t)
        # running average of the weights (se/ema.py); it watches only, training is unchanged
        self.ema = WeightAverage(model, self.t["ema_decay"]) if self.t["ema_decay"] > 0 else None

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            loss = self.loss_fn(self.model, noisy, clean)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            total += loss.item() * noisy.size(0)
        return {"train_loss": total / len(loader.dataset)}

    def validation_loss(self, noisy, clean) -> torch.Tensor:
        return self.loss_fn(self.model, noisy, clean)

    def _step(self, scheduler, val_pesq: float) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_pesq)
        else:
            scheduler.step()

    def end_epoch(self, val_pesq: float) -> None:
        """val_pesq = the score weights are selected by (train.select_by)."""
        self._step(self.scheduler, val_pesq)

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                **({"ema": self.ema.state_dict()} if self.ema is not None else {})}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])


class MetricGANTrainer(Trainer):
    """Per batch (as in MP-SENet's train.py):

    1. enhance the batch and measure the real PESQ-WB of each enhanced segment;
    2. discriminator step: D(clean, clean) -> 1 and D(clean, enhanced) -> (PESQ - 1) / 3.5
       (the second part is skipped when PESQ fails on a segment, e.g. one without speech);
    3. generator step: weighted magnitude, phase, complex, consistency and time losses
       plus METRIC_WEIGHT * (D(clean, enhanced) - 1)^2. The config's loss_weights replaces
       the weights it names (training only; the validation loss keeps the source's weights).
    """

    def __init__(self, model, cfg: dict, device, pool=None):
        super().__init__(model, cfg, device, pool)
        self.pool = pool
        self.weights = {**MPSENET_WEIGHTS, "metric": METRIC_WEIGHT, **cfg["loss_weights"]}
        if cfg["loss_weights"]:
            print(f"loss weights: {self.weights}", flush=True)
        self.disc = DISCRIMINATORS[cfg["discriminator"]]().to(device)
        if cfg["init_discriminator"]:       # published discriminator weights as the start
            start = torch.load(ROOT / cfg["init_discriminator"], map_location=device,
                               weights_only=True)
            self.disc.load_state_dict(start["state_dict"])
            print(f"discriminator starts from {cfg['init_discriminator']} ({start['source']})",
                  flush=True)
        self.optimizer_d = make_optimizer(self.disc.parameters(), self.t)
        self.scheduler_d = make_scheduler(self.optimizer_d, self.t)

    def _pesq(self, clean: torch.Tensor, audio: torch.Tensor):
        pairs = list(zip(clean.cpu().numpy(), audio.cpu().numpy()))
        scores = np.array(list(self.pool.map(pesq_pair, pairs)) if self.pool
                          else [pesq_pair(p) for p in pairs])
        if not np.all(np.isfinite(scores)):
            return None
        return torch.as_tensor((scores - 1) / 3.5, dtype=torch.float32, device=self.device)

    def train_epoch(self, loader) -> dict:
        self.model.train()
        self.disc.train()
        sums = {k: 0.0 for k in ("train_loss", "disc_loss", "metric", "mag", "pha", "com", "stft",
                                 "time", "train_pesq")}
        n_pesq = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            b = noisy.size(0)
            ones = torch.ones(b, device=self.device)
            terms, aux = mpsenet_terms(self.model, noisy, clean)
            target = self._pesq(aux["clean"], aux["audio"].detach())

            self.optimizer_d.zero_grad()
            loss_d = F.mse_loss(self.disc(aux["clean_mag"], aux["clean_mag"]).flatten(), ones)
            if target is not None:
                loss_d = loss_d + F.mse_loss(
                    self.disc(aux["clean_mag"], aux["mag_hat"].detach()).flatten(), target)
                sums["train_pesq"] += float(target.mean() * 3.5 + 1) * b
                n_pesq += b
            loss_d.backward()
            self.optimizer_d.step()

            self.optimizer.zero_grad()
            metric = F.mse_loss(self.disc(aux["clean_mag"], aux["mag_hat"]).flatten(), ones)
            loss = weighted(terms, self.weights) + self.weights["metric"] * metric
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)

            sums["train_loss"] += loss.item() * b
            sums["disc_loss"] += loss_d.item() * b
            sums["metric"] += metric.item() * b
            for k, v in terms.items():
                sums[k] += v.item() * b
        n = len(loader.dataset)
        out = {k: v / n for k, v in sums.items() if k != "train_pesq"}
        out["train_pesq"] = sums["train_pesq"] / n_pesq if n_pesq else float("nan")
        return out

    def end_epoch(self, val_pesq: float) -> None:
        super().end_epoch(val_pesq)
        self._step(self.scheduler_d, val_pesq)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "discriminator": self.disc.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "scheduler_d": self.scheduler_d.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.disc.load_state_dict(state["discriminator"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        self.scheduler_d.load_state_dict(state["scheduler_d"])


def build_trainer(model, cfg: dict, device, pool=None) -> Trainer:
    return (MetricGANTrainer if cfg["discriminator"] else Trainer)(model, cfg, device, pool)
