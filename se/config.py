"""Experiment configuration: one JSON file per experiment under configs/.

A config names the model and its settings, the loss, and the training
settings. Missing keys take the defaults below (the course notebook's
example), unknown keys are an error so that a typo never goes unnoticed.

    {"model": "mask_cnn",
     "model_args": {"n_fft": 512, "hop": 128, "hidden_channels": [64, 64], "kernel_sizes": [5, 3, 1]},
     "loss": "log_mag_l1",
     "train": {"seed": 42, "batch_size": 16, "lr": 0.001, ...}}

`--set train.lr=0.0005` style overrides go through apply_overrides.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

TRAIN_DEFAULTS = {
    "seed": 42,
    "batch_size": 16,
    "lr": 1e-3,
    "adam_betas": [0.9, 0.999],
    "weight_decay": 0.0,
    "max_epochs": 200,        # longest run
    "min_epochs": 30,         # no early stop before this
    "patience": 20,           # epochs without a MIN_DELTA validation PESQ gain before stopping
    "min_delta": 0.001,
    "lr_factor": 0.5,         # ReduceLROnPlateau on validation PESQ
    "lr_patience": 5,
    "min_lr": 1e-5,
    "save_epochs": [10, 20, 40, 80],   # extra checkpoints kept for listening comparisons
    "num_workers": 0,
}

DEFAULTS = {
    "model": "mask_cnn",
    "model_args": {},
    "loss": "log_mag_l1",
    "train": TRAIN_DEFAULTS,
    "note": "",
}


def _merge(base: dict, new: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key {where}{k!r} (known: {sorted(base)})")
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(cfg: dict) -> dict:
    from se.losses import LOSSES
    from se.models import MODELS
    if cfg["model"] not in MODELS:
        raise ValueError(f"unknown model {cfg['model']!r}; registered: {sorted(MODELS)}")
    if cfg["loss"] not in LOSSES:
        raise ValueError(f"unknown loss {cfg['loss']!r}; registered: {sorted(LOSSES)}")
    t = cfg["train"]
    if t["batch_size"] <= 0 or not 0 < t["min_lr"] <= t["lr"]:
        raise ValueError("batch_size > 0 and 0 < min_lr <= lr are required")
    if not 1 <= t["min_epochs"] <= t["max_epochs"] or t["patience"] <= 0:
        raise ValueError("1 <= min_epochs <= max_epochs and patience > 0 are required")
    if not 0 < t["lr_factor"] < 1:
        raise ValueError("0 < lr_factor < 1 is required")
    MODELS[cfg["model"]](**cfg["model_args"])   # fails early on bad model settings
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = _merge(DEFAULTS, json.loads(Path(path).read_text(encoding="utf-8")))
    cfg = apply_overrides(cfg, overrides or [])
    return validate(cfg)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """['train.lr=5e-4', 'model_args.hidden_channels=[128,128]'] -> cfg (values are JSON)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--set needs key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if parents and parents[0] == "train" and last not in TRAIN_DEFAULTS:
            raise KeyError(f"unknown train setting {last!r}")
        node[last] = value
    return cfg
