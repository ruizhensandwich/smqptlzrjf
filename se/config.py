"""Experiment configuration: one JSON file per experiment under configs/.

A config names the model and its settings, the loss, and the training
settings. Missing keys take the defaults below (the course notebook's
example), unknown keys are an error so that a typo never goes unnoticed.

    {"model": "mask_cnn",
     "model_args": {"n_fft": 512, "hop": 128, "hidden_channels": [64, 64], "kernel_sizes": [5, 3, 1]},
     "loss": "log_mag_l1",
     "discriminator": "",
     "data": {"segment_seconds": 0, "remix_prob": 0, ...},
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
    "optimizer": "adam",      # adam | adamw
    "scheduler": "plateau",   # plateau (lr_factor, lr_patience, min_lr) | exponential (lr_gamma)
    "lr_gamma": 0.99,         # exponential: lr *= lr_gamma after every epoch
    "val_batch_size": 0,      # 0 = batch_size
}

# Training data. The defaults are the course example: every pair whole, as recorded.
DATA_DEFAULTS = {
    "segment_seconds": 0.0,          # > 0: a random segment of this length per item
    "remix_prob": 0.0,               # share of items whose noise comes from another train pair
    "remix_snr_db": [-5.0, 20.0],    # SNR range of remixed items (uniform)
    "gain_db": [0.0, 0.0],           # random level change of the pair (uniform)
    "speed_factors": [],             # e.g. [0.9, 0.95, 1.05, 1.1]: playback speeds of the speech
    "speed_prob": 0.0,               # share of remixed items whose speech gets one of them
    "external_noise": "",            # "" | "esc50": noise pool that is not course data
    "external_noise_prob": 0.0,      # share of remixed items whose noise comes from that pool
    "external_speech": "",           # "" | "zeroth": clean speech pool that is not course data
    "external_speech_prob": 0.0,     # share of remixed items whose speech comes from that pool
}

DEFAULTS = {
    "model": "mask_cnn",
    "model_args": {},
    "loss": "log_mag_l1",
    "discriminator": "",      # "" | "metric" (PESQ-predicting discriminator, se/trainers.py)
    "init_weights": "",       # start from this checkpoint (path from the repository root) instead
                              # of random weights; its model and model_args must equal this config's
    "data": DATA_DEFAULTS,
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
    from se.arch.metric_discriminator import DISCRIMINATORS
    if cfg["discriminator"] and (cfg["discriminator"] not in DISCRIMINATORS
                                 or cfg["loss"] != "mpsenet"):
        raise ValueError(f"discriminator must be '' or one of {sorted(DISCRIMINATORS)} and "
                         "needs the loss 'mpsenet'")
    d = cfg["data"]
    if not 0 <= d["speed_prob"] <= 1 or any(not 0.5 <= f <= 2 for f in d["speed_factors"]) \
            or (d["speed_prob"] > 0 and not (d["speed_factors"] and d["remix_prob"] > 0)):
        raise ValueError("data: speed_prob needs speed_factors within [0.5, 2] and remix_prob > 0")
    from se.data import EXTERNAL_NOISE
    if d["external_noise"] not in ("", *EXTERNAL_NOISE) or not 0 <= d["external_noise_prob"] <= 1 \
            or (d["external_noise_prob"] > 0 and not (d["external_noise"] and d["remix_prob"] > 0)):
        raise ValueError(f"data: external_noise is '' or one of {sorted(EXTERNAL_NOISE)}; "
                         "external_noise_prob needs it and remix_prob > 0")
    from se.data import EXTERNAL_SPEECH
    if d["external_speech"] not in ("", *EXTERNAL_SPEECH) or not 0 <= d["external_speech_prob"] <= 1             or (d["external_speech_prob"] > 0 and not (d["external_speech"] and d["remix_prob"] > 0)):
        raise ValueError(f"data: external_speech is '' or one of {sorted(EXTERNAL_SPEECH)}; "
                         "external_speech_prob needs it and remix_prob > 0")
    if d["segment_seconds"] < 0 or not 0 <= d["remix_prob"] <= 1 or \
            d["remix_snr_db"][0] > d["remix_snr_db"][1] or d["gain_db"][0] > d["gain_db"][1]:
        raise ValueError("data: segment_seconds >= 0, 0 <= remix_prob <= 1, ranges are [low, high]")
    t = cfg["train"]
    if t["optimizer"] not in ("adam", "adamw") or t["scheduler"] not in ("plateau", "exponential") \
            or not 0 < t["lr_gamma"] <= 1 or t["val_batch_size"] < 0:
        raise ValueError("train: optimizer adam|adamw, scheduler plateau|exponential, "
                         "0 < lr_gamma <= 1, val_batch_size >= 0")
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


def complete(cfg: dict) -> dict:
    """A stored config (run folder, checkpoint) with settings added since it was written."""
    return validate(_merge(DEFAULTS, cfg))


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
        known = {"train": TRAIN_DEFAULTS, "data": DATA_DEFAULTS}.get(parents[0] if parents else "")
        if known is not None and last not in known:
            raise KeyError(f"unknown {parents[0]} setting {last!r}")
        node[last] = value
    return cfg
