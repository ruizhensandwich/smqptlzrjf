"""Self-contained weight files.

A checkpoint is a plain dict saved with torch.save and read with
weights_only=True (no pickled code):

    {"format": "se-ckpt-1", "model": "mask_cnn", "model_args": {...},
     "state_dict": {...}, "meta": {"seed": 42, "best_epoch": 89, ...}}

so denoise.py needs nothing but the file to rebuild the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from se.models import build_model

FORMAT = "se-ckpt-1"


def save_checkpoint(path: Path, model, model_name: str, model_args: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # plain JSON types only, or weights_only=True loading fails (e.g. torch.__version__ is a
    # TorchVersion object, not a str)
    plain = lambda d: json.loads(json.dumps(d, default=str))  # noqa: E731
    data = {"format": FORMAT, "model": model_name, "model_args": plain(model_args),
            "state_dict": state, "meta": plain(meta)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def read_checkpoint(path: Path) -> dict:
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (use tools/promote.py to convert "
                         "a bare state_dict)")
    return data


def load_model(path: Path, device: str | torch.device = "cpu"):
    """(model in eval mode on device, checkpoint dict)."""
    data = read_checkpoint(path)
    model = build_model(data["model"], data["model_args"])
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval(), data
