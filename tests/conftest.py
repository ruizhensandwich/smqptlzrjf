import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se.checkpoint import save_checkpoint  # noqa: E402
from se.models import build_model  # noqa: E402

SMALL = {"n_fft": 256, "hop": 64, "hidden_channels": [8, 8], "kernel_sizes": [3, 3, 1]}


@pytest.fixture
def tiny_weights(tmp_path) -> Path:
    torch.manual_seed(0)
    model = build_model("mask_cnn", SMALL)
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, "mask_cnn", SMALL, {"seed": 0, "best_epoch": 1})
    return path


def write_wav(path: Path, n: int = 16000, sr: int = 16000, channels: int = 1,
              subtype: str = "PCM_16", seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    x = 0.1 * rng.standard_normal((n, channels))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x[:, 0] if channels == 1 else x, sr, subtype=subtype)
    return path
