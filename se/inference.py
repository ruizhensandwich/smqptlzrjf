"""Inference shared by denoise.py (submission) and evaluate.py (scoring).

evaluate.py scores exactly what denoise.py writes: the same enhance() and the
same write path (to_file_bytes), so 16-bit rounding is included in the score.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from se import SAMPLE_RATE

FULL_SCALE = 1.0 - 2.0 ** -15


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.inference_mode()
def enhance(model, x: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """One channel, any length -> enhanced float64 of the same length.

    Inputs at another rate are resampled to 16 kHz and back (the challenge
    input is 16 kHz, so this is only a safety net).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    y16 = resample_poly(x, SAMPLE_RATE, sr) if sr != SAMPLE_RATE else x
    wave = torch.from_numpy(y16.astype(np.float32)).unsqueeze(0).to(device)
    y = model.enhance(wave)[0].double().cpu().numpy()
    if sr != SAMPLE_RATE:
        y = resample_poly(y, sr, SAMPLE_RATE)
    y = y[:n] if y.size >= n else np.pad(y, (0, n - y.size))
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def enhance_channels(model, x: np.ndarray, sr: int, device) -> np.ndarray:
    """(samples, channels) -> same shape, each channel on its own, clipped for writing."""
    y = np.empty_like(x, dtype=np.float64)
    for ch in range(x.shape[1]):
        y[:, ch] = enhance(model, x[:, ch], sr, device)
    return np.clip(y, -1.0, FULL_SCALE)


def to_file_bytes(y: np.ndarray, sr: int, subtype: str, fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype=subtype, format=fmt)
    return buf.getvalue()


def as_written(y: np.ndarray, sr: int, subtype: str = "PCM_16") -> np.ndarray:
    """What a reader gets back after denoise.py writes y in this format (mono)."""
    back, _ = sf.read(io.BytesIO(to_file_bytes(y, sr, subtype)), dtype="float64")
    return back
