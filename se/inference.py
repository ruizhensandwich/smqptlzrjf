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

# Quiet-edge guard (user decision 2026-11-26). The models lose 0.1-1.0 PESQ when digital
# silence or near-silence (e.g. -70 dBFS hiss) sits before or after the speech
# (tools/format_check.py), while the same length of the file's own noise costs nothing.
# A leading or trailing run of 20 ms frames more than EDGE_GAP_DB quieter than the file's
# own loud level (90th percentile of its frame levels) that lasts at least EDGE_MIN_SECONDS
# is cut off before enhancement and written as zeros. The threshold is relative so that a
# quietly recorded file is treated like a loud one (an absolute -55 dBFS threshold cut the
# real edges of files 30 dB quieter than the course files, -0.17 PESQ). In the course noisy
# files (train 1,296, validation 164) the edges are at most 25 dB below that level and no
# file has a qualifying run, so on course-like files the guard does nothing (bit-identical).
# The level is a percentile of frame levels, not the peak: a click or pop shorter than 10 %
# of the file does not move it (tests/test_quiet_edges.py).
# Tried and dropped on 2026-11-26 (tools/format_check.py, 1 s of silence or -70 dBFS hiss
# around the speech; with the plain cut below the loss is 0.00):
#   - keeping 40 ms of the quiet run on each cut side: MP-SENet -0.22, TF-Locoformer -0.39 /
#     -0.31, SEMamba -0.32 / -0.36, MossFormerGAN -0.07 / -0.04 PESQ (164 validation files).
#     Even 40 ms of near-silence at the start of the input hurts these models.
#   - a 10 ms fade of the output at each cut, no margin: -0.02 (MP-SENet, 12 files). The cut
#     is at the first/last loud frame, so the fade attenuates real signal.
EDGE_FRAME = 320                # 20 ms at 16 kHz
EDGE_GAP_DB = 40.0              # below the file's 90th-percentile frame level
EDGE_MIN_SECONDS = 0.1
EDGE_MIN_KEEP_SECONDS = 0.25    # if less than this would be left, enhance the whole file


def quiet_edges(x: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[int, int]:
    """(start, stop) of the part to enhance; (0, len(x)) when there are no quiet edges."""
    n = x.size
    frames = n // EDGE_FRAME
    if frames == 0:
        return 0, n
    f = x[:frames * EDGE_FRAME].reshape(frames, EDGE_FRAME)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-20)
    loud = db >= np.percentile(db, 90) - EDGE_GAP_DB
    min_run = int(np.ceil(EDGE_MIN_SECONDS * sr / EDGE_FRAME))
    head = int(np.argmax(loud))
    tail = int(np.argmax(loud[::-1]))
    start = head * EDGE_FRAME if head >= min_run else 0
    # the remainder after the last whole frame counts as part of the tail run
    stop = (frames - tail) * EDGE_FRAME if tail >= min_run else n
    if stop - start < EDGE_MIN_KEEP_SECONDS * sr:
        return 0, n
    return start, stop


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.inference_mode()
def enhance(model, x: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """One channel, any length -> enhanced float64 of the same length.

    Inputs at another rate are resampled to 16 kHz and back (the challenge
    input is 16 kHz, so this is only a safety net). Quiet edges (see quiet_edges)
    are not enhanced and come out as zeros.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    y16 = resample_poly(x, SAMPLE_RATE, sr) if sr != SAMPLE_RATE else x
    a, b = quiet_edges(y16)
    wave = torch.from_numpy(y16[a:b].astype(np.float32)).unsqueeze(0).to(device)
    y = model.enhance(wave)[0].double().cpu().numpy()
    if (a, b) != (0, y16.size):
        mid = y[:b - a] if y.size >= b - a else np.pad(y, (0, b - a - y.size))
        y = np.concatenate([np.zeros(a), mid, np.zeros(y16.size - b)])
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
