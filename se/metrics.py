"""PESQ-WB and SNR as the course notebook computes them.

No level normalisation or time alignment; scores are plain means over files.
Clean speech is read only by evaluation and training code, never by denoise.py.
"""
from __future__ import annotations

import numpy as np

from se import SAMPLE_RATE


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """10 log10(sum s^2 / sum (s - s_hat)^2)."""
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if clean.shape != estimate.shape:
        raise ValueError("SNR needs equal lengths")
    energy = float(np.sum(clean ** 2))
    if energy <= 0:
        return float("nan")
    err = float(np.sum((clean - estimate) ** 2))
    return float("inf") if err == 0 else float(10 * np.log10(energy / err))


def pesq_score(clean: np.ndarray, estimate: np.ndarray, sr: int = SAMPLE_RATE):
    """(PESQ-WB, error message). A failure is NaN with the message, never 0."""
    from pesq import PesqError, pesq
    try:
        value = float(pesq(sr, np.asarray(clean, dtype=np.float64),
                           np.asarray(estimate, dtype=np.float64), mode="wb",
                           on_error=PesqError.RAISE_EXCEPTION))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid PESQ value {value}")
        return value, ""
    except Exception as e:  # noqa: BLE001 - reported per file
        return float("nan"), f"{type(e).__name__}: {e}"


def pesq_pair(args) -> float:
    """(clean, estimate) -> PESQ-WB or NaN; top-level so process pools can pickle it."""
    return pesq_score(*args)[0]


def score_pair(args):
    """(clean, estimate) -> (pesq, snr, error); top-level so process pools can pickle it."""
    clean, estimate = args
    p, err = pesq_score(clean, estimate)
    return p, snr_db(clean, estimate), err
