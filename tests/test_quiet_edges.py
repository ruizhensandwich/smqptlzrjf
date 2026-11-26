"""Quiet-edge guard in se.inference.enhance: no-op on course-like files, trims silent edges."""
import numpy as np
import torch

from se.inference import enhance, quiet_edges

SR = 16000


class ContextModel:
    """Output depends on the whole input (scaled by its RMS), like a real model's normalisation."""

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x / x.pow(2).mean().sqrt().clamp_min(1e-8)


def noisy(seconds=3.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 0.05


def test_no_quiet_edges_is_untouched():
    x = noisy()
    assert quiet_edges(x) == (0, x.size)
    plain = ContextModel().enhance(torch.from_numpy(x.astype(np.float32))[None])[0].double().numpy()
    assert np.array_equal(enhance(ContextModel(), x, SR, torch.device("cpu")), plain)


def test_short_quiet_run_is_kept():
    x = noisy()
    x[:SR // 20] = 0.0                       # 50 ms of silence: shorter than EDGE_MIN_SECONDS
    assert quiet_edges(x) == (0, x.size)


def test_silent_and_hiss_edges_are_cut():
    core = noisy()
    pad = np.zeros(SR)
    hiss = np.random.default_rng(1).standard_normal(SR // 2) * 10 ** (-70 / 20)
    x = np.concatenate([pad, core, hiss])
    a, b = quiet_edges(x)
    assert (a, b) == (SR, SR + core.size)
    y = enhance(ContextModel(), x, SR, torch.device("cpu"))
    assert y.size == x.size
    assert np.all(y[:a] == 0) and np.all(y[b:] == 0)
    alone = enhance(ContextModel(), core, SR, torch.device("cpu"))
    assert np.allclose(y[a:b], alone)


def test_quiet_recording_is_not_cut():
    x = noisy() * 10 ** (-40 / 20)                  # -66 dBFS throughout: quiet, but no silent edges
    assert quiet_edges(x) == (0, x.size)
    x = np.concatenate([np.zeros(SR), x])           # true silence before it is still cut
    assert quiet_edges(x) == (SR, x.size)


def test_all_quiet_or_too_little_left():
    assert quiet_edges(np.zeros(SR * 2)) == (0, SR * 2)
    x = np.zeros(SR * 2)
    x[SR:SR + SR // 10] = noisy(0.1)         # only 0.1 s of sound: enhance the whole file
    assert quiet_edges(x) == (0, x.size)
