"""Quiet-edge guard in se.inference.enhance: no-op on course-like files, trims silent edges."""
import numpy as np
import torch

from se.inference import EDGE_FADE_SECONDS, EDGE_MARGIN_SECONDS, enhance, quiet_edges

SR = 16000
MARGIN = int(EDGE_MARGIN_SECONDS * SR)


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
    assert (a, b) == (SR - MARGIN, SR + core.size + MARGIN)      # 40 ms of each quiet run is kept
    y = enhance(ContextModel(), x, SR, torch.device("cpu"))
    assert y.size == x.size
    assert np.all(y[:a] == 0) and np.all(y[b:] == 0)
    # what the model saw is the kept part only, and the fade stays inside the margin
    kept = ContextModel().enhance(torch.from_numpy(x[a:b].astype(np.float32))[None])[0].double().numpy()
    assert np.array_equal(y[SR:SR + core.size], kept[MARGIN:MARGIN + core.size])
    assert abs(y[a]) < 0.01 * np.abs(kept).max() or kept[0] == 0   # no step at the cut
    fade = int(EDGE_FADE_SECONDS * SR)
    assert np.all(np.abs(y[a:a + fade]) <= np.abs(kept[:fade]) + 1e-12)
    assert np.all(np.abs(y[b - fade:b]) <= np.abs(kept[-fade:]) + 1e-12)


def test_only_the_cut_side_is_faded():
    x = np.concatenate([np.zeros(SR), noisy()])     # silence before, real signal up to the end
    a, b = quiet_edges(x)
    assert (a, b) == (SR - MARGIN, x.size)
    y = enhance(ContextModel(), x, SR, torch.device("cpu"))
    kept = ContextModel().enhance(torch.from_numpy(x[a:b].astype(np.float32))[None])[0].double().numpy()
    assert np.array_equal(y[SR:], kept[MARGIN:])    # the uncut end is not faded


def test_pop_in_the_middle_does_not_cut_quiet_speech():
    # speech-like noise at -45 dBFS with a 60 ms full-scale pop in the middle: the pop is far
    # more than 40 dB above the edges, but too short to move the 90th-percentile level
    x = noisy() * 10 ** (-19 / 20)
    x[SR:SR + SR * 6 // 100] = 0.9
    assert quiet_edges(x) == (0, x.size)
    # weaker beginning and end (30 dB below the middle) with the same pop: still not cut
    x = noisy() * 10 ** (-19 / 20)
    x[:SR // 2] *= 10 ** (-30 / 20)
    x[-SR // 2:] *= 10 ** (-30 / 20)
    x[SR:SR + SR * 6 // 100] = 0.9
    assert quiet_edges(x) == (0, x.size)


def test_quiet_recording_is_not_cut():
    x = noisy() * 10 ** (-40 / 20)                  # -66 dBFS throughout: quiet, but no silent edges
    assert quiet_edges(x) == (0, x.size)
    x = np.concatenate([np.zeros(SR), x])           # true silence before it is still cut
    assert quiet_edges(x) == (SR - MARGIN, x.size)


def test_all_quiet_or_too_little_left():
    assert quiet_edges(np.zeros(SR * 2)) == (0, SR * 2)
    x = np.zeros(SR * 2)
    x[SR:SR + SR // 10] = noisy(0.1)         # only 0.1 s of sound: enhance the whole file
    assert quiet_edges(x) == (0, x.size)
