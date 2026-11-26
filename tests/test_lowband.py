import numpy as np

from tools.make_lowband import SYNTH_KINDS, low_share, synth_noise


def test_synthetic_noises_are_low_frequency_and_dc_free():
    rng = np.random.default_rng(0)
    for kind in SYNTH_KINDS:
        x, detail = synth_noise(kind, 48000, rng)
        assert x.shape == (48000,) and np.all(np.isfinite(x)) and detail
        assert low_share(x) > 0.6, kind
        assert abs(np.mean(x)) < 0.05 * np.std(x), kind


def test_low_share_separates_bands():
    t = np.arange(48000) / 16000
    assert low_share(np.sin(2 * np.pi * 100 * t)) > 0.95
    assert low_share(np.sin(2 * np.pi * 2000 * t)) < 0.05
