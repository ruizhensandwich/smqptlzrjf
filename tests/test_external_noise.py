import csv

import numpy as np
import pytest
import torch

from se.config import load_config
from se.data import EXTERNAL_NOISE, HELD_OUT_ESC_FOLD, RemixedSpeech, load_external_noise
from tools.make_external_noise import training_folds, usable


class Base:
    """Four train pairs of 1 s: speech-like clean, clean plus white noise at 10 dB."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        clean = 0.1 * torch.randn(4, 16000, generator=g)
        noise = 0.1 * 10 ** (-10 / 20) * torch.randn(4, 16000, generator=g)
        self.waveforms = torch.stack((clean + noise, clean), dim=1)
        self.manifest = [{"segment_id": f"s{i}", "target_snr_db": "10.0"} for i in range(4)]

    def __len__(self):
        return 4


def test_the_held_out_fold_can_never_be_training_material(tmp_path):
    assert training_folds([4, 1, 1]) == [1, 4]
    for bad in ([5], [1, 5], [], [0]):
        with pytest.raises(ValueError):
            training_folds(bad)
    rows = [{"filename": "4-1-A-0.wav", "src_file": "1"}, {"filename": "4-2-A-0.wav", "src_file": "2"}]
    assert usable(rows, [{"filename": "5-2-A-3.wav", "src_file": "2"}]) == rows[:1]
    path = tmp_path / EXTERNAL_NOISE["esc50"]
    path.parent.mkdir(parents=True)
    np.save(path, np.full((2, 16000), 100, dtype=np.int16))
    for fold, ok in ((1, True), (HELD_OUT_ESC_FOLD, False)):
        with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "fold"])
            w.writeheader()
            w.writerows({"filename": f"{i}.wav", "fold": fold} for i in range(2))
        if ok:
            pool = load_external_noise("esc50", tmp_path)
            assert pool.shape == (2, 16000) and pool.dtype == torch.float32
        else:
            with pytest.raises(ValueError):
                load_external_noise("esc50", tmp_path)


def test_external_noise_replaces_the_noise_at_the_requested_snr():
    base = Base()
    tone = 0.3 * torch.sin(torch.arange(16000) * 0.2).repeat(3, 1)      # unlike any train noise
    torch.manual_seed(1)
    items = RemixedSpeech(base, 0, 1.0, (5.0, 5.0), external_noise=tone, external_prob=1.0)
    for i in range(4):
        noisy, clean = items[i]
        noise = noisy - clean
        snr = 10 * torch.log10(clean.pow(2).mean() / noise.pow(2).mean())
        assert torch.equal(clean, base.waveforms[i, 1]) and abs(float(snr) - 5.0) < 1e-3
        spectrum = torch.fft.rfft(noise).abs()
        assert spectrum.max() > 50 * spectrum.median()                  # a tone, not white noise


def test_no_external_noise_keeps_the_old_random_draws():
    base = Base()
    torch.manual_seed(3)
    old = [RemixedSpeech(base, 8000, 0.5)[i] for i in range(4)]
    torch.manual_seed(3)
    new = [RemixedSpeech(base, 8000, 0.5, external_noise=torch.ones(2, 16000), external_prob=0.0)[i]
           for i in range(4)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(old, new))


def test_config_checks_the_external_noise_settings():
    with pytest.raises(ValueError):
        load_config(None, ['data.external_noise="nope"'])
    with pytest.raises(ValueError):                                     # needs remixing
        load_config(None, ['data.external_noise="esc50"', "data.external_noise_prob=0.5"])
    cfg = load_config(None, ['data.external_noise="esc50"', "data.external_noise_prob=0.5",
                             "data.remix_prob=0.5"])
    assert cfg["data"]["external_noise_prob"] == 0.5
