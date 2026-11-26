"""Magnitude-mean ensemble with PCS, TF-Locoformer's level correction, clipping on the way to 16 bit."""
import numpy as np
import pytest
import torch

from se.checkpoint import load_model, save_checkpoint
from se.inference import FULL_SCALE, as_written, enhance_channels
from se.models import build_model
from se.pcs import PCS400, pcs_magnitude, pcs_stretch
from tools.make_ensemble import combine

CNN = {"model": "mask_cnn", "model_args": {}}


def saved(tmp_path, name, seed):
    torch.manual_seed(seed)
    save_checkpoint(tmp_path / name, build_model("mask_cnn"), "mask_cnn", {}, {"source": name})
    return tmp_path / name


def test_pcs_table_and_strength():
    assert PCS400.shape == (201,) and float(PCS400.min()) == 1.0 and float(PCS400.max()) == 1.4
    mag = torch.rand(2, 201, 30, dtype=torch.float64) * 20
    assert torch.allclose(pcs_magnitude(mag, 0.0), mag)                       # strength 0 = nothing
    full = torch.expm1(PCS400[:, None] * torch.log1p(mag))                    # the source's formula
    assert torch.allclose(pcs_magnitude(mag, 1.0), full)
    half = pcs_magnitude(mag, 0.5)
    assert torch.all(half >= mag - 1e-12) and torch.all(half <= full + 1e-12)


def test_pcs_stretch_keeps_level_handles_silence_and_ignores_input_level():
    torch.manual_seed(0)
    y = 0.1 * torch.randn(3, 16000, dtype=torch.float64)
    y[2] = 0.0                                                                # a silent file: no division by zero
    window = torch.hann_window(400, dtype=torch.float64)
    s = torch.stft(y, 400, 100, window=window, return_complex=True)
    out = pcs_stretch(s.abs(), torch.angle(s), y, 0.5, window)
    assert out.shape == y.shape and torch.isfinite(out).all() and torch.equal(out[2], y[2])
    ratio = (out[:2].pow(2).mean(-1) / y[:2].pow(2).mean(-1)).sqrt()
    assert torch.all((ratio > 0.9) & (ratio < 1.1))                           # level of the input kept
    assert not torch.allclose(out[:2], y[:2], atol=1e-4)                      # but the signal changed
    quiet = pcs_stretch(s.abs() * 0.01, torch.angle(s), y * 0.01, 0.5, window)
    assert torch.allclose(quiet[:2], out[:2] * 0.01, atol=1e-10)              # same stretch 40 dB lower


def test_magnitude_mean_of_equal_members_is_the_member_and_pcs_changes_it(tmp_path):
    a = saved(tmp_path, "a.pt", 1)
    x = 0.1 * torch.randn(2, 16000)
    alone = load_model(a)[0]
    model, model_args, meta = combine([a, a], [0.5, 0.5], None, "mag", 0, 0.0)
    assert model_args["combine"] == "mag" and "magnitude mean" in meta["source"]
    with torch.inference_mode():
        want, got = alone.enhance(x), model.enhance(x)
        short = model.enhance(x[:, :300])                                     # too short for the STFT
    assert got.shape == x.shape and got.dtype == x.dtype
    assert torch.allclose(got[:, 400:-400], want[:, 400:-400], atol=1e-5)     # equal away from the edges
    assert short.shape == (2, 300) and torch.allclose(short, alone.enhance(x[:, :300]), atol=1e-6)
    stretched, args2, meta2 = combine([a, a], [0.5, 0.5], None, "mag", 0, 0.5)
    save_checkpoint(tmp_path / "e.pt", stretched, "ensemble", args2, meta2)
    again, data = load_model(tmp_path / "e.pt")                               # settings travel in the file
    assert data["model_args"]["pcs"] == 0.5 and "PCS strength 0.5" in data["meta"]["source"]
    with torch.inference_mode():
        y = again.enhance(x)
        silent = again.enhance(torch.zeros(1, 16000))
    assert y.shape == x.shape and torch.isfinite(y).all() and not torch.allclose(y, got, atol=1e-4)
    assert torch.isfinite(silent).all() and float(silent.abs().max()) < 1e-6


def test_ensemble_rejects_bad_combination_settings():
    members = [{**CNN, "weight": 1.0}, {**CNN, "weight": 1.0}]
    for bad in ({"combine": "median"}, {"combine": "mag", "phase_member": 2}, {"pcs": 0.5},
                {"combine": "mag", "pcs": -1.0}):
        with pytest.raises(ValueError):
            build_model("ensemble", {"members": members, **bad})


def test_tflocoformer_level_correction(monkeypatch):
    pytest.importorskip("rotary_embedding_torch")
    from se.arch.tflocoformer import LocoDenoiser
    torch.manual_seed(0)
    speech, noise = 0.1 * torch.randn(1, 16000), 0.03 * torch.randn(1, 16000)
    noisy = speech + noise

    def fake(self, mixture, scale=1 / 1.5):          # the published model: clean speech, but too quiet
        return speech[..., :mixture.shape[-1]] * scale

    monkeypatch.setattr(LocoDenoiser, "_enhance_any_length", fake)
    assert torch.allclose(LocoDenoiser().enhance(noisy), speech / 1.5)                     # default: as published
    assert torch.allclose(LocoDenoiser(level="const").enhance(noisy), speech / 1.5 * 1.484)
    out = LocoDenoiser(level="proj").enhance(noisy)
    gain = float((out / (speech / 1.5)).mean())
    assert 1.45 < gain < 1.55                                                              # found without clean
    monkeypatch.setattr(LocoDenoiser, "_enhance_any_length", lambda self, m: speech * 0.2)  # would need 5x
    assert torch.allclose(LocoDenoiser(level="proj").enhance(noisy), speech * 0.2 * 1.65)  # limited
    monkeypatch.setattr(LocoDenoiser, "_enhance_any_length", lambda self, m: torch.zeros_like(m))
    silent = LocoDenoiser(level="proj").enhance(noisy)
    assert torch.isfinite(silent).all() and float(silent.abs().max()) == 0.0               # silent output: no NaN
    with pytest.raises(ValueError):
        LocoDenoiser(level="auto")


class Loud:
    """A model whose output exceeds full scale, as PCS can produce on a file recorded near the limit."""

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        return 1.6 * x


def test_output_over_full_scale_is_clipped_not_wrapped_in_16_bit():
    t = np.arange(16000) / 16000
    x = 0.9 * np.sin(2 * np.pi * 220 * t)                     # 1.6 * 0.9 = 1.44 at the peaks
    y = enhance_channels(Loud(), x[:, None], 16000, torch.device("cpu"))[:, 0]
    assert y.max() == FULL_SCALE and y.min() == -1.0          # limited before writing
    back = as_written(y, 16000)                               # what a reader of the PCM_16 file gets
    assert back.max() == FULL_SCALE and back.min() == -1.0
    loud = np.abs(1.6 * x) > 1.0
    assert np.all(np.sign(back[loud]) == np.sign(x[loud]))    # no sign flip (integer wrap-around)
    assert np.allclose(back[~loud], 1.6 * x[~loud], atol=2 / 32768)
