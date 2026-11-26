"""SEMamba with Mamba in plain PyTorch: the fast scan equals the reference, causality, interface."""
import pytest
import torch

from se.models import build_model

SMALL = {"hid_feature": 16, "num_tfmamba": 1, "expand": 2}


def test_fast_scan_equals_the_reference_scan():
    from se.arch.semamba.mamba_torch import selective_scan, selective_scan_reference
    torch.manual_seed(0)
    b, length, d, n = 3, 37, 8, 16
    u, delta = torch.randn(b, length, d), torch.rand(b, length, d) * 2
    A = -torch.exp(torch.randn(d, n))
    B, C = torch.randn(b, length, n), torch.randn(b, length, n)
    ref = selective_scan_reference(u, delta, A, B, C)
    assert torch.allclose(selective_scan(u, delta, A, B, C), ref, rtol=1e-5, atol=1e-5)


def test_mamba_is_causal_and_matches_reference_path():
    from se.arch.semamba.mamba_torch import Mamba
    torch.manual_seed(1)
    m = Mamba(16, d_state=16, d_conv=4, expand=4).eval()
    x = torch.randn(2, 50, 16)
    with torch.inference_mode():
        y = m(x)
        x2 = x.clone()
        x2[:, 30:] += 1.0                                           # a change at step 30 ...
        y2 = m(x2)
        m.reference_scan = True
        y_ref = m(x)
    assert torch.allclose(y[:, :30], y2[:, :30], atol=1e-6)         # ... leaves earlier outputs alone
    assert not torch.allclose(y[:, 30:], y2[:, 30:])
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-5)


def test_semamba_keeps_length_passes_silence_and_follows_level():
    pytest.importorskip("einops")
    torch.manual_seed(2)
    model = build_model("semamba", {**SMALL, "chunk_seconds": 0.5, "overlap_seconds": 0.1}).eval()
    x = 0.1 * torch.randn(2, 4050)                                  # not a multiple of hop
    x[1] = 0.0
    with torch.inference_mode():
        y = model.enhance(x)
        long = model.enhance(0.1 * torch.randn(1, 19000))           # several chunks
        # scale in = scale out. In float64: with random weights the net amplifies float32
        # rounding of the normalised input to several percent, which is not what is tested here
        model.double()
        xd = x[:1, :3200].double()
        a, b = model.enhance(xd), model.enhance(10 * xd)
    assert y.shape == x.shape and torch.isfinite(y).all() and torch.equal(y[1], x[1])
    assert long.shape == (1, 19000) and torch.isfinite(long).all() and long.abs().max() > 0
    assert torch.allclose(10 * a, b, rtol=1e-6, atol=1e-9)


def test_semamba_state_dict_has_the_source_names():
    pytest.importorskip("einops")
    keys = set(build_model("semamba", SMALL).state_dict())
    block = "net.TSMamba.0.time_mamba.forward_blocks.0."
    for name in ("mixer.A_log", "mixer.D", "mixer.in_proj.weight", "mixer.conv1d.weight",
                 "mixer.conv1d.bias", "mixer.x_proj.weight", "mixer.dt_proj.weight",
                 "mixer.dt_proj.bias", "mixer.out_proj.weight", "norm.weight"):
        assert block + name in keys
    assert not any("in_proj.bias" in k or "out_proj.bias" in k for k in keys)
