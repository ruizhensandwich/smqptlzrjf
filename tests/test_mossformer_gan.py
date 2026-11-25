import pytest
import torch

from se.models import build_model


def test_mossformer_gan_keeps_length_passes_silence_and_segments_long_inputs():
    pytest.importorskip("einops")
    torch.manual_seed(0)
    model = build_model("mossformer_gan_se", {"window_seconds": 0.5}).eval()   # 8000-sample windows
    x = 0.1 * torch.randn(2, 4050)                                             # not a multiple of hop
    x[1] = 0.0
    with torch.inference_mode():
        y = model.enhance(x)
        long = model.enhance(0.1 * torch.randn(1, 19000))                     # several windows
    assert y.shape == x.shape and torch.isfinite(y).all() and torch.equal(y[1], x[1])
    assert long.shape == (1, 19000) and torch.isfinite(long).all() and long.abs().max() > 0


def test_mossformer_gan_output_follows_the_input_level():
    pytest.importorskip("einops")
    torch.manual_seed(1)
    model = build_model("mossformer_gan_se").eval()
    x = 0.05 * torch.randn(1, 3200)
    with torch.inference_mode():
        a, b = model.enhance(x), model.enhance(10 * x)
    assert torch.allclose(10 * a, b, rtol=1e-3, atol=1e-5)                  # scale in = scale out
