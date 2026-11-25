import pytest
import torch

from se.checkpoint import load_model, save_checkpoint
from se.models import build_model
from tools.make_ensemble import combine

CNN = {"model": "mask_cnn", "model_args": {}}


def saved(tmp_path, name, seed):
    torch.manual_seed(seed)
    model = build_model("mask_cnn")
    save_checkpoint(tmp_path / name, model, "mask_cnn", {}, {"source": name, "best_epoch": seed})
    return tmp_path / name


def test_one_file_gives_the_weighted_mean_of_its_members(tmp_path):
    a, b = saved(tmp_path, "a.pt", 1), saved(tmp_path, "b.pt", 2)
    model, model_args, meta = combine([a, b], [3.0, 1.0])
    save_checkpoint(tmp_path / "e.pt", model, "ensemble", model_args, meta)
    both, data = load_model(tmp_path / "e.pt")
    x = 0.1 * torch.randn(2, 16000)
    with torch.inference_mode():
        want = 0.75 * load_model(a)[0].enhance(x) + 0.25 * load_model(b)[0].enhance(x)
        got = both.enhance(x)
    assert got.shape == x.shape and torch.allclose(got, want, atol=1e-6)
    assert [m["weight"] for m in data["meta"]["members"]] == [0.75, 0.25]
    assert [m["best_epoch"] for m in data["meta"]["members"]] == [1, 2]


def test_members_must_be_real_models_with_positive_weights():
    for bad in ([{**CNN, "weight": 1.0}],                                    # one member
                [{**CNN, "weight": 1.0}, {**CNN, "weight": 0.0}],
                [{**CNN, "weight": 1.0}, {"model": "ensemble", "model_args": {}, "weight": 1.0}]):
        with pytest.raises(ValueError):
            build_model("ensemble", {"members": bad})


def test_xlstm_senet_keeps_the_length_and_passes_silence_through():
    pytest.importorskip("einops")
    torch.manual_seed(0)
    model = build_model("xlstm_senet", {"hid_feature": 8, "embed_dim": 8, "num_tfxlstm": 1}).eval()
    x = 0.1 * torch.randn(2, 4050)                                           # not a multiple of hop
    x[1] = 0.0
    with torch.inference_mode():
        y = model.enhance(x)
        long = model.enhance(0.1 * torch.randn(1, 16000 * 5))               # over one 4 s piece
    assert y.shape == x.shape and torch.isfinite(y).all() and torch.equal(y[1], x[1])
    assert long.shape == (1, 80000) and torch.isfinite(long).all()
