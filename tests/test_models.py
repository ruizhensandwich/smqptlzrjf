import pytest
import torch

from se.checkpoint import load_model, read_checkpoint
from se.config import DEFAULTS, apply_overrides, load_config
from se.losses import LOSSES
from se.models import MaskCNN, build_model


def test_mask_cnn_keeps_length_and_notebook_parameter_names():
    model = MaskCNN()
    x = torch.randn(2, 48000)
    assert model.enhance(x).shape == x.shape
    assert model(x).shape == (2, 257, 376)          # notebook: 257 bins x 376 frames for 3 s
    keys = set(model.state_dict())
    assert {"window", "mask_net.0.weight", "mask_net.2.weight", "mask_net.4.bias"} <= keys


def test_loss_is_finite_and_differentiable():
    model = MaskCNN(n_fft=256, hop=64, hidden_channels=(8, 8), kernel_sizes=(3, 3, 1))
    noisy, clean = torch.randn(2, 8000), torch.randn(2, 8000)
    loss = LOSSES["log_mag_l1"](model, noisy, clean)
    loss.backward()
    assert torch.isfinite(loss)


def test_config_defaults_are_the_notebook_example():
    cfg = load_config()
    assert cfg["model"] == "mask_cnn" and cfg["train"]["seed"] == 42
    assert cfg["train"]["batch_size"] == 16 and cfg["train"]["lr"] == 1e-3


def test_config_rejects_typos_and_applies_overrides(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"train": {"learning_rate": 0.1}}', encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)
    cfg = apply_overrides(DEFAULTS, ["train.lr=5e-4", "model_args.hidden_channels=[32,32]"])
    assert cfg["train"]["lr"] == 5e-4 and cfg["model_args"]["hidden_channels"] == [32, 32]
    with pytest.raises(KeyError):
        apply_overrides(DEFAULTS, ["train.lrr=1"])


def test_checkpoint_meta_with_torch_objects_still_loads_safely(tmp_path):
    from se.checkpoint import save_checkpoint
    model = build_model("mask_cnn", {"n_fft": 256, "hop": 64, "hidden_channels": [8, 8],
                                     "kernel_sizes": [3, 3, 1]})
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "mask_cnn", {"n_fft": 256, "hop": 64, "hidden_channels": (8, 8),
                                              "kernel_sizes": (3, 3, 1)},
                    {"torch": torch.__version__, "device": torch.device("cpu"), "path": tmp_path})
    meta = read_checkpoint(path)["meta"]        # weights_only=True
    assert meta["torch"] == str(torch.__version__) and meta["device"] == "cpu"


def test_checkpoint_round_trip(tiny_weights):
    model, ck = load_model(tiny_weights)
    assert ck["model"] == "mask_cnn" and ck["meta"]["best_epoch"] == 1
    again = build_model(ck["model"], ck["model_args"])
    again.load_state_dict(read_checkpoint(tiny_weights)["state_dict"])
    x = torch.randn(1, 4000)
    assert torch.allclose(model.enhance(x), again.eval().enhance(x))
