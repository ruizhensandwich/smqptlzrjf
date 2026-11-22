from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from se.checkpoint import load_model, save_checkpoint
from se.config import complete, load_config
from se.data import RemixedSpeech, mixing_stats, remix_problems
from se.losses import LOSSES, mpsenet_terms
from se.models import build_model
from se.trainers import MetricGANTrainer, Trainer, build_trainer

ROOT = Path(__file__).resolve().parents[1]
TINY = {"dense_channel": 8, "num_tsblocks": 1, "n_heads": 2, "chunk_seconds": 1.0,
        "overlap_seconds": 0.25}


def tiny():
    torch.manual_seed(0)
    return build_model("mpsenet", TINY)


def test_mpsenet_keeps_any_length_also_in_chunks():
    model = tiny().eval()
    with torch.inference_mode():
        for n in (16000, 15999, 123, 16000 * 3 + 17):       # one chunk, odd, tiny, several chunks
            x = 0.1 * torch.randn(2, n)
            y = model.enhance(x)
            assert y.shape == x.shape and torch.isfinite(y).all()
        assert model.enhance(torch.zeros(1, 8000)).abs().max() < 1e-6   # silence stays silence


def test_mpsenet_output_scales_with_the_input_level():
    model = tiny().eval()
    x = 0.1 * torch.randn(1, 8000)
    with torch.inference_mode():
        assert torch.allclose(model.enhance(0.1 * x), 0.1 * model.enhance(x), atol=1e-6)


def test_full_size_mpsenet_and_checkpoint_round_trip(tmp_path):
    cfg = load_config(ROOT / "configs" / "mpsenet.json")
    model = build_model(cfg["model"], cfg["model_args"])
    assert 2_000_000 < sum(p.numel() for p in model.parameters()) < 2_500_000
    assert "window" not in model.state_dict()
    small = tiny()
    path = tmp_path / "m.pt"
    save_checkpoint(path, small, "mpsenet", TINY, {"seed": 0})
    again, ck = load_model(path)
    x = 0.1 * torch.randn(1, 4000)
    with torch.inference_mode():
        assert ck["model"] == "mpsenet" and torch.allclose(small.eval().enhance(x), again.enhance(x))


def test_mpsenet_loss_terms_are_finite_and_train_every_part():
    model = tiny()
    clean = 0.1 * torch.randn(2, 8000)
    noisy = clean + 0.05 * torch.randn(2, 8000)
    terms, aux = mpsenet_terms(model, noisy, clean)
    assert set(terms) == {"mag", "pha", "com", "stft", "time"}
    assert aux["audio"].shape == aux["clean"].shape == (2, 8000)
    loss = LOSSES["mpsenet"](model, noisy, clean)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def _speechlike(n, seed):
    """Something PESQ accepts: bursts of harmonics with pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
    x = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / 16000) / k for k in range(1, 20))
    return (0.1 * x * (np.sin(2 * np.pi * 1.5 * t) > -0.3)).astype(np.float32)


def test_metric_gan_trainer_trains_resumes_and_picks_by_config():
    pytest.importorskip("pesq")
    cfg = complete({"model": "mpsenet", "model_args": TINY, "loss": "mpsenet",
                    "discriminator": "metric",
                    "train": {"optimizer": "adamw", "scheduler": "exponential", "lr": 1e-3,
                              "lr_gamma": 0.5, "batch_size": 2}})
    clean = torch.from_numpy(np.stack([_speechlike(16000, s) for s in range(4)]))
    noisy = clean + 0.03 * torch.randn_like(clean)
    loader = DataLoader(TensorDataset(noisy, clean), batch_size=2)
    model = tiny()
    trainer = build_trainer(model, cfg, torch.device("cpu"))
    assert isinstance(trainer, MetricGANTrainer)
    before = {k: v.clone() for k, v in trainer.disc.state_dict().items()}
    stats = trainer.train_epoch(loader)
    assert {"train_loss", "disc_loss", "metric", "mag", "pha", "train_pesq"} <= set(stats)
    assert all(np.isfinite(v) for v in stats.values()) and 1.0 <= stats["train_pesq"] <= 4.7
    assert any(not torch.equal(v, before[k]) for k, v in trainer.disc.state_dict().items())
    trainer.end_epoch(1.5)
    assert trainer.lr == pytest.approx(5e-4)
    assert torch.isfinite(trainer.validation_loss(noisy, clean))

    other = build_trainer(tiny(), cfg, torch.device("cpu"))
    other.load_state_dict(saved_and_loaded(trainer.state_dict()))
    assert other.lr == pytest.approx(5e-4)
    assert all(torch.equal(v, other.disc.state_dict()[k])
               for k, v in trainer.disc.state_dict().items())

    plain = build_trainer(build_model("mask_cnn", {}), complete({}), torch.device("cpu"))
    assert type(plain) is Trainer and isinstance(plain.optimizer, torch.optim.Adam)


def saved_and_loaded(state):
    """What torch.save / torch.load(weights_only=True) gives back."""
    import io
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def test_discriminator_needs_the_mpsenet_loss():
    with pytest.raises(ValueError):
        complete({"discriminator": "metric"})
    with pytest.raises(KeyError):
        complete({"data": {"remix": 1}})


class _Base:
    """Stands in for PairedSpeech: 6 pairs at 10 dB, two of them not clean + noise."""

    def __init__(self, n_files=6, n=4000, snr_db=10.0):
        g = torch.Generator().manual_seed(1)
        clean = 0.1 * torch.randn(n_files, n, generator=g)
        noise = torch.randn(n_files, n, generator=g)
        noise *= (clean.pow(2).mean(-1, keepdim=True) / noise.pow(2).mean(-1, keepdim=True)
                  / 10 ** (snr_db / 10)).sqrt()
        self.waveforms = torch.stack((clean + noise, clean), dim=1)
        self.waveforms[0, 0] = clean[0] * 0.5 + noise[0]        # speech level changed
        self.waveforms[1, 0, 400:] = clean[1, 400:]             # noise mostly silent
        self.manifest = [{"segment_id": f"s{i}", "target_snr_db": str(snr_db)}
                         for i in range(n_files)]

    def __len__(self):
        return len(self.manifest)


def test_remix_material_excludes_pairs_that_are_not_clean_plus_noise():
    w = _Base().waveforms.numpy()
    why = [remix_problems(mixing_stats(w[i, 0], w[i, 1]), 10.0) for i in range(6)]
    assert "gain" in why[0] and "quiet" in why[1] and all(not x for x in why[2:])


def test_remixed_items_have_the_requested_snr_and_untouched_speech():
    base = _Base()
    ds = RemixedSpeech(base, segment_samples=0, remix_prob=1.0, snr_db=(3.0, 3.0))
    assert ds.material == [2, 3, 4, 5] and set(ds.excluded) == {"s0", "s1"} and len(ds) == 6
    torch.manual_seed(0)
    for i in range(6):
        noisy, clean = ds[i]
        scale = float(clean.abs().max() / base.waveforms[i, 1].abs().max())   # peak guard only
        assert torch.allclose(clean, base.waveforms[i, 1] * scale)
        snr = 10 * torch.log10(clean.pow(2).sum() / (noisy - clean).pow(2).sum())
        assert abs(float(snr) - 3.0) < 1e-3                  # whole file: the noise is only rolled
        assert float(noisy.abs().max()) <= 0.99 + 1e-6


def test_segments_are_aligned_and_reproducible_without_remixing():
    base = _Base()
    ds = RemixedSpeech(base, segment_samples=1000)
    torch.manual_seed(3)
    noisy, clean = ds[2]
    assert noisy.shape == clean.shape == (1000,)
    w = base.waveforms[2]
    starts = [a for a in range(3001) if torch.equal(clean, w[1, a:a + 1000])]
    assert len(starts) == 1 and torch.equal(noisy, w[0, starts[0]:starts[0] + 1000])
    torch.manual_seed(3)
    assert torch.equal(ds[2][0], noisy)
    torch.manual_seed(5)
    mixed = RemixedSpeech(base, segment_samples=1000, remix_prob=1.0)
    a, b = mixed[3]
    assert a.shape == (1000,) and not torch.equal(a, b)



def test_speed_changed_speech_is_new_but_the_pair_stays_consistent():
    base = _Base()
    ds = RemixedSpeech(base, segment_samples=1000, remix_prob=1.0, snr_db=(5.0, 5.0),
                       speed_factors=[0.9, 1.1], speed_prob=1.0)
    torch.manual_seed(0)
    noisy, clean = ds[3]
    w = base.waveforms[3, 1]
    assert noisy.shape == clean.shape == (1000,) and torch.isfinite(noisy).all()
    assert not any(torch.allclose(clean, w[a:a + 1000], atol=1e-4) for a in range(3001))
    assert float((noisy - clean).abs().max()) > 0          # noise was added to the changed speech
    whole = RemixedSpeech(base, remix_prob=1.0, speed_factors=[1.1], speed_prob=1.0)
    assert whole[2][0].shape == (4000,)                    # faster = shorter, padded back
    with pytest.raises(ValueError):
        complete({"data": {"speed_prob": 0.5}})            # needs factors and remixing
