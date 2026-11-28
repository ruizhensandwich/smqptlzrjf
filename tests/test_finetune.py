"""Fine-tuning harness: epoch 0 score, selection by other sets, weight average, WiSE-FT."""
import csv
import json

import numpy as np
import pytest
import soundfile as sf
import torch

import train
from conftest import SMALL
from se.checkpoint import read_checkpoint, save_checkpoint
from se.config import DEFAULTS, apply_overrides, validate
from se.ema import WeightAverage
from se.models import build_model
from tools.interpolate import interpolate, main as interpolate_main

COLUMNS = ["segment_id", "clean_path", "noisy_path", "sample_rate_hz", "num_samples",
           "noise_category", "noise_collection", "noise_code", "noise_difficulty",
           "target_snr_db", "split"]


def make_split(root, split, kinds, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i, kind in enumerate(kinds):
        clean = 0.1 * np.sin(2 * np.pi * (200 + 50 * i) * np.arange(16000) / 16000)
        noisy = clean + 0.05 * rng.standard_normal(16000)
        for sub, x in (("clean", clean), ("noisy", noisy)):
            path = root / split / sub / f"{split}{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), x, 16000, subtype="PCM_16")
        rows.append({"segment_id": f"{split}{i}", "clean_path": f"{split}/clean/{split}{i}.wav",
                     "noisy_path": f"{split}/noisy/{split}{i}.wav", "sample_rate_hz": 16000,
                     "num_samples": 16000, "noise_category": "c", "noise_collection": "x",
                     "noise_code": "n", "noise_difficulty": kind, "target_snr_db": 5.0,
                     "split": split})
    with open(root / split / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


class InlinePool:
    """Stands in for the process pool so the test needs no PESQ and no worker processes."""

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, items, chunksize=1):
        return [fn(i) for i in items]


def fake_pesq(clean, estimate, sr=16000):
    return 1.0 + 3.5 / (1.0 + 1e3 * float(np.mean((np.asarray(clean) - np.asarray(estimate)) ** 2))), ""


def read_history(run):
    with open(run / "history.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_start_weights_are_epoch_0_and_selection_follows_select_by(tmp_path, monkeypatch, tiny_weights):
    data = tmp_path / "data"
    make_split(data, "train", ["easy"] * 4, 0)
    make_split(data, "validation", ["easy"] * 2, 1)
    make_split(data, "newnoise", ["new"] * 2, 2)
    make_split(data, "hard", ["newlow", "lowsnr", "newlow", "lowsnr"], 3)
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setattr(train, "HERE", tmp_path)
    monkeypatch.setattr(train, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train, "TRAINING_LOG", tmp_path / "training_runs.csv")
    monkeypatch.setattr(train, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(train, "pesq_score", fake_pesq)
    # a log written before the two selection columns existed gets them added
    old = train.LOG_COLUMNS[:-2]
    (tmp_path / "training_runs.csv").write_text(
        ",".join(old) + "\n" + ",".join(["old"] * len(old)) + "\n", encoding="utf-8")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({
        "model": "mask_cnn", "model_args": SMALL, "init_weights": str(tiny_weights),
        "train": {"batch_size": 2, "max_epochs": 2, "min_epochs": 1, "save_epochs": [],
                  "select_by": ["newnoise", "hard:newlow"], "ema_decay": 0.5}}), encoding="utf-8")
    assert train.main(["--config", str(config), "--tag", "t", "--no-eval", "--device", "cpu"]) == 0
    run = next((tmp_path / "runs").iterdir())
    rows = read_history(run)
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    first = rows[0]
    assert first["train_loss"] == "nan" and first["ema_select_pesq"] == ""
    for r in rows:              # the selection score is the mean of the two named sets
        mean = (float(r["newnoise_pesq"]) + float(r["hard_newlow_pesq"])) / 2
        assert abs(float(r["select_pesq"]) - mean) < 1e-9
    assert all(r["ema_validation_pesq"] and r["ema_hard_newlow_pesq"] for r in rows[1:])
    scores = [float(r["select_pesq"]) for r in rows]
    best = read_checkpoint(run / "best.pt")
    assert best["meta"]["best_epoch"] == int(np.argmax(scores))
    assert abs(best["meta"]["best_select_pesq"] - max(scores)) < 1e-9
    assert abs(best["meta"]["best_val_pesq"]
               - float(rows[best["meta"]["best_epoch"]]["validation_pesq_wb"])) < 1e-9
    assert "newnoise + hard:newlow" in best["meta"]["selection"]
    # hard:newlow is the two newlow files only
    assert len(train.scored_set("hard:newlow")) == 2
    with pytest.raises(ValueError):
        train.scored_set("hard:nothing")
    # the weight average is saved every epoch and differs from the trained weights
    ema, raw = read_checkpoint(run / "epoch_0002_ema.pt"), read_checkpoint(run / "epoch_0002.pt")
    assert (run / "best_ema.pt").is_file() and ema["meta"]["ema_decay"] == 0.5
    assert any(not torch.equal(ema["state_dict"][k], raw["state_dict"][k]) for k in raw["state_dict"])
    with open(tmp_path / "training_runs.csv", encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    assert log[0]["run_id"] == "old" and log[0]["select_by"] == ""
    assert log[1]["select_by"] == "newnoise+hard:newlow" and float(log[1]["best_select_pesq"]) > 0

    # --resume continues (epoch 0 is not scored again) and keeps the average's state
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg["train"]["max_epochs"] = 3
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert train.main(["--resume", str(run), "--no-eval", "--device", "cpu"]) == 0
    again = read_history(run)
    assert [r["epoch"] for r in again] == ["0", "1", "2", "3"] and again[:3] == rows


def test_default_selection_is_validation_and_bad_settings_fail():
    assert DEFAULTS["train"]["select_by"] == ["validation"] and DEFAULTS["train"]["ema_decay"] == 0
    assert train.column("hard:newlow") == "hard_newlow_pesq"
    for bad in ('train.select_by=["test"]', "train.select_by=[]", 'train.select_by="newnoise"',
                'train.select_by=["newnoise","newnoise"]', 'train.select_by=["hard:"]',
                "train.ema_decay=1"):
        with pytest.raises(ValueError):
            validate(apply_overrides(DEFAULTS, [bad]))


def test_weight_average_follows_the_weights_and_does_not_touch_them():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    start = {k: v.clone() for k, v in model.state_dict().items()}
    avg = WeightAverage(model, 0.9)
    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].num_batches_tracked.add_(7)
    after = {k: v.clone() for k, v in model.state_dict().items()}
    avg.update(model)
    assert torch.allclose(avg.state["0.weight"], start["0.weight"] + 0.1)
    assert int(avg.state["1.num_batches_tracked"]) == 7              # counters are copied
    assert all(torch.equal(v, after[k]) for k, v in model.state_dict().items())
    other = WeightAverage(model, 0.5)
    other.load_state_dict(avg.state_dict())
    assert other.decay == 0.9 and torch.equal(other.state["0.weight"], avg.state["0.weight"])


def test_interpolation_ends_at_both_files_and_is_linear_between(tmp_path, tiny_weights):
    torch.manual_seed(1)
    tuned_path = tmp_path / "tuned.pt"
    save_checkpoint(tuned_path, build_model("mask_cnn", SMALL), "mask_cnn", SMALL,
                    {"epoch": 3, "run_id": "r"})
    base, tuned = read_checkpoint(tiny_weights), read_checkpoint(tuned_path)
    for alpha, want in ((0.0, base), (1.0, tuned)):
        got = interpolate(base, tuned, alpha)
        assert all(torch.equal(got[k], want["state_dict"][k]) for k in got)
    half = interpolate(base, tuned, 0.5)
    k = next(k for k in half if half[k].is_floating_point())
    assert torch.allclose(half[k], (base["state_dict"][k] + tuned["state_dict"][k]) / 2)
    with pytest.raises(ValueError):
        interpolate(base, {**tuned, "model_args": {**SMALL, "hop": 32}}, 0.5)
    with pytest.raises(ValueError):
        interpolate(base, tuned, 1.5)
    out = tmp_path / "w_{alpha}.pt"
    assert interpolate_main(["--base", str(tiny_weights), "--tuned", str(tuned_path),
                             "--alpha", "0.25", "0.75", "--out", str(out)]) == 0
    made = read_checkpoint(tmp_path / "w_0.25.pt")
    assert made["meta"]["wise_ft"]["alpha"] == 0.25 and made["meta"]["best_epoch"] == 3
    assert "0.75 x" in made["meta"]["source"] and (tmp_path / "w_0.75.pt").is_file()


def test_discriminator_can_start_from_a_file(tmp_path, monkeypatch):
    import se.trainers as trainers
    from se.arch.metric_discriminator import MetricDiscriminator
    torch.manual_seed(3)
    start = MetricDiscriminator()
    torch.save({"state_dict": start.state_dict(), "source": "test"}, tmp_path / "d.pt")
    monkeypatch.setattr(trainers, "ROOT", tmp_path)
    args = {"dense_channel": 8, "num_tsblocks": 1, "n_heads": 2}
    cfg = validate(apply_overrides(DEFAULTS, ['model="mpsenet"', f"model_args={json.dumps(args)}",
                                              'loss="mpsenet"', 'discriminator="metric"',
                                              'init_discriminator="d.pt"']))
    torch.manual_seed(4)
    tr = trainers.build_trainer(build_model("mpsenet", args), cfg, torch.device("cpu"))
    assert all(torch.equal(v, start.state_dict()[k]) for k, v in tr.disc.state_dict().items())
    with pytest.raises(ValueError):
        validate(apply_overrides(DEFAULTS, ['init_discriminator="d.pt"']))
