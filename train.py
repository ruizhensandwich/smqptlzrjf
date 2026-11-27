"""Train a model and keep the weights with the best selection score (validation PESQ-WB).

    python train.py --config configs/mask_cnn_baseline.json --tag baseline
    python train.py --config configs/X.json --set train.lr=5e-4 --set train.batch_size=32
    python train.py --resume runs/<run_id>          # continue after an interruption (Spot VM)

Each epoch: one pass over the 1,296 train pairs, then the loss and the mean
PESQ-WB over all 164 validation files. As in the course notebook the weights
with the highest validation PESQ are kept and training stops early after
`patience` epochs without a `min_delta` gain (not before `min_epochs`). The
config chooses the rest: optimizer and learning-rate schedule (train.*), a
metric discriminator (discriminator, se/trainers.py), random segments and
dynamic remixing of the train pairs (data.*, se/data.py).

train.select_by names the sets whose mean PESQ-WB is that selection score (default
["validation"]; ["newnoise", "hard:newlow"] selects by unseen noise instead, validation is
then only recorded). A run that starts from init_weights first scores those weights as
"epoch 0": they are the first best.pt, so training has to beat them to be chosen.
train.ema_decay > 0 also keeps a running average of the weights (se/ema.py), scored each
epoch and written as epoch_XXXX_ema.pt / best_ema.pt.

Writes runs/<run_id>/ (git-ignored):
    config.json   resolved config         history.csv   one row per epoch
    best.pt       best weights (se-ckpt)  epoch_XXXX.pt weights of every epoch
    last.pt       state for --resume      summary.json  result and settings
and appends one line to results/training_runs.csv (committed). Then the best
weights are scored as submitted (evaluate.py --split all, 16-bit output) and the
model board with every model so far is printed (skip with --no-eval).
Promote a run to the submission weights with  python tools/promote.py runs/<run_id>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from se.checkpoint import read_checkpoint, save_checkpoint  # noqa: E402
from se.config import complete, load_config  # noqa: E402
from se.data import PairedSpeech, RemixedSpeech, data_root, load_external_noise, load_external_speech  # noqa: E402
from se import SAMPLE_RATE  # noqa: E402
from se.inference import pick_device  # noqa: E402
from se.metrics import pesq_score  # noqa: E402
from se.models import build_model, count_parameters  # noqa: E402
from se.trainers import build_trainer  # noqa: E402

RUNS = HERE / "runs"
TRAINING_LOG = HERE / "results" / "training_runs.csv"
LOG_COLUMNS = ["run_id", "tag", "model", "loss", "params", "seed", "epochs", "best_epoch",
               "best_val_pesq", "stop_reason", "elapsed_min", "device", "torch", "config",
               "note", "select_by", "best_select_pesq"]

# --------------------------------------------------------------------------
# validation PESQ in worker processes (clean waveforms sent once per worker)
_CLEAN = None


def _init_worker(clean):
    global _CLEAN
    _CLEAN = clean


def _pesq_of(item):
    i, estimate = item
    return pesq_score(_CLEAN[i], estimate)[0]


@torch.inference_mode()
def validate(model, loader, trainer, device, pool) -> tuple[float, float]:
    model.eval()
    total, estimates = 0.0, []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        total += trainer.validation_loss(noisy, clean).item() * noisy.size(0)
        estimates.extend(model.enhance(noisy).float().cpu().numpy())
    scores = list(pool.map(_pesq_of, enumerate(estimates), chunksize=4))
    if len(scores) != len(loader.dataset) or not np.all(np.isfinite(scores)):
        raise RuntimeError("PESQ failed on some validation files")
    return total / len(loader.dataset), float(np.mean(scores))


def _pesq_pair(item):
    clean, estimate = item
    return pesq_score(clean, estimate)[0]


@torch.inference_mode()
def mean_pesq(model, dataset, batch_size, device, pool) -> float:
    """Mean PESQ-WB of a whole split; a look only, nothing is selected by it."""
    model.eval()
    estimates = []
    for noisy, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        estimates.extend(model.enhance(noisy.to(device)).float().cpu().numpy())
    clean = dataset.waveforms[:, 1].numpy()
    return float(np.nanmean(list(pool.map(_pesq_pair, zip(clean, estimates), chunksize=4))))


def column(name: str) -> str:
    """history.csv column of a scored set: "hard:newlow" -> "hard_newlow_pesq"."""
    return name.replace(":", "_") + "_pesq"


def scored_set(name: str) -> PairedSpeech:
    split, _, subset = name.partition(":")
    data = PairedSpeech(split)
    return data.subset(subset) if subset else data


def write_history(path: Path, history: list[dict]) -> None:
    fields = list(dict.fromkeys(k for h in history for k in h))    # epoch 0 has fewer columns
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(history)


def append_log(row: dict) -> None:
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not TRAINING_LOG.exists()
    if not new:                                 # columns added since the file was started
        with open(TRAINING_LOG, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old, header = list(reader), reader.fieldnames
        if header != LOG_COLUMNS:
            with open(TRAINING_LOG, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, restval="")
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in LOG_COLUMNS} for r in old)
    with open(TRAINING_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="configs/*.json (default: course example)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override, e.g. train.lr=5e-4 or model_args.hidden_channels=[128,128]")
    ap.add_argument("--tag", default="", help="short name for the run")
    ap.add_argument("--note", default="", help="what changed (goes to results/training_runs.csv)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes for validation PESQ")
    ap.add_argument("--resume", type=Path, help="run folder to continue")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the final evaluate.py --split all (and the model board)")
    args = ap.parse_args(argv)

    if args.resume:
        if args.config or args.set or args.note:
            ap.error("--resume continues with the run folder's config.json: --config, --set and "
                     "--note are not read (edit that config.json to change a setting)")
        run_dir = args.resume.resolve()
        cfg = complete(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        run_id = run_dir.name
    else:
        cfg = load_config(args.config, args.set)
        if args.note:
            cfg["note"] = args.note
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
        run_dir = RUNS / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
        state = None
    t = cfg["train"]
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    else:   # faster convolutions and matrix products on the GPU (TF32, tuned cuDNN kernels)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    train_set, val_set = PairedSpeech("train"), PairedSpeech("validation")
    d = cfg["data"]
    train_items = train_set
    if d["segment_seconds"] > 0 or d["remix_prob"] > 0 or tuple(d["gain_db"]) != (0, 0):
        pool_noise = load_external_noise(d["external_noise"]) if d["external_noise_prob"] > 0 \
            else None
        pool_speech = load_external_speech(d["external_speech"])             if d["external_speech_prob"] > 0 else None
        train_items = RemixedSpeech(train_set, int(round(d["segment_seconds"] * SAMPLE_RATE)),
                                    d["remix_prob"], d["remix_snr_db"], d["gain_db"],
                                    d["speed_factors"], d["speed_prob"],
                                    pool_noise, d["external_noise_prob"],
                                    pool_speech, d["external_speech_prob"])
        if pool_speech is not None:
            print(f"external speech: {len(pool_speech)} clips of {d['external_speech']} for "
                  f"{d['external_speech_prob']:.0%} of the remixed items", flush=True)
        if pool_noise is not None:
            print(f"external noise: {len(pool_noise)} clips of {d['external_noise']} for "
                  f"{d['external_noise_prob']:.0%} of the remixed items", flush=True)
        if d["remix_prob"] > 0:
            print(f"remixing: {len(train_items.material)} of {len(train_set)} train pairs usable "
                  f"as noise material ({len(train_items.excluded)} excluded)", flush=True)
    # Same order as the notebook (seed -> loaders -> model) so a CPU run reproduces it.
    torch.manual_seed(t["seed"])
    train_loader = DataLoader(train_items, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"])
    val_loader = DataLoader(val_set, batch_size=t["val_batch_size"] or t["batch_size"],
                            shuffle=False)
    # Sets scored every epoch besides validation: the unseen-noise split (tools/make_newnoise.py)
    # if it has been built, and whatever train.select_by selects the weights by.
    select_by = t["select_by"]
    watch = {"newnoise": PairedSpeech("newnoise")} \
        if (data_root() / "newnoise" / "samples.csv").is_file() else {}
    for name in select_by:
        if name != "validation" and name not in watch:
            watch[name] = scored_set(name)
    selection = "" if select_by == ["validation"] else (
        "학습 중 " + " + ".join(select_by) + " 평균 PESQ-WB가 가장 높은 epoch"
        + (" (0 epoch = 시작 가중치 그대로 포함)" if cfg["init_weights"] else ""))
    model = build_model(cfg["model"], cfg["model_args"]).to(device)
    if cfg["init_weights"] and state is None:
        init = read_checkpoint(HERE / cfg["init_weights"])
        if (init["model"], init["model_args"]) != (cfg["model"], cfg["model_args"]):
            sys.exit(f"init_weights {cfg['init_weights']} is {init['model']} {init['model_args']}, "
                     "not this config's model")
        model.load_state_dict(init["state_dict"])
        print(f"starting from {cfg['init_weights']} ({init['meta'].get('source', '')})", flush=True)

    # best_pesq / ref_pesq are the selection score; best_val is validation PESQ at that epoch
    best_pesq, ref_pesq, best_epoch, stale, start, history = -1.0, -1.0, 0, 0, 1, []
    best_val, best_ema, best_ema_epoch = -1.0, -1.0, 0
    elapsed_before = 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        best_pesq, ref_pesq = state["best_pesq"], state["ref_pesq"]
        best_epoch, stale, start = state["best_epoch"], state["stale"], state["epoch"] + 1
        elapsed_before = state["elapsed"]
        best_val = state.get("best_val", best_pesq)
        best_ema, best_ema_epoch = state.get("best_ema", -1.0), state.get("best_ema_epoch", 0)
        with open(run_dir / "history.csv", encoding="utf-8") as f:
            history = [{k: int(float(v)) if k in ("epoch", "best_epoch", "stale_epochs")
                        else float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]
        history = [h for h in history if h["epoch"] < start]
        if history and "newnoise_pesq" not in history[-1] and "newnoise" not in select_by:
            watch.pop("newnoise", None)     # a run started before the split existed keeps its columns
        print(f"resuming {run_id} at epoch {start}")

    config_path = "" if args.resume else (args.config.as_posix() if args.config else "")
    if args.resume and (run_dir / "best.pt").exists():
        config_path = torch.load(run_dir / "best.pt", map_location="cpu",
                                 weights_only=True)["meta"].get("config_path", "")
    meta_base = {"run_id": run_id, "seed": t["seed"], "config": cfg, "loss": cfg["loss"],
                 "config_path": config_path, "overrides": [] if args.resume else args.set,
                 "torch": str(torch.__version__), "device": str(device),
                 "train_files": len(train_set), "validation_files": len(val_set),
                 **({"selection": selection} if selection else {})}
    n_params = count_parameters(model)
    print(f"{run_id} | {cfg['model']} {n_params:,} params | device {device} | "
          f"train {len(train_set)} / validation {len(val_set)} | max {t['max_epochs']} epochs",
          flush=True)

    stop_reason, epoch = "max_epochs", start - 1
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(val_set.waveforms[:, 1].numpy(),)) as pool:
        trainer = build_trainer(model, cfg, device, pool)
        if state is not None:
            # last.pt of runs made before se/trainers.py kept these two at the top level
            trainer.load_state_dict(state.get("trainer") or
                                    {k: state[k] for k in ("optimizer", "scheduler")})
            torch.set_rng_state(state["rng"])
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], device)
        batch = t["val_batch_size"] or t["batch_size"]

        def score(val_pesq: float, prefix: str = "") -> tuple[dict, float]:
            """({column: PESQ of each watched set}, selection score) of the weights in `model`."""
            seen = {name: mean_pesq(model, data, batch, device, pool) for name, data in watch.items()}
            select = float(np.mean([val_pesq if n == "validation" else seen[n] for n in select_by]))
            out = {prefix + column(n): v for n, v in seen.items()}
            if selection:
                out[prefix + "select_pesq"] = select
            return out, select

        if state is None and cfg["init_weights"]:
            # epoch 0: the start weights as they are. They are the first best.pt and the
            # reference early stopping counts from.
            val_loss, val_pesq = validate(model, val_loader, trainer, device, pool)
            seen, select = score(val_pesq)
            best_pesq = ref_pesq = select
            best_val = val_pesq
            save_checkpoint(run_dir / "best.pt", model, cfg["model"], cfg["model_args"],
                            {**meta_base, "best_epoch": 0, "best_val_pesq": val_pesq,
                             **({"best_select_pesq": select} if selection else {})})
            history.append({"epoch": 0, "train_loss": float("nan"), "validation_loss": val_loss,
                            "validation_pesq_wb": val_pesq, "learning_rate": trainer.lr,
                            "best_epoch": 0, "best_pesq_wb": best_pesq, "stale_epochs": 0,
                            "elapsed_seconds": time.perf_counter() - started, **seen})
            write_history(run_dir / "history.csv", history)
            print(f"epoch   0 | start weights as they are | val loss {val_loss:.5f} | "
                  f"val PESQ {val_pesq:.4f}"
                  + "".join(f" | {k} {v:.3f}" for k, v in seen.items()), flush=True)
        for epoch in range(start, t["max_epochs"] + 1):
            lr = trainer.lr
            train_stats = trainer.train_epoch(train_loader)
            train_loss = train_stats.pop("train_loss")
            val_loss, val_pesq = validate(model, val_loader, trainer, device, pool)
            seen, select = score(val_pesq)
            if select > best_pesq:
                best_pesq, best_epoch, best_val = select, epoch, val_pesq
                save_checkpoint(run_dir / "best.pt", model, cfg["model"], cfg["model_args"],
                                {**meta_base, "best_epoch": epoch, "best_val_pesq": val_pesq,
                                 **({"best_select_pesq": select} if selection else {})})
            if select >= ref_pesq + t["min_delta"]:
                ref_pesq, stale = select, 0
            else:
                stale += 1
            trainer.end_epoch(select)
            train_stats.update(seen)
            if trainer.ema is not None:
                # score the averaged weights in the same model, then put the trained ones back
                trained = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(trainer.ema.state)
                ema_val = mean_pesq(model, val_set, batch, device, pool)
                ema_seen, ema_select = score(ema_val, "ema_")
                train_stats.update({"ema_validation_pesq": ema_val, **ema_seen})
                ema_meta = {**meta_base, "epoch": epoch, "val_pesq": ema_val,
                            "select_pesq": ema_select, "ema_decay": t["ema_decay"]}
                if ema_select > best_ema:
                    best_ema, best_ema_epoch = ema_select, epoch
                    save_checkpoint(run_dir / "best_ema.pt", model, cfg["model"],
                                    cfg["model_args"], ema_meta)
                if t["save_every_epoch"] or epoch in t["save_epochs"]:
                    save_checkpoint(run_dir / f"epoch_{epoch:04d}_ema.pt", model, cfg["model"],
                                    cfg["model_args"], ema_meta)
                model.load_state_dict(trained)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                            "validation_pesq_wb": val_pesq, "learning_rate": lr,
                            "best_epoch": best_epoch, "best_pesq_wb": best_pesq,
                            "stale_epochs": stale, "elapsed_seconds": elapsed,
                            **train_stats})
            write_history(run_dir / "history.csv", history)
            print(f"epoch {epoch:3d} | train {train_loss:.5f} | val loss {val_loss:.5f} | "
                  f"val PESQ {val_pesq:.4f} | best {best_pesq:.4f} (epoch {best_epoch}) | "
                  f"lr {lr:.2g} | {elapsed / 60:.1f} min"
                  + "".join(f" | {k} {v:.3f}" for k, v in train_stats.items()), flush=True)
            if t["save_every_epoch"] or epoch in t["save_epochs"]:
                save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, cfg["model"],
                                cfg["model_args"], {**meta_base, "epoch": epoch,
                                                    "val_pesq": val_pesq, "select_pesq": select})
            torch.save({"model": model.state_dict(), "trainer": trainer.state_dict(),
                        "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device)
                        if device.type == "cuda" else None,
                        "epoch": epoch, "best_pesq": best_pesq, "ref_pesq": ref_pesq,
                        "best_epoch": best_epoch, "stale": stale, "elapsed": elapsed,
                        "best_val": best_val, "best_ema": best_ema,
                        "best_ema_epoch": best_ema_epoch},
                       run_dir / "last.pt.tmp")
            (run_dir / "last.pt.tmp").replace(run_dir / "last.pt")
            if epoch >= t["min_epochs"] and stale >= t["patience"]:
                stop_reason = "early_stopping"
                break

    elapsed = elapsed_before + time.perf_counter() - started
    summary = {"run_id": run_id, "status": "complete", "stop_reason": stop_reason,
               "epochs_completed": epoch, "best_epoch": best_epoch,
               "best_validation_pesq_wb": best_val, "select_by": select_by,
               "best_select_pesq": best_pesq, "parameters": n_params,
               **({"best_ema_epoch": best_ema_epoch, "best_ema_select_pesq": best_ema}
                  if t["ema_decay"] > 0 else {}),
               "elapsed_seconds": elapsed, "device": str(device), "torch": str(torch.__version__),
               "python": platform.python_version(), "config": cfg}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    append_log({"run_id": run_id, "tag": args.tag or run_id.split("_", 1)[-1], "model": cfg["model"],
                "loss": cfg["loss"], "params": n_params, "seed": t["seed"], "epochs": epoch,
                "best_epoch": best_epoch, "best_val_pesq": f"{best_val:.4f}",
                "select_by": "+".join(select_by), "best_select_pesq": f"{best_pesq:.4f}",
                "stop_reason": stop_reason, "elapsed_min": f"{elapsed / 60:.1f}",
                "device": str(device), "torch": str(torch.__version__),
                "config": " ".join([config_path] + [f"--set {s}" for s in meta_base["overrides"]]),
                "note": cfg.get("note", "")})
    tag = args.tag or run_id.split("_", 1)[-1]
    best_rel = (run_dir / "best.pt").relative_to(HERE).as_posix()
    print(f"done ({stop_reason}, {epoch} epochs): best {'+'.join(select_by)} PESQ {best_pesq:.4f} "
          f"at epoch {best_epoch} (validation {best_val:.4f}) -> {best_rel}", flush=True)
    if args.no_eval:
        print(f"next: python evaluate.py --weights {best_rel} --split all --tag {tag}")
        return 0
    # score the chosen weights as submitted (16-bit, train + validation) and show the model board
    import evaluate
    return evaluate.main(["--weights", best_rel, "--split", "all", "--device", args.device,
                          "--tag", tag, "--note", cfg.get("note", ""),
                          "--workers", str(args.workers)])


if __name__ == "__main__":
    raise SystemExit(main())
