"""Train a model and keep the weights with the best validation PESQ-WB.

    python train.py --config configs/mask_cnn_baseline.json --tag baseline
    python train.py --config configs/X.json --set train.lr=5e-4 --set train.batch_size=32
    python train.py --resume runs/<run_id>          # continue after an interruption (Spot VM)

Each epoch: one pass over the 1,296 train pairs (Adam), then the loss and the
mean PESQ-WB over all 164 validation files. As in the course notebook the
weights with the highest validation PESQ are kept, the learning rate halves
when PESQ stalls (ReduceLROnPlateau) and training stops early after
`patience` epochs without a `min_delta` gain (not before `min_epochs`).

Writes runs/<run_id>/ (git-ignored):
    config.json   resolved config         history.csv   one row per epoch
    best.pt       best weights (se-ckpt)  epoch_XXXX.pt weights at train.save_epochs
    last.pt       state for --resume      summary.json  result and settings
and appends one line to results/training_runs.csv (committed).
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

from se.checkpoint import save_checkpoint  # noqa: E402
from se.config import load_config  # noqa: E402
from se.data import PairedSpeech  # noqa: E402
from se.inference import pick_device  # noqa: E402
from se.losses import LOSSES  # noqa: E402
from se.metrics import pesq_score  # noqa: E402
from se.models import build_model, count_parameters  # noqa: E402

RUNS = HERE / "runs"
TRAINING_LOG = HERE / "results" / "training_runs.csv"
LOG_COLUMNS = ["run_id", "tag", "model", "loss", "params", "seed", "epochs", "best_epoch",
               "best_val_pesq", "stop_reason", "elapsed_min", "device", "torch", "config",
               "note"]

# --------------------------------------------------------------------------
# validation PESQ in worker processes (clean waveforms sent once per worker)
_CLEAN = None


def _init_worker(clean):
    global _CLEAN
    _CLEAN = clean


def _pesq_of(item):
    i, estimate = item
    return pesq_score(_CLEAN[i], estimate)[0]


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = loss_fn(model, noisy, clean)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * noisy.size(0)
    return total / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, loss_fn, device, pool) -> tuple[float, float]:
    model.eval()
    total, estimates = 0.0, []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        total += loss_fn(model, noisy, clean).item() * noisy.size(0)
        estimates.extend(model.enhance(noisy).float().cpu().numpy())
    scores = list(pool.map(_pesq_of, enumerate(estimates), chunksize=4))
    if len(scores) != len(loader.dataset) or not np.all(np.isfinite(scores)):
        raise RuntimeError("PESQ failed on some validation files")
    return total / len(loader.dataset), float(np.mean(scores))


def append_log(row: dict) -> None:
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not TRAINING_LOG.exists()
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
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = args.resume.resolve()
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
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

    train_set, val_set = PairedSpeech("train"), PairedSpeech("validation")
    # Same order as the notebook (seed -> loaders -> model) so a CPU run reproduces it.
    torch.manual_seed(t["seed"])
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"])
    val_loader = DataLoader(val_set, batch_size=t["batch_size"], shuffle=False)
    model = build_model(cfg["model"], cfg["model_args"]).to(device)
    loss_fn = LOSSES[cfg["loss"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], betas=tuple(t["adam_betas"]),
                                 weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t["lr_factor"], patience=t["lr_patience"],
        threshold=t["min_delta"], threshold_mode="abs", min_lr=t["min_lr"])

    best_pesq, ref_pesq, best_epoch, stale, start, history = -1.0, -1.0, 0, 0, 1, []
    elapsed_before = 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        best_pesq, ref_pesq = state["best_pesq"], state["ref_pesq"]
        best_epoch, stale, start = state["best_epoch"], state["stale"], state["epoch"] + 1
        elapsed_before = state["elapsed"]
        with open(run_dir / "history.csv", encoding="utf-8") as f:
            history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        history = [h for h in history if h["epoch"] < start]
        print(f"resuming {run_id} at epoch {start}")

    config_path = "" if args.resume else (args.config.as_posix() if args.config else "")
    if args.resume and (run_dir / "best.pt").exists():
        config_path = torch.load(run_dir / "best.pt", map_location="cpu",
                                 weights_only=True)["meta"].get("config_path", "")
    meta_base = {"run_id": run_id, "seed": t["seed"], "config": cfg, "loss": cfg["loss"],
                 "config_path": config_path, "overrides": [] if args.resume else args.set,
                 "torch": str(torch.__version__), "device": str(device),
                 "train_files": len(train_set), "validation_files": len(val_set)}
    n_params = count_parameters(model)
    print(f"{run_id} | {cfg['model']} {n_params:,} params | device {device} | "
          f"train {len(train_set)} / validation {len(val_set)} | max {t['max_epochs']} epochs",
          flush=True)

    stop_reason, epoch = "max_epochs", start - 1
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(val_set.waveforms[:, 1].numpy(),)) as pool:
        for epoch in range(start, t["max_epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss, val_pesq = validate(model, val_loader, loss_fn, device, pool)
            if val_pesq > best_pesq:
                best_pesq, best_epoch = val_pesq, epoch
                save_checkpoint(run_dir / "best.pt", model, cfg["model"], cfg["model_args"],
                                {**meta_base, "best_epoch": epoch, "best_val_pesq": val_pesq})
            if val_pesq >= ref_pesq + t["min_delta"]:
                ref_pesq, stale = val_pesq, 0
            else:
                stale += 1
            scheduler.step(val_pesq)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                            "validation_pesq_wb": val_pesq, "learning_rate": lr,
                            "best_epoch": best_epoch, "best_pesq_wb": best_pesq,
                            "stale_epochs": stale, "elapsed_seconds": elapsed})
            with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0]))
                w.writeheader()
                w.writerows(history)
            print(f"epoch {epoch:3d} | train {train_loss:.5f} | val loss {val_loss:.5f} | "
                  f"val PESQ {val_pesq:.4f} | best {best_pesq:.4f} (epoch {best_epoch}) | "
                  f"lr {lr:.2g} | {elapsed / 60:.1f} min", flush=True)
            if epoch in t["save_epochs"]:
                save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, cfg["model"],
                                cfg["model_args"], {**meta_base, "epoch": epoch,
                                                    "val_pesq": val_pesq})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(),
                        "epoch": epoch, "best_pesq": best_pesq, "ref_pesq": ref_pesq,
                        "best_epoch": best_epoch, "stale": stale, "elapsed": elapsed},
                       run_dir / "last.pt.tmp")
            (run_dir / "last.pt.tmp").replace(run_dir / "last.pt")
            if epoch >= t["min_epochs"] and stale >= t["patience"]:
                stop_reason = "early_stopping"
                break

    elapsed = elapsed_before + time.perf_counter() - started
    summary = {"run_id": run_id, "status": "complete", "stop_reason": stop_reason,
               "epochs_completed": epoch, "best_epoch": best_epoch,
               "best_validation_pesq_wb": best_pesq, "parameters": n_params,
               "elapsed_seconds": elapsed, "device": str(device), "torch": str(torch.__version__),
               "python": platform.python_version(), "config": cfg}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    append_log({"run_id": run_id, "tag": args.tag or run_id.split("_", 1)[-1], "model": cfg["model"],
                "loss": cfg["loss"], "params": n_params, "seed": t["seed"], "epochs": epoch,
                "best_epoch": best_epoch, "best_val_pesq": f"{best_pesq:.4f}",
                "stop_reason": stop_reason, "elapsed_min": f"{elapsed / 60:.1f}",
                "device": str(device), "torch": str(torch.__version__),
                "config": " ".join([config_path] + [f"--set {s}" for s in meta_base["overrides"]]),
                "note": cfg.get("note", "")})
    print(f"done ({stop_reason}, {epoch} epochs): best validation PESQ {best_pesq:.4f} at epoch "
          f"{best_epoch} -> {run_dir / 'best.pt'}\n"
          f"next: python evaluate.py --weights {(run_dir / 'best.pt').relative_to(HERE).as_posix()} "
          f"--split all --tag {args.tag or 'run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
