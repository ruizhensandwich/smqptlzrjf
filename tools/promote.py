"""Make a trained checkpoint the submission weights (weights/submission.pt).

    python tools/promote.py runs/<run_id>                 # its best.pt (validation-PESQ best)
    python tools/promote.py runs/<run_id> --checkpoint epoch_0040.pt
    python tools/promote.py --state-dict path/to/model.pt --config configs/X.json \
        --summary path/to/training_summary.json --source "course example weights"
    python tools/promote.py --state-dict other.pt --model xlstm_senet --source "..." --out W.pt

The second form converts a bare state_dict (e.g. the course notebook's
best_mask_cnn.pt, or a published checkpoint that keeps it under "generator")
into the self-contained format denoise.py reads. The
training facts the README must show (seed, epochs, selected epoch, validation
score, settings) are stored in the file's "meta".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se.checkpoint import read_checkpoint, save_checkpoint  # noqa: E402
from se.config import load_config  # noqa: E402
from se.models import build_model, count_parameters  # noqa: E402
from tools.common import MAX_FILE_MB, WEIGHTS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path, help="runs/<run_id>")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--state-dict", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--model", help="registered model name, for a --state-dict trained elsewhere")
    ap.add_argument("--summary", type=Path, help="training summary JSON for --state-dict")
    ap.add_argument("--source", default="")
    ap.add_argument("--selection", default="",
                    help="why this epoch, when it is not the validation-PESQ best (shown in README)")
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    args = ap.parse_args(argv)
    args.out = args.out.resolve()

    if args.run:
        ckpt = read_checkpoint(args.run / args.checkpoint)
        model = build_model(ckpt["model"], ckpt["model_args"])
        model.load_state_dict(ckpt["state_dict"])
        meta = dict(ckpt["meta"])
        summary = args.run / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text(encoding="utf-8"))
            meta.update({"epochs_completed": s["epochs_completed"], "stop_reason": s["stop_reason"],
                         "elapsed_seconds": s["elapsed_seconds"]})
        meta["source"] = args.source or f"trained here: {args.run.as_posix()}/{args.checkpoint}"
        if meta.get("best_epoch") is None and meta.get("epoch") is not None:   # epoch_NNNN.pt
            meta.update({"best_epoch": meta["epoch"], "best_val_pesq": meta.get("val_pesq")})
        name, model_args = ckpt["model"], ckpt["model_args"]
    elif args.state_dict and args.config:
        cfg = load_config(args.config)
        name, model_args = cfg["model"], cfg["model_args"]
        model = build_model(name, model_args)
        state = torch.load(args.state_dict, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("generator", state))
        meta = {"seed": cfg["train"]["seed"], "config": cfg, "loss": cfg["loss"],
                "config_path": args.config.as_posix(), "source": args.source}
        if args.summary:
            s = json.loads(args.summary.read_text(encoding="utf-8"))
            meta.update({"run_id": s.get("run_id"), "best_epoch": s.get("best_epoch"),
                         "best_val_pesq": s.get("best_validation_pesq_wb"),
                         "epochs_completed": s.get("epochs_completed"),
                         "stop_reason": s.get("stop_reason"),
                         "elapsed_seconds": s.get("elapsed_seconds"),
                         "torch": s.get("torch_version"), "device": s.get("device")})
    elif args.state_dict and args.model:
        # a checkpoint trained outside this repository: {"model": state_dict, "config": {...},
        # "epoch": n, "val_pesq_wb": x} (or a bare state_dict); model arguments are the defaults
        name, model_args = args.model, {}
        model = build_model(name, model_args)
        data = torch.load(args.state_dict, map_location="cpu", weights_only=True)
        state = data["model"] if isinstance(data.get("model"), dict) else data
        if hasattr(model, "net") and not any(k.startswith("net.") for k in state):
            state = {f"net.{k}": v for k, v in state.items()}      # wrapper around the source net
        model.load_state_dict(state)
        meta = {"source": args.source, "best_epoch": data.get("epoch"),
                "best_val_pesq": data.get("val_pesq_wb"),
                "seed": (data.get("config") or {}).get("seed"),
                "training_config": data.get("config")}
    else:
        sys.exit("give runs/<run_id>, or --state-dict with --config or --model")

    meta["parameters"] = count_parameters(model)
    if args.selection:
        meta["selection"] = args.selection
    save_checkpoint(args.out, model, name, model_args, meta)
    mb = args.out.stat().st_size / 2 ** 20
    print(f"{args.out.relative_to(ROOT).as_posix()}: {name}, {meta['parameters']:,} params, "
          f"{mb:.2f} MB, selected epoch {meta.get('best_epoch')}, "
          f"validation PESQ while training {meta.get('best_val_pesq')}")
    if mb > MAX_FILE_MB:
        print(f"[WARN] over {MAX_FILE_MB} MB: GitHub will refuse it (use Git LFS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
