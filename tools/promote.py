"""Make a trained checkpoint the submission weights (weights/submission.pt).

    python tools/promote.py runs/<run_id>                 # its best.pt (validation-PESQ best)
    python tools/promote.py runs/<run_id> --checkpoint epoch_0040.pt
    python tools/promote.py --state-dict path/to/model.pt --config configs/X.json \
        --summary path/to/training_summary.json --source "course example weights"

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
    ap.add_argument("--summary", type=Path, help="training summary JSON for --state-dict")
    ap.add_argument("--source", default="")
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
    else:
        sys.exit("give runs/<run_id>, or --state-dict with --config")

    meta["parameters"] = count_parameters(model)
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
