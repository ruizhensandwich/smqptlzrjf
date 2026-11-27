"""Weights between a public checkpoint and its fine-tuned version (WiSE-FT).

    python tools/interpolate.py --base weights/pretrained/mpsenet_dns.pt \\
        --tuned runs/<run_id>/best.pt --alpha 0.5 --out weights/candidates/X_wise50.pt
    python tools/interpolate.py --base B.pt --tuned T.pt --alpha 0.25 0.5 0.75 \\
        --out runs/wise/X_{alpha}.pt

    weights = (1 - alpha) * base + alpha * tuned        (alpha 0 = base, 1 = tuned)

Fine-tuning fits the weights to the course data and can lose some of what the public
weights knew about other noise; a point between the two often keeps most of both
(Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022). Both files must
be the same model with the same settings; --tuned can be a best.pt, an epoch_NNNN.pt or
an averaged epoch_NNNN_ema.pt of train.py. The result is an ordinary checkpoint of this
repository. alpha is one fixed number for all files, to be chosen on the unseen-noise
sets in coarse steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se.checkpoint import read_checkpoint, save_checkpoint  # noqa: E402
from se.models import build_model  # noqa: E402

KEPT = ("run_id", "seed", "config", "loss", "config_path", "overrides", "torch", "device",
        "epochs_completed", "stop_reason", "elapsed_seconds", "selection")


def shown(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def interpolate(base: dict, tuned: dict, alpha: float) -> dict:
    """State dict (1 - alpha) * base + alpha * tuned; non-float entries come from tuned."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    if (base["model"], base["model_args"]) != (tuned["model"], tuned["model_args"]):
        raise ValueError("base and tuned are not the same model with the same settings")
    a, b = base["state_dict"], tuned["state_dict"]
    if list(a) != list(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("base and tuned have different tensors")
    return {k: torch.lerp(a[k].double(), b[k].double(), alpha).to(b[k].dtype)
            if b[k].is_floating_point() else b[k].clone() for k in b}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="the weights training started from")
    ap.add_argument("--tuned", type=Path, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", required=True, help="share of --tuned")
    ap.add_argument("--out", required=True, help="with several alphas: a name containing {alpha}")
    args = ap.parse_args(argv)
    if len(args.alpha) > 1 and "{alpha}" not in args.out:
        ap.error("several alphas need {alpha} in --out")
    base, tuned = read_checkpoint(args.base), read_checkpoint(args.tuned)
    tm = tuned["meta"]
    for alpha in args.alpha:
        model = build_model(tuned["model"], tuned["model_args"])
        model.load_state_dict(interpolate(base, tuned, alpha))
        epoch = tm.get("best_epoch") if tm.get("best_epoch") is not None else tm.get("epoch")
        meta = {**{k: tm[k] for k in KEPT if k in tm},
                "source": f"interpolated weights: {1 - alpha:g} x {shown(args.base)} "
                          f"({base['meta'].get('source', '')}) + {alpha:g} x {shown(args.tuned)}",
                "wise_ft": {"alpha": alpha, "base": shown(args.base), "tuned": shown(args.tuned)},
                "best_epoch": epoch}
        out = Path(args.out.format(alpha=f"{alpha:g}"))
        save_checkpoint(out, model, tuned["model"], tuned["model_args"], meta)
        print(f"{out.as_posix()}: {meta['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
