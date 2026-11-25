"""Put several trained models into one weight file whose output is their weighted mean.

    python tools/make_ensemble.py --members A.pt B.pt --weights 0.5 0.5 --out weights/candidates/E.pt

Each member is a checkpoint of this repository (se/checkpoint.py). The result is again one
such checkpoint (model "ensemble", se/models.py), so denoise.py, evaluate.py and
tools/promote.py handle it like any other: one file, nothing downloaded.
The weights are fixed numbers for all input files. Choose them on validation in coarse steps
(0.5 : 0.5, 0.6 : 0.4, ...); choosing a model per file would be hand-tuning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se.checkpoint import read_checkpoint, save_checkpoint  # noqa: E402
from se.models import build_model, count_parameters  # noqa: E402
from tools.common import MAX_FILE_MB  # noqa: E402

MEMBER_FACTS = ("source", "run_id", "seed", "best_epoch", "best_val_pesq", "epochs_completed",
                "stop_reason", "elapsed_seconds", "config_path", "parameters")


def member_of(model, i: int):
    return model.members[i]


def combine(paths: list[Path], weights: list[float]):
    """(ensemble model with the members' weights loaded, model_args, meta)."""
    if len(paths) != len(weights):
        raise ValueError("one weight per member")
    ckpts = [read_checkpoint(p) for p in paths]
    members = [{"model": c["model"], "model_args": c["model_args"], "weight": float(w)}
               for c, w in zip(ckpts, weights)]
    model = build_model("ensemble", {"members": members})
    for member, c in zip(model.members, ckpts):
        member.load_state_dict(c["state_dict"])
    facts = []
    for p, c, w in zip(paths, ckpts, model.weights):
        try:
            where = p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            where = p.name
        m = c["meta"]
        fact = {"file": where, "model": c["model"], "weight": round(w, 6),
                **{k: m.get(k) for k in MEMBER_FACTS}}
        if fact["best_epoch"] is None and m.get("epoch") is not None:     # a saved epoch_NNNN.pt
            fact.update({"best_epoch": m["epoch"], "best_val_pesq": m.get("val_pesq")})
        fact["parameters"] = fact["parameters"] or count_parameters(member_of(model, len(facts)))
        facts.append(fact)
    meta = {"source": "ensemble: " + " + ".join(f"{f['weight']:g} x {f['file']}" for f in facts),
            "members": facts, "parameters": count_parameters(model)}
    return model, {"members": members}, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    model, model_args, meta = combine(args.members, args.weights)
    save_checkpoint(args.out, model, "ensemble", model_args, meta)
    mb = args.out.stat().st_size / 2 ** 20
    print(f"{args.out.as_posix()}: {meta['source']}, {meta['parameters']:,} params, {mb:.2f} MB")
    if mb > MAX_FILE_MB:
        print(f"[WARN] over {MAX_FILE_MB} MB: GitHub will refuse it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
