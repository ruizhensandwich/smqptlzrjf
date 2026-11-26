"""Put several trained models into one weight file whose output is their weighted mean.

    python tools/make_ensemble.py --members A.pt B.pt --weights 0.5 0.5 --out weights/candidates/E.pt
    python tools/make_ensemble.py --members L.pt F.pt M.pt --weights 0.5 0.2 0.3 \\
        --member-set 0:level=proj --combine mag --phase-member 0 --pcs 0.5 --out E.pt

--combine mag averages STFT magnitudes and takes the phase of --phase-member; --pcs applies
Perceptual Contrast Stretching of that strength (se/models.py Ensemble, se/pcs.py).
--member-set INDEX:KEY=VALUE adds or replaces one model argument of a member (0-based index,
VALUE read as JSON if it parses, else as text), e.g. the output-level correction of
tflocoformer; the member's weights are not changed.

Each member is a checkpoint of this repository (se/checkpoint.py). The result is again one
such checkpoint (model "ensemble", se/models.py), so denoise.py, evaluate.py and
tools/promote.py handle it like any other: one file, nothing downloaded.
The weights are fixed numbers for all input files. Choose them on validation in coarse steps
(0.5 : 0.5, 0.6 : 0.4, ...); choosing a model per file would be hand-tuning.
"""
from __future__ import annotations

import argparse
import json
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


def combine(paths: list[Path], weights: list[float], member_args: list[dict] | None = None,
            combine: str = "wave", phase_member: int = 0, pcs: float = 0.0):
    """(ensemble model with the members' weights loaded, model_args, meta)."""
    if len(paths) != len(weights):
        raise ValueError("one weight per member")
    member_args = member_args or [{} for _ in paths]
    if len(member_args) != len(paths):
        raise ValueError("one member_args entry per member")
    ckpts = [read_checkpoint(p) for p in paths]
    members = [{"model": c["model"], "model_args": {**(c["model_args"] or {}), **extra},
                "weight": float(w)} for c, w, extra in zip(ckpts, weights, member_args)]
    model_args = {"members": members}
    if combine != "wave":
        model_args.update({"combine": combine, "phase_member": phase_member, "pcs": pcs})
    model = build_model("ensemble", model_args)
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
    how = "" if combine == "wave" else (f" [STFT magnitude mean, phase of member {phase_member + 1}"
                                        + (f", PCS strength {pcs:g}" if pcs else "") + "]")
    for f, extra in zip(facts, member_args):
        if extra:
            f["model_args_added"] = extra
    meta = {"source": "ensemble: " + " + ".join(f"{f['weight']:g} x {f['file']}" for f in facts) + how,
            "members": facts, "parameters": count_parameters(model)}
    return model, model_args, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--member-set", nargs="*", default=[], help="INDEX:KEY=VALUE, e.g. 0:level=proj")
    ap.add_argument("--combine", choices=["wave", "mag"], default="wave")
    ap.add_argument("--phase-member", type=int, default=0, help="0-based index of the member whose phase is used")
    ap.add_argument("--pcs", type=float, default=0.0)
    args = ap.parse_args(argv)
    extra = [{} for _ in args.members]
    for item in args.member_set:
        index, pair = item.split(":", 1)
        key, value = pair.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        extra[int(index)][key] = value
    model, model_args, meta = combine(args.members, args.weights, extra, args.combine,
                                      args.phase_member, args.pcs)
    save_checkpoint(args.out, model, "ensemble", model_args, meta)
    mb = args.out.stat().st_size / 2 ** 20
    print(f"{args.out.as_posix()}: {meta['source']}, {meta['parameters']:,} params, {mb:.2f} MB")
    if mb > MAX_FILE_MB:
        print(f"[WARN] over {MAX_FILE_MB} MB: GitHub will refuse it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
