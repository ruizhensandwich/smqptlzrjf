"""Score weights with PESQ-WB and SNR on train and/or validation, and log the run.

    python evaluate.py                                   # weights/submission.pt, validation (164)
    python evaluate.py --weights runs/<id>/best.pt --split all --tag lr5e-4
    python evaluate.py --split train --n 200             # first 200 train files (quick look)
    python evaluate.py --round 1                         # submission round 1: validation + train
                                                         #   -> results/submissions/round1/ (+ train/)

The enhanced audio is produced exactly as denoise.py writes it (same
inference code, then written and read back in the input's format, PCM_16),
so the score is the score of the submitted files. Scoring follows the course
notebook: PESQ-WB and SNR against the clean file, no alignment, plain mean.
Clean speech is read only here (and in train.py), never by denoise.py.

Writes results/runs/<run_id>/<split>/ (per_file.csv, summary.csv,
run_config.json; git-ignored) and appends one line per method and split to
results/experiments.csv (committed).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from se import SAMPLE_RATE  # noqa: E402
from se.checkpoint import load_model  # noqa: E402
from se.data import data_root, load_waveforms, read_manifest  # noqa: E402
from se.inference import as_written, enhance_channels, pick_device  # noqa: E402
from se.metrics import score_pair  # noqa: E402
from tools.common import fingerprint, round_dir  # noqa: E402

SUBMISSION_WEIGHTS = HERE / "weights" / "submission.pt"
LOG = HERE / "results" / "experiments.csv"
LOG_COLUMNS = ["date", "run_id", "tag", "split", "method", "n", "pesq", "snr_db", "pesq_failed",
               "weights", "weights_sha", "code_fingerprint", "note"]
INFO_COLUMNS = ["noise_category", "noise_code", "noise_difficulty", "target_snr_db"]


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def score_split(split, model, device, method, n, pool) -> tuple[list[dict], int]:
    root = data_root()
    rows = read_manifest(split, root)
    waves = load_waveforms(split, root)
    if n:
        rows, waves = rows[:n], waves[:n]
    subtype = sf.info(str(root / rows[0]["noisy_path"])).subtype
    noisy = waves[:, 0].astype(np.float64)
    clean = waves[:, 1].astype(np.float64)
    outputs = []
    for i in range(len(rows)):
        y = enhance_channels(model, noisy[i][:, None], SAMPLE_RATE, device)[:, 0]
        outputs.append(as_written(y, SAMPLE_RATE, subtype))
    per_file = []
    for name, est in (("noisy_input", noisy), (method, outputs)):
        for r, (p, s, err) in zip(rows, pool.map(score_pair, zip(clean, est), chunksize=8)):
            per_file.append({"segment_id": r["segment_id"], "method": name, "pesq": p,
                             "snr_db": s, "pesq_error": err,
                             **{k: r[k] for k in INFO_COLUMNS}})
    return per_file, len(rows)


def summarize(per_file: list[dict]) -> list[dict]:
    out = []
    for m in dict.fromkeys(r["method"] for r in per_file):
        rs = [r for r in per_file if r["method"] == m]
        p = np.array([r["pesq"] for r in rs], dtype=float)
        s = np.array([r["snr_db"] for r in rs], dtype=float)
        out.append({"method": m, "n": len(rs), "pesq": float(np.nanmean(p)),
                    "snr_db": float(np.nanmean(s)), "pesq_failed": int(np.isnan(p).sum())})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_breakdown(per_file: list[dict], method: str) -> None:
    by = {r["segment_id"]: r for r in per_file if r["method"] == "noisy_input"}
    for key in ("target_snr_db", "noise_difficulty"):
        groups: dict[str, list] = {}
        for r in per_file:
            if r["method"] == method:
                groups.setdefault(r[key], []).append((by[r["segment_id"]]["pesq"], r["pesq"]))
        line = "  ".join(f"{g}: {np.nanmean([a for a, _ in v]):.3f}->{np.nanmean([b for _, b in v]):.3f}"
                         for g, v in sorted(groups.items()))
        print(f"  by {key}: {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=SUBMISSION_WEIGHTS)
    ap.add_argument("--split", choices=["validation", "train", "all"], default="validation")
    ap.add_argument("--n", type=int, default=0, help="only the first n files (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--round", type=int, help="score weights/submission.pt on all files of both "
                    "splits into results/submissions/roundN/")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    if args.round is not None:
        if args.weights.resolve() != SUBMISSION_WEIGHTS.resolve() or args.n:
            sys.exit("--round scores weights/submission.pt on all files; drop --weights/--n")
        args.split, args.tag = "all", args.tag or f"round{args.round}"
    weights = args.weights.resolve()
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model, ckpt = load_model(weights, device)
    method = "submission" if args.round is not None else (args.tag or ckpt["model"])
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    splits = ["train", "validation"] if args.split == "all" else [args.split]
    try:
        weights_rel = weights.relative_to(HERE).as_posix()
    except ValueError:
        weights_rel = str(weights)
    fp = fingerprint()
    meta = {k: v for k, v in ckpt["meta"].items() if k != "config"}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in splits:
            per_file, n = score_split(split, model, device, method, args.n, pool)
            summary = summarize(per_file)
            if args.round is not None:
                out = round_dir(args.round) / ("train" if split == "train" else "")
            else:
                out = HERE / "results" / "runs" / run_id / split
            out.mkdir(parents=True, exist_ok=True)
            write_csv(out / "per_file.csv", per_file)
            write_csv(out / "summary.csv", summary)
            info = {"run_id": run_id, "split": split, "files": n,
                    "subset": "all" if not args.n else f"first {n}",
                    "methods": [s["method"] for s in summary], "weights": weights_rel,
                    "weights_sha": file_sha(weights), "model": ckpt["model"],
                    "model_args": ckpt["model_args"], "checkpoint_meta": meta,
                    "code_fingerprint": fp, "device": str(device), "torch": str(torch.__version__),
                    "python": platform.python_version(), "scored_as": "written PCM_16, read back",
                    "note": args.note}
            (out / "run_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False)
                                                 + "\n", encoding="utf-8")
            new = not LOG.exists()
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                if new:
                    w.writeheader()
                for s in summary:
                    w.writerow({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run_id": run_id,
                                "tag": args.tag, "split": split, "method": s["method"], "n": s["n"],
                                "pesq": f"{s['pesq']:.4f}", "snr_db": f"{s['snr_db']:.3f}",
                                "pesq_failed": s["pesq_failed"], "weights": weights_rel,
                                "weights_sha": info["weights_sha"], "code_fingerprint": fp,
                                "note": args.note})
            print(f"[{split}] {n} files")
            for s in summary:
                print(f"  {s['method']:<14} PESQ {s['pesq']:.4f}  SNR {s['snr_db']:.2f} dB"
                      + (f"  PESQ failed {s['pesq_failed']}" if s["pesq_failed"] else ""))
            print_breakdown(per_file, method)
            try:
                shown = out.relative_to(HERE).as_posix()
            except ValueError:
                shown = str(out)
            print(f"  -> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
