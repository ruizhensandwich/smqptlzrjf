"""Build the external noise pool for training: ESC-50 folds 1-4 as one array.

    python tools/make_external_noise.py --fetch     # download the clips first (about 700 MB)
    python tools/make_external_noise.py             # build from data/external/esc50/ already there

ESC-50 (Piczak 2015, https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0): 2,000 five-second
environmental recordings, 50 classes, 5 folds; clips cut from one source recording share a
fold. **Fold 5 is the noise of the "newnoise" evaluation split (tools/make_newnoise.py) and
is refused here**, so a model trained with this pool has never heard that split's noise.

Each clip: mono, 16 kHz, its loudest 3 s (many clips are padded with silence), stored as
int16. Output next to the clips, read by se/data.py (config data.external_noise = "esc50"):
    train_noise_16k.npy   [clips, 48000] int16
    train_noise_16k.csv   filename, category, fold, start_s  (row i = clip i)
Nothing here is committed: data/ is git-ignored.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import EXTERNAL_NOISE, HELD_OUT_ESC_FOLD, data_root  # noqa: E402
from tools.make_newnoise import (MIN_NOISE_POWER, esc_dir, fetch, loudest_window,  # noqa: E402
                                 read_esc_meta, to_16k_mono)

SECONDS = 3


def training_folds(folds) -> list[int]:
    folds = sorted(set(int(f) for f in folds))
    if HELD_OUT_ESC_FOLD in folds or not folds or not set(folds) <= {1, 2, 3, 4, 5}:
        raise ValueError(f"folds must be among 1-4: fold {HELD_OUT_ESC_FOLD} is the newnoise "
                         "evaluation split and must never be trained on")
    return folds


def build(folder: Path, folds: list[int]) -> tuple[np.ndarray, list[dict]]:
    n = SECONDS * SAMPLE_RATE
    clips, rows = [], []
    for fold in training_folds(folds):
        for r in sorted(read_esc_meta(folder, fold), key=lambda r: r["filename"]):
            x, rate = sf.read(str(folder / "audio" / r["filename"]), dtype="float64")
            w, start = loudest_window(to_16k_mono(x, rate), n)
            if np.mean(w ** 2) < MIN_NOISE_POWER:
                continue                                    # silent clip: not noise
            clips.append(np.round(np.clip(w, -1.0, 32767 / 32768) * 32768).astype(np.int16))
            rows.append({"filename": r["filename"], "category": r["category"], "fold": fold,
                         "start_s": round(start / SAMPLE_RATE, 2)})
    return np.stack(clips), rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download the folds first")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args(argv)
    folds, folder = training_folds(args.folds), esc_dir()
    if args.fetch:
        for fold in folds:
            fetch(folder, fold)
    clips, rows = build(folder, folds)
    out = data_root() / EXTERNAL_NOISE["esc50"]
    np.save(out, clips, allow_pickle=False)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips of folds {folds}, {len({r['category'] for r in rows})} classes "
          f"-> {out.as_posix()} ({out.stat().st_size / 2 ** 20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
