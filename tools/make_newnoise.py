"""Build the "newnoise" split: validation speech mixed with noise the course data does not have.

    python tools/make_newnoise.py --fetch        # download the ESC-50 clips it needs, then build
    python tools/make_newnoise.py                # build from data/external/esc50/ already there

Why: every noise recording of the course validation split is also in train (only the
speakers are new), so validation PESQ says nothing about unseen noise. This split keeps
the 164 validation clean files and their SNRs and replaces the noise with ESC-50 clips
(Piczak 2015, https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0; 2,000 five-second
environmental recordings in 50 classes, 5 folds). Only fold 5 is used here and it is
never used for training, so folds 1-4 stay free as training material.

Each file: one ESC-50 clip (classes in turn, a clip at most once), resampled to 16 kHz,
its loudest 3 s, scaled so that power(clean) / power(noise) over the whole file is the
file's target_snr_db - the same SNR definition as the course data. If the mixture would
clip, noisy and clean are lowered together (column gain_db).

Output: <data>/newnoise/{clean,noisy}/*.wav (PCM_16) and samples.csv in the course format,
read by evaluate.py --split newnoise. The files contain students' recordings: data/ is
git-ignored and nothing here may be published. Deterministic for a given --seed.
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import data_root, load_waveforms, read_manifest  # noqa: E402

SPLIT = "newnoise"
ESC_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
MIN_NOISE_POWER = 1e-8          # a clip whose loudest 3 s are this quiet is not noise
EXTRA_COLUMNS = ["source_segment_id", "noise_start_s", "gain_db"]


def esc_dir() -> Path:
    return data_root() / "external" / "esc50"


def read_esc_meta(folder: Path, fold: int) -> list[dict]:
    with open(folder / "meta" / "esc50.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r["fold"]) == fold]


def fetch(folder: Path, fold: int) -> None:
    """meta/esc50.csv, LICENSE and the clips of one fold (about 440 kB each)."""
    (folder / "meta").mkdir(parents=True, exist_ok=True)
    (folder / "audio").mkdir(exist_ok=True)
    for rel in ("meta/esc50.csv", "LICENSE"):
        if not (folder / rel).is_file():
            urllib.request.urlretrieve(ESC_RAW + rel, folder / rel)
    rows = read_esc_meta(folder, fold)
    for i, r in enumerate(rows, 1):
        target = folder / "audio" / r["filename"]
        if not target.is_file():
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(ESC_RAW + "audio/" + r["filename"], tmp)
            tmp.replace(target)
        if i % 50 == 0 or i == len(rows):
            print(f"  ESC-50 fold {fold}: {i}/{len(rows)} clips", flush=True)


def to_16k_mono(x: np.ndarray, rate: int) -> np.ndarray:
    x = x.mean(axis=1) if x.ndim == 2 else x
    if rate != SAMPLE_RATE:
        g = np.gcd(rate, SAMPLE_RATE)
        x = resample_poly(x.astype(np.float64), SAMPLE_RATE // g, rate // g)
    return np.asarray(x, dtype=np.float64)


def loudest_window(x: np.ndarray, n: int, hop: int = SAMPLE_RATE // 4) -> tuple[np.ndarray, int]:
    """The n samples of x with the most energy (start on a hop grid); x is tiled if shorter."""
    if x.size < n:
        x = np.tile(x, int(np.ceil(n / max(x.size, 1))))
    power = np.concatenate(([0.0], np.cumsum(x ** 2)))
    starts = np.arange(0, x.size - n + 1, hop)
    best = int(starts[np.argmax(power[starts + n] - power[starts])])
    return x[best:best + n], best


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(noisy, clean, gain_db): noise scaled to snr_db, both lowered together if needed."""
    scale = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    noisy = clean + scale * noise
    peak = float(np.max(np.abs(noisy)))
    gain = 0.99 / peak if peak > 0.99 else 1.0
    return noisy * gain, clean * gain, 20 * np.log10(gain)


def plan(val_rows: list[dict], esc_rows: list[dict], seed: int) -> list[dict]:
    """One ESC-50 clip per validation file: classes in turn, no clip twice."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for r in sorted(esc_rows, key=lambda r: r["filename"]):
        by_class.setdefault(r["category"], []).append(r)
    for clips in by_class.values():
        rng.shuffle(clips)
    classes = sorted(by_class)
    if len(val_rows) > sum(len(c) for c in by_class.values()):
        raise ValueError("more validation files than noise clips")
    order, chosen = rng.permutation(len(val_rows)), [None] * len(val_rows)
    k = 0
    for i in order:                      # which file gets which class does not follow file order
        while not by_class[classes[k % len(classes)]]:
            k += 1
        chosen[i] = by_class[classes[k % len(classes)]].pop()
        k += 1
    return chosen


def build(folder: Path, fold: int, seed: int) -> list[dict]:
    root = data_root()
    val_rows = read_manifest("validation", root)
    clean_all = load_waveforms("validation", root)[:, 1].astype(np.float64)
    n = clean_all.shape[-1]
    chosen = plan(val_rows, read_esc_meta(folder, fold), seed)
    out = root / SPLIT
    for sub in ("clean", "noisy"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.wav"):
            old.unlink()
    rows = []
    for v, e, clean in zip(val_rows, chosen, clean_all):
        x, rate = sf.read(str(folder / "audio" / e["filename"]), dtype="float64")
        noise, start = loudest_window(to_16k_mono(x, rate), n)
        if np.mean(noise ** 2) < MIN_NOISE_POWER or np.mean(clean ** 2) <= 0:
            raise ValueError(f"silent material: {e['filename']} / {v['segment_id']}")
        snr = float(v["target_snr_db"])
        noisy, clean, gain_db = mix(clean, noise, snr)
        sid = f"{v['segment_id']}_nn"
        paths = {"clean": f"{SPLIT}/clean/{sid}.wav", "noisy": f"{SPLIT}/noisy/{sid}_noisy.wav"}
        sf.write(str(root / paths["clean"]), clean, SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(root / paths["noisy"]), noisy, SAMPLE_RATE, subtype="PCM_16")
        rows.append({"segment_id": sid, "clean_path": paths["clean"], "noisy_path": paths["noisy"],
                     "sample_rate_hz": SAMPLE_RATE, "num_samples": n,
                     "noise_category": e["category"], "noise_collection": "ESC-50",
                     "noise_code": e["filename"], "noise_difficulty": "external",
                     "target_snr_db": snr, "split": SPLIT,
                     "source_segment_id": v["segment_id"],
                     "noise_start_s": round(start / SAMPLE_RATE, 2), "gain_db": round(gain_db, 2)})
    with open(out / "samples.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download the ESC-50 fold first")
    ap.add_argument("--fold", type=int, default=5, help="ESC-50 fold kept for this split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    folder = esc_dir()
    if args.fetch:
        fetch(folder, args.fold)
    if not (folder / "meta" / "esc50.csv").is_file():
        sys.exit(f"{folder} has no meta/esc50.csv: run with --fetch")
    rows = build(folder, args.fold, args.seed)
    classes = {r["noise_category"] for r in rows}
    lowered = sum(r["gain_db"] < 0 for r in rows)
    print(f"{SPLIT}: {len(rows)} pairs, {len(classes)} noise classes, "
          f"{lowered} lowered to avoid clipping -> {(data_root() / SPLIT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
