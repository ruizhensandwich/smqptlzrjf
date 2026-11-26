"""Build the "lowband" split: validation speech under low-frequency noise the models have not heard.

    python tools/make_lowband.py            # needs ESC-50 fold 5 in data/external/esc50/

Why: the hidden test set has noise that is not in the course data (notice 2026-11-26) and is
harder than the first challenge's. On validation the weakest files are those whose noise has
30-70 % of its energy below 300 Hz (about 0.4 PESQ below the others), but those noises are
also in train. This split asks how the models do on low-frequency noise they have never
heard. Every subset keeps the 164 validation clean files:

  esc_low      an ESC-50 fold 5 clip whose loudest 3 s has at least half of its energy below
               300 Hz (95 clips: helicopter, thunderstorm, washing machine, engine, wind, ...),
               at the file's own validation SNR (0/5/10/15 dB)
  synth_low    synthetic low-frequency noise at the file's own SNR, one of
                 brown     1/f^2 noise
                 lowpass   white noise low-passed at 100-300 Hz
                 hum       an engine-like hum: harmonics of 25-120 Hz up to 400 Hz with a slowly
                           drifting pitch and level, plus some low-passed noise
  esc_low_snr  the esc_low clip at -5 or 0 dB: unseen low-frequency noise AND low SNR

Every noise is high-passed at 20 Hz (no DC or sub-audio drift). SNR = power(speech) /
power(noise) over the whole file, as in the course data; noisy and clean are lowered together
if the mixture would clip (column gain_db). ESC-50 fold 5 is never used for training.

Output: <data>/lowband/{clean,noisy}/*.wav (PCM_16) and samples.csv in the course format with
the subset in noise_difficulty, read by evaluate.py --split lowband (one line per subset).
The files contain students' recordings: data/ is git-ignored and nothing here may be published.
Deterministic for a given --seed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import data_root, load_waveforms, read_manifest  # noqa: E402
from tools.make_newnoise import esc_dir, loudest_window, read_esc_meta, to_16k_mono  # noqa: E402

SPLIT = "lowband"
SUBSETS = ("esc_low", "synth_low", "esc_low_snr")
LOW_HZ = 300.0
MIN_LOW_SHARE = 0.5
LOW_SNRS = (-5.0, 0.0)
SYNTH_KINDS = ("brown", "lowpass", "hum")


def low_share(x: np.ndarray, rate: int = SAMPLE_RATE, cutoff: float = LOW_HZ) -> float:
    """Share of the signal's power below cutoff Hz."""
    f, p = welch(x, rate, nperseg=1024)
    return float(p[f < cutoff].sum() / max(p.sum(), 1e-20))


def highpass(x: np.ndarray, hz: float = 20.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    return sosfilt(butter(4, hz, "highpass", fs=rate, output="sos"), x)


def lowpass(x: np.ndarray, hz: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    return sosfilt(butter(6, hz, "lowpass", fs=rate, output="sos"), x)


def synth_noise(kind: str, n: int, rng: np.random.Generator, rate: int = SAMPLE_RATE) -> tuple[np.ndarray, str]:
    """(noise, description) of one synthetic low-frequency noise, high-passed at 20 Hz."""
    if kind == "brown":
        x, detail = np.cumsum(rng.standard_normal(n + rate)), "brown"
        x = highpass(x)[rate:]                                   # drop the filter's start-up second
    elif kind == "lowpass":
        hz = float(rng.uniform(100, 300))
        x, detail = lowpass(rng.standard_normal(n + rate), hz)[rate:], f"lowpass={hz:.0f}Hz"
    elif kind == "hum":
        f0 = float(rng.uniform(25, 120))
        t = np.arange(n) / rate
        drift = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * f0 * np.cumsum(drift) / rate
        x = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k for k in range(1, int(400 // f0) + 1))
        x = x * (1 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t))
        x = x / np.std(x) + 0.3 * lowpass(rng.standard_normal(n + rate), 300)[rate:] / 0.1
        detail = f"hum f0={f0:.0f}Hz"
    else:
        raise ValueError(kind)
    return highpass(x), detail


def low_clips(esc: Path, n: int) -> list[tuple[dict, np.ndarray]]:
    """ESC-50 fold 5 clips (loudest n samples) with at least MIN_LOW_SHARE of power below LOW_HZ."""
    out = []
    for r in sorted(read_esc_meta(esc, 5), key=lambda r: r["filename"]):
        x, rate = sf.read(str(esc / "audio" / r["filename"]), dtype="float64")
        w, _ = loudest_window(to_16k_mono(x, rate), n)
        if low_share(w) >= MIN_LOW_SHARE:
            out.append((r, highpass(w)))
    return out


def build(seed: int) -> list[dict]:
    root = data_root()
    val_rows = read_manifest("validation", root)
    pairs = load_waveforms("validation", root).astype(np.float64)
    n = pairs.shape[-1]
    rng = np.random.default_rng(seed)
    clips = low_clips(esc_dir(), n)
    order = np.concatenate([rng.permutation(len(clips)) for _ in range(-(-len(val_rows) // len(clips)))])
    out = root / SPLIT
    for sub in ("clean", "noisy"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.wav"):
            old.unlink()
    rows = []
    for i, v in enumerate(val_rows):
        clean0 = pairs[i, 1]
        meta, esc_noise = clips[int(order[i])]
        low = LOW_SNRS[int(rng.integers(len(LOW_SNRS)))]
        kind = SYNTH_KINDS[i % len(SYNTH_KINDS)]
        for subset in SUBSETS:
            snr = float(v["target_snr_db"])
            if subset == "synth_low":
                noise, detail = synth_noise(kind, n, rng)
                category = f"synth_{kind}"
            else:
                noise, detail, category = esc_noise, meta["filename"], meta["category"]
                if subset == "esc_low_snr":
                    snr = low
            scale = np.sqrt(np.mean(clean0 ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
            noisy = clean0 + scale * noise
            gain = min(1.0, 0.99 / float(np.max(np.abs(noisy))))
            noisy, ref = noisy * gain, clean0 * gain
            sid = f"{v['segment_id']}_{subset}"
            paths = {"clean": f"{SPLIT}/clean/{sid}.wav", "noisy": f"{SPLIT}/noisy/{sid}_noisy.wav"}
            sf.write(str(root / paths["clean"]), ref, SAMPLE_RATE, subtype="PCM_16")
            sf.write(str(root / paths["noisy"]), noisy, SAMPLE_RATE, subtype="PCM_16")
            rows.append({"segment_id": sid, "clean_path": paths["clean"], "noisy_path": paths["noisy"],
                         "sample_rate_hz": SAMPLE_RATE, "num_samples": n,
                         "noise_category": category, "noise_collection": subset,
                         "noise_code": detail[:60], "noise_difficulty": subset,
                         "target_snr_db": snr, "split": SPLIT, "source_segment_id": v["segment_id"],
                         "detail": detail, "low_share": round(low_share(noise), 3),
                         "gain_db": round(20 * np.log10(gain), 2)})
    with open(out / "samples.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = build(args.seed)
    for s in SUBSETS:
        own = [r for r in rows if r["noise_difficulty"] == s]
        snrs = sorted({r["target_snr_db"] for r in own})
        share = np.mean([r["low_share"] for r in own])
        print(f"{s:11s} {len(own)} pairs, SNR {snrs}, mean share below {LOW_HZ:.0f} Hz {share:.2f}, "
              f"{sum(r['gain_db'] < 0 for r in own)} lowered")
    print(f"-> {(data_root() / SPLIT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
