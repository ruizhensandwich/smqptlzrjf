"""Build the "hard" split: validation speech under conditions harder than the course data.

    python tools/make_hardset.py            # needs data/newnoise's ESC-50 fold 5 clips and,
                                            # for the babble subset, Zeroth-Korean in data/external/zeroth/full/

Why: round 1 scored about 2.4 PESQ on the hidden test set (said to be harder than the first
challenge's) against 3.02 on validation and 3.05 on the newnoise split, so neither predicts it.
This split asks which kind of difficulty costs the most. Every subset keeps the 164 validation
clean files and changes ONE thing, so the subsets can be compared with each other and with
validation:

  lowsnr      validation's own noise (noisy - clean of the same pair), SNR -5 or 0 dB
              -> only the SNR is harder (validation: 0/5/10/15 dB)
  newlow      ESC-50 fold 5 noise (unseen, as in the newnoise split) at -5 or 0 dB
              -> unseen noise AND low SNR
  babble      several talkers at once (3 to 6) at the file's own validation SNR. The talkers are
              the 10 Zeroth-Korean speakers left out of the training speech pool for background
              noise (data/external/zeroth/speakers.csv, used = 0), so no model has trained on them
              -> a noise that is speech
  reverb      the clean speech convolved with a synthetic room response (exponentially decaying
              noise, RT60 0.3-0.8 s, direct sound at sample 0), plus validation's own noise at the
              file's SNR. The reference stays the DRY clean file
              -> room echo the model has to remove (if the test set has none, this subset says nothing)

SNR = power(speech in the mixture) / power(noise) over the whole file, as in the course data;
noisy and clean are lowered together if the mixture would clip (column gain_db).
ESC-50 fold 5 is never used for training, Zeroth's left-out speakers are not in the training
pool; nothing here is training material.

Output: <data>/hard/{clean,noisy}/*.wav (PCM_16) and samples.csv in the course format with the
subset in noise_difficulty, read by evaluate.py --split hard (which prints one line per subset).
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
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import EXTERNAL_SPEECH, data_root, load_waveforms, read_manifest  # noqa: E402
from tools.make_newnoise import (esc_dir, loudest_window, mix, plan, read_esc_meta,  # noqa: E402
                                 to_16k_mono)

SPLIT = "hard"
SUBSETS = ("lowsnr", "newlow", "babble", "reverb")
LOW_SNRS = (-5.0, 0.0)
TALKERS = (3, 6)
RT60 = (0.3, 0.8)
EXTRA_COLUMNS = ["source_segment_id", "detail", "gain_db"]


def room_response(rt60: float, rng: np.random.Generator, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic room impulse response: direct sound 1 at sample 0, then decaying noise.

    The tail starts 2.5 ms after the direct sound and decays by 60 dB in rt60 seconds;
    its energy is set so the direct-to-reverberant ratio follows from rt60 (longer = wetter).
    """
    n = int(rt60 * rate)
    t = np.arange(n) / rate
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)     # 10^(-60/20) at t = rt60
    tail[:int(0.0025 * rate)] = 0.0
    tail *= np.sqrt(rt60 / 0.3) * 0.5 / np.sqrt(np.sum(tail ** 2))
    h = tail
    h[0] = 1.0
    return h


def reverberate(clean: np.ndarray, h: np.ndarray) -> np.ndarray:
    return fftconvolve(clean, h)[:clean.size]


def babble(files: list[Path], n: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """k talkers from different files, each its loudest n samples at the same power, summed."""
    out, names = np.zeros(n), []
    for i in rng.choice(len(files), k, replace=False):
        x, rate = sf.read(str(files[i]), dtype="float64")
        w, _ = loudest_window(to_16k_mono(x, rate), n)
        out += w / np.sqrt(np.mean(w ** 2) + 1e-12)
        names.append(files[i].stem)
    return out, "+".join(names)


def left_out_talkers(root: Path) -> list[Path]:
    folder = (root / EXTERNAL_SPEECH["zeroth"]).parent
    with open(folder / "speakers.csv", encoding="utf-8") as f:
        left = {r["speaker"] for r in csv.DictReader(f) if r["used"] == "0"}
    files = sorted(p for p in (folder / "full").rglob("*.flac") if p.name.split("_")[0] in left)
    if len({p.name.split("_")[0] for p in files}) < TALKERS[1]:
        raise FileNotFoundError(f"need Zeroth files of the left-out speakers under {folder / 'full'}")
    return files


def build(seed: int) -> list[dict]:
    root = data_root()
    val_rows = read_manifest("validation", root)
    pairs = load_waveforms("validation", root).astype(np.float64)
    n = pairs.shape[-1]
    rng = np.random.default_rng(seed)
    esc = esc_dir()
    chosen = plan(val_rows, read_esc_meta(esc, 5), seed + 1)
    talkers = left_out_talkers(root)
    out = root / SPLIT
    for sub in ("clean", "noisy"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.wav"):
            old.unlink()
    rows = []
    for i, v in enumerate(val_rows):
        noisy0, clean0 = pairs[i]
        own_noise = noisy0 - clean0
        low = LOW_SNRS[int(rng.integers(len(LOW_SNRS)))]
        for subset in SUBSETS:
            speech, ref, category, detail = clean0, clean0, v["noise_category"], v["noise_code"]
            snr = float(v["target_snr_db"])
            if subset == "lowsnr":
                noise, snr = own_noise, low
            elif subset == "newlow":
                x, rate = sf.read(str(esc / "audio" / chosen[i]["filename"]), dtype="float64")
                noise, _ = loudest_window(to_16k_mono(x, rate), n)
                snr, category, detail = low, chosen[i]["category"], chosen[i]["filename"]
            elif subset == "babble":
                k = int(rng.integers(TALKERS[0], TALKERS[1] + 1))
                noise, detail = babble(talkers, n, k, rng)
                category = f"babble_{k}"
            else:
                rt60 = float(rng.uniform(*RT60))
                speech = reverberate(clean0, room_response(rt60, rng))
                noise, detail = own_noise, f"rt60={rt60:.2f}"
            scale = np.sqrt(np.mean(speech ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
            noisy = speech + scale * noise
            gain = min(1.0, 0.99 / float(np.max(np.abs(noisy))))
            noisy, ref = noisy * gain, ref * gain
            sid = f"{v['segment_id']}_{subset}"
            paths = {"clean": f"{SPLIT}/clean/{sid}.wav", "noisy": f"{SPLIT}/noisy/{sid}_noisy.wav"}
            sf.write(str(root / paths["clean"]), ref, SAMPLE_RATE, subtype="PCM_16")
            sf.write(str(root / paths["noisy"]), noisy, SAMPLE_RATE, subtype="PCM_16")
            rows.append({"segment_id": sid, "clean_path": paths["clean"], "noisy_path": paths["noisy"],
                         "sample_rate_hz": SAMPLE_RATE, "num_samples": n,
                         "noise_category": category, "noise_collection": subset,
                         "noise_code": detail[:60], "noise_difficulty": subset,
                         "target_snr_db": snr, "split": SPLIT, "source_segment_id": v["segment_id"],
                         "detail": detail, "gain_db": round(20 * np.log10(gain), 2)})
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
        print(f"{s:7s} {len(own)} pairs, SNR {snrs}, {sum(r['gain_db'] < 0 for r in own)} lowered")
    print(f"-> {(data_root() / SPLIT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
