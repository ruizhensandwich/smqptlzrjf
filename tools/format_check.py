"""Does a model keep its score when the input file looks different from the course files?

    python tools/format_check.py --weights W [--n 0]     # on the VM (GPU); --n 0 = all 164

The hidden test files may differ from the 3 s course files in ways that are not "noise":
length, level, silence around the speech. The model should not care. For each validation
file this runs denoise.py's code path (se.inference: enhance, clip, write 16-bit, read back)
on several versions of the same file and scores the part that corresponds to the original
3 s against the same clean reference:

  base       the file as it is
  quiet      30 dB lower (both noisy and clean; 16-bit rounding included, as in a real file)
  loud       scaled so that the noisy peak is at 0.99 of full scale
  silence    1 s of digital silence (exact zeros) before and after
  hiss       1 s of very quiet noise (-70 dBFS) before and after: tells a problem with exact
             zeros apart from a problem with the extra length
  long12     4 files joined into one 12 s file, cut back into 3 s pieces after enhancement
  long30     10 files joined into one 30 s file, same
  dc         a DC offset of 0.01 added to the noisy input

Prints the mean PESQ-WB per version and its difference from base. A drop of more than a few
hundredths means the model or its wrapper handles that kind of file worse.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.checkpoint import load_model  # noqa: E402
from se.data import load_waveforms  # noqa: E402
from se.inference import as_written, enhance_channels, pick_device  # noqa: E402

VERSIONS = ("base", "quiet", "loud", "silence", "hiss", "long12", "long30", "dc")


def _pesq(pair):
    from pesq import pesq
    clean, out = pair
    try:
        return float(pesq(SAMPLE_RATE, clean, out, "wb"))
    except Exception:  # noqa: BLE001
        return float("nan")


def run(model, device, x: np.ndarray) -> np.ndarray:
    """denoise.py's path for one mono signal: enhance, clip, 16-bit write and read back."""
    return as_written(enhance_channels(model, x[:, None], SAMPLE_RATE, device)[:, 0], SAMPLE_RATE)


def as_file(x: np.ndarray) -> np.ndarray:
    """A signal as a 16-bit file stores it (the input side of a real file)."""
    return as_written(np.clip(x, -1.0, 1.0 - 2 ** -15), SAMPLE_RATE)


def outputs(model, device, noisy: np.ndarray, clean: np.ndarray, version: str):
    """[(clean_ref, output)] for every file, the output cut to that file's 3 s."""
    n = noisy.shape[1]
    pairs = []
    if version in ("long12", "long30"):
        k = 4 if version == "long12" else 10
        for a in range(0, len(noisy), k):
            group = noisy[a:a + k]
            y = run(model, device, as_file(group.reshape(-1)))
            pairs += [(clean[a + j], y[j * n:(j + 1) * n]) for j in range(len(group))]
        return pairs
    for x, c in zip(noisy, clean):
        if version == "quiet":
            g = 10 ** (-30 / 20)
            pairs.append((as_file(c * g), run(model, device, as_file(x * g))))
        elif version == "loud":
            g = 0.99 / np.max(np.abs(x))
            pairs.append((c * g, run(model, device, as_file(x * g))))
        elif version == "silence":
            z = np.zeros(SAMPLE_RATE)
            y = run(model, device, as_file(np.concatenate([z, x, z])))
            pairs.append((c, y[SAMPLE_RATE:SAMPLE_RATE + n]))
        elif version == "hiss":
            z = np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 10 ** (-70 / 20)
            y = run(model, device, as_file(np.concatenate([z, x, z])))
            pairs.append((c, y[SAMPLE_RATE:SAMPLE_RATE + n]))
        elif version == "dc":
            pairs.append((c, run(model, device, as_file(x + 0.01))))
        else:
            pairs.append((c, run(model, device, as_file(x))))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="validation files (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    model, _ = load_model(args.weights.resolve(), device)
    w = load_waveforms("validation").astype(np.float64)
    if args.n:
        w = w[:args.n]
    noisy, clean = w[:, 0], w[:, 1]
    base = None
    with ProcessPoolExecutor(args.workers) as pool:
        for v in VERSIONS:
            scores = np.array(list(pool.map(_pesq, outputs(model, device, noisy, clean, v),
                                            chunksize=8)))
            mean = float(np.nanmean(scores))
            base = mean if base is None else base
            fails = int(np.isnan(scores).sum())
            print(f"{v:8s} PESQ {mean:.4f}  {mean - base:+.4f}"
                  + (f"  ({fails} PESQ errors)" if fails else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
