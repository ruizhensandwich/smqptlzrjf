"""Is  noisy - clean  a usable noise recording?  (check before dynamic remixing)

    python tools/check_mixing.py                 # train split
    python tools/check_mixing.py --split validation

Dynamic remixing (se/data.py, RemixedSpeech) takes the noise of a train pair as
noisy - clean and adds it to another clean utterance. That is only right when
the noisy file is the clean file plus noise, sample for sample. If the two are
shifted or the speech level was changed, the difference still contains speech
and the model would be taught to remove it. Per file this measures

    lag         shift (samples) at the peak of the noisy/clean cross-correlation   -> 0
    gain        least-squares speech gain <noisy, clean> / <clean, clean>          -> 1
    corr        correlation coefficient of (noisy - clean) and clean               -> 0
    snr_db      10 log10(|clean|^2 / |noisy - clean|^2), compared with the manifest's target
    noise_rms, quiet_frac (20 ms noise frames below -70 dBFS), clip_frac (|noisy| at full scale)

Writes results/mixing_check/<split>.csv (one row per file, numbers only, no
audio) and summary.json. Files outside the limits (se/data.py, REMIX_LIMITS)
are never remixing material: RemixedSpeech applies the same test when it
loads the data and uses those files as original pairs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se.data import (REMIX_LIMITS, load_waveforms, mixing_stats,  # noqa: E402
                     read_manifest, remix_problems)

OUT = ROOT / "results" / "mixing_check"
LIMITS = {"max_abs_lag": 0, **REMIX_LIMITS}      # limits chosen from the first run over train


def lag_samples(noisy: np.ndarray, clean: np.ndarray) -> int:
    """Shift at the cross-correlation peak (> 0: speech is later in noisy than in clean)."""
    size = 1 << (2 * noisy.size - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(noisy.astype(np.float64), size)
                      * np.conj(np.fft.rfft(clean.astype(np.float64), size)), size)
    k = int(np.argmax(xc))
    return k if k < size // 2 else k - size


def measure(noisy: np.ndarray, clean: np.ndarray) -> dict:
    return {"lag": lag_samples(noisy, clean), **mixing_stats(noisy, clean)}


def reasons(m: dict) -> list[str]:
    lag = ["lag"] if abs(m["lag"]) > LIMITS["max_abs_lag"] else []
    return lag + remix_problems(m, m["target_snr_db"])


def _dist(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [0, 1, 5, 50, 95, 99, 100])
    return dict(zip(["min", "p1", "p5", "median", "p95", "p99", "max"], (round(float(x), 5) for x in q)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    args = ap.parse_args(argv)

    manifest, waves = read_manifest(args.split), load_waveforms(args.split)
    rows = []
    for r, (noisy, clean) in zip(manifest, waves):
        m = {"segment_id": r["segment_id"], "noise_category": r["noise_category"],
             "target_snr_db": float(r["target_snr_db"]), **measure(noisy, clean)}
        m["snr_error_db"] = m["snr_db"] - m["target_snr_db"]
        m["excluded"] = ";".join(reasons(m))
        rows.append(m)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for m in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in m.items()})
    excluded = [m for m in rows if m["excluded"]]
    by_reason: dict[str, int] = {}
    for m in excluded:
        for why in m["excluded"].split(";"):
            by_reason[why] = by_reason.get(why, 0) + 1
    summary = {"split": args.split, "files": len(rows), "usable": len(rows) - len(excluded),
               "excluded": len(excluded), "excluded_by_reason": by_reason, "limits": LIMITS,
               "lag_counts": {str(k): int(v) for k, v in
                              zip(*np.unique([m["lag"] for m in rows], return_counts=True))},
               "distribution": {k: _dist([m[k] for m in rows]) for k in
                                ("gain", "corr", "snr_error_db", "noise_rms", "quiet_frac",
                                 "clip_frac")}}
    path = OUT / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[args.split] = summary
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
