"""Build the external speech pool for training: clean 3 s clips of Zeroth-Korean.

    python tools/make_external_speech.py                    # data/external/zeroth/full/ -> pool
    python tools/make_external_speech.py --min-range-db 45 --per-speaker 60

Zeroth-Korean (OpenSLR 40, https://www.openslr.org/40/, CC BY 4.0): read Korean speech,
115 speakers (105 train + 10 test), 22,720 utterances, 16 kHz FLAC. Get the archive with
    bash tools/fetch_segments.sh https://www.openslr.org/resources/40/zeroth_korean.tar.gz \
         data/external/zeroth/zeroth_korean.tar.gz 10339720618
and extract it into data/external/zeroth/full/.

The clips become the *clean target* of training items, so recordings with background noise
are left out:
  - per file, the level of 20 ms frames in dB: floor = 5th percentile, speech = 95th
    percentile, range = speech - floor. A quiet room gives a large range;
  - a speaker whose median range is below --min-range-db is left out entirely
    (noise of the room or microphone is in every file of that speaker);
  - a file that touches full scale (clipped) is left out.
Each remaining speaker gives the same number of clips at most (--per-speaker), so no voice
dominates: files in a fixed random order, up to two 3 s windows per file, only windows
that hold speech (power at least a quarter of the file's).

Output next to the data, read by se/data.py (config data.external_speech = "zeroth"):
    train_speech_16k.npy     [clips, 48000] int16
    train_speech_16k.csv     speaker, file, start_s            (row i = clip i)
    speakers.csv             every speaker: files, clipped files, median floor / range, used, clips
Nothing here is committed: data/ is git-ignored.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import EXTERNAL_SPEECH, data_root  # noqa: E402

SECONDS = 3
FRAME = 320                     # 20 ms
CLIP_PEAK = 0.999
MIN_WINDOW_SHARE = 0.25         # a window's power against the whole file's
WINDOWS_PER_FILE = 2


def file_stats(x: np.ndarray) -> dict | None:
    """Noise floor, speech level and their distance in dB, and the peak (None: too short)."""
    n = x.size // FRAME
    if n < 20:
        return None
    db = 10 * np.log10(np.mean(x[:n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)
    floor, speech = np.percentile(db, [5, 95])
    return {"floor_db": float(floor), "range_db": float(speech - floor),
            "peak": float(np.max(np.abs(x)))}


def speaker_of(path: Path) -> str:
    return path.name.split("_")[0]


def measure(path: str) -> dict | None:
    try:
        x, rate = sf.read(path, dtype="float64")
    except (RuntimeError, OSError):
        return None
    if rate != SAMPLE_RATE or x.ndim != 1:
        return None
    s = file_stats(x)
    return s and {**s, "path": path, "speaker": speaker_of(Path(path))}


def judge_speakers(stats: list[dict], min_range_db: float) -> list[dict]:
    """One row per speaker; used = its median range reaches min_range_db."""
    rows = []
    for spk in sorted({s["speaker"] for s in stats}):
        own = [s for s in stats if s["speaker"] == spk]
        med = float(np.median([s["range_db"] for s in own]))
        rows.append({"speaker": spk, "files": len(own),
                     "clipped_files": sum(s["peak"] >= CLIP_PEAK for s in own),
                     "median_floor_db": round(float(np.median([s["floor_db"] for s in own])), 1),
                     "median_range_db": round(med, 1), "used": int(med >= min_range_db),
                     "clips": 0})
    return rows


def speech_windows(x: np.ndarray, n: int, most: int = WINDOWS_PER_FILE) -> list[int]:
    """Starts of up to `most` non-overlapping n-sample windows that hold speech, loudest first."""
    count = x.size // n
    if count == 0:
        return []
    power = np.mean(x[:count * n].reshape(count, n) ** 2, axis=1)
    order = [k for k in np.argsort(-power, kind="stable")
             if power[k] >= MIN_WINDOW_SHARE * np.mean(x ** 2) and power[k] > 0]
    return sorted(int(k) * n for k in order[:most])


def build(stats: list[dict], speakers: list[dict], per_speaker: int, seed: int):
    n = SECONDS * SAMPLE_RATE
    rng = np.random.default_rng(seed)
    clips, rows = [], []
    for sp in speakers:
        if not sp["used"]:
            continue
        own = sorted((s for s in stats if s["speaker"] == sp["speaker"] and s["peak"] < CLIP_PEAK),
                     key=lambda s: s["path"])
        for k in rng.permutation(len(own)):
            if sp["clips"] >= per_speaker:
                break
            x, _ = sf.read(own[k]["path"], dtype="float64")
            for start in speech_windows(x, n)[:per_speaker - sp["clips"]]:
                clips.append(np.round(x[start:start + n] * 32768).astype(np.int16))
                rows.append({"speaker": sp["speaker"], "file": Path(own[k]["path"]).name,
                             "start_s": round(start / SAMPLE_RATE, 2)})
                sp["clips"] += 1
    return np.stack(clips), rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-range-db", type=float, default=45.0,
                    help="leave out speakers whose median speech-minus-floor range is below this")
    ap.add_argument("--per-speaker", type=int, default=60, help="clips per speaker at most")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    out = data_root() / EXTERNAL_SPEECH["zeroth"]
    files = sorted(str(p) for p in (out.parent / "full").rglob("*.flac"))
    if not files:
        raise SystemExit(f"no FLAC under {out.parent / 'full'}: see the top of this file")
    with Pool(args.workers) as pool:
        measured = pool.map(measure, files, chunksize=64)
    stats = [s for s in measured if s]
    speakers = judge_speakers(stats, args.min_range_db)
    clips, rows = build(stats, speakers, args.per_speaker, args.seed)
    np.save(out, clips, allow_pickle=False)
    write_csv(out.with_suffix(".csv"), rows)
    write_csv(out.parent / "speakers.csv", speakers)
    left = [s for s in speakers if not s["used"]]
    print(f"{len(files)} files, {len(files) - len(stats)} unreadable or too short, "
          f"{sum(s['clipped_files'] for s in speakers)} clipped")
    print(f"speakers: {len(speakers)} measured, {len(left)} left out (median range < "
          f"{args.min_range_db:g} dB): "
          + ", ".join(f"{s['speaker']} ({s['median_range_db']})" for s in left))
    per = [s["clips"] for s in speakers if s["used"]]
    print(f"{len(rows)} clips of {len(per)} speakers (per speaker {min(per)}-{max(per)}) "
          f"-> {out.as_posix()} ({out.stat().st_size / 2 ** 20:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
