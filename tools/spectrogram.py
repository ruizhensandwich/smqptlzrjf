"""Look at and listen to models side by side: spectrograms + WAVs + one HTML page.

    python tools/spectrogram.py                                 # submission weights, 4 diverse files
    python tools/spectrogram.py --models runs/<idA> runs/<idB>/best.pt weights/submission.pt
    python tools/spectrogram.py --pick worst --n 5              # files the first model hurts most
    python tools/spectrogram.py --pick best --n 3 --split train
    python tools/spectrogram.py <segment_id> [<segment_id> ...] --models A B
    python tools/spectrogram.py --noisy my_recording.wav [--clean ref.wav]

For each file: Noisy, Clean and every model's output (as denoise.py writes it,
16-bit) in one figure with the same time/frequency axes and colour range (dB
relative to the noisy STFT peak, -80..0 dB, as in the course notebook), the
PESQ-WB/SNR of each clip, and WAVs at one common playback gain so loudness
differences are real. Everything goes to results/listening/<time>/ with an
index.html to open in a browser (audio players under each figure).

Models: a weight file, a run folder (its best.pt) or a run id under runs/.
Labels: --labels, otherwise the run tag or file name.
Picks: diverse (default: one file per input SNR 0/5/10/15 dB, cycling noise
difficulty), worst / best (by the first model's PESQ gain over the input).

The output contains students' recordings: results/listening/ is git-ignored and
blocked by tools/secret_scan.py. Do not publish it.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.checkpoint import load_model  # noqa: E402
from se.data import data_root, load_waveforms, read_manifest  # noqa: E402
from se.inference import as_written, enhance_channels, pick_device  # noqa: E402
from se.metrics import pesq_score, score_pair, snr_db  # noqa: E402

OUT = ROOT / "results" / "listening"
N_FFT, HOP = 512, 128


# --------------------------------------------------------------------------
def resolve_model(arg: str) -> tuple[Path, str]:
    p = Path(arg)
    for cand in (p, ROOT / p, ROOT / "runs" / arg):
        if cand.is_dir() and (cand / "best.pt").is_file():
            return cand / "best.pt", cand.name.split("_", 1)[-1]
        if cand.is_file():
            label = cand.parent.name.split("_", 1)[-1] if cand.name == "best.pt" else cand.stem
            return cand, label
    sys.exit(f"model not found: {arg} (weight file, run folder or run id)")


def stft_db(wave: np.ndarray, ref: float):
    wave = np.asarray(wave, dtype=np.float64)
    f, t, z = signal.stft(wave, fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
                          noverlap=N_FFT - HOP, nfft=N_FFT, boundary="zeros", padded=True)
    return f, t, 20 * np.log10(np.maximum(np.abs(z), 1e-12) / ref)


def figure(clips: dict[str, np.ndarray], scores: dict[str, str], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = max(np.abs(signal.stft(next(iter(clips.values())), fs=SAMPLE_RATE, window="hann",
                                 nperseg=N_FFT, noverlap=N_FFT - HOP)[2]).max(), 1e-12)
    cols = 2
    rows = (len(clips) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    mesh = None
    for ax, (name, wave) in zip(axes.flat, clips.items()):
        f, t, db = stft_db(wave, ref)
        mesh = ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set(title=f"{name}  {scores.get(name, '')}", xlim=(0, len(wave) / SAMPLE_RATE),
               ylim=(0, SAMPLE_RATE / 2000))
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (kHz)")
    for ax in list(axes.flat)[len(clips):]:
        ax.set_visible(False)
    fig.colorbar(mesh, ax=list(axes.flat)[:len(clips)], label="dB relative to noisy STFT peak")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
def pick_files(rows: list[dict], waves: np.ndarray, how: str, n: int, model, device,
               workers: int) -> list[int]:
    if how == "diverse":
        out, diff = [], ["easy", "moderate", "hard"]
        for k, snr in enumerate(["0.0", "5.0", "10.0", "15.0"] * ((n + 3) // 4)):
            want = diff[k % 3]
            cand = [i for i, r in enumerate(rows) if r["target_snr_db"] == snr and i not in out]
            pref = [i for i in cand if rows[i]["noise_difficulty"] == want] or cand
            if pref:
                out.append(pref[len(out) * 7 % len(pref)])   # spread over speakers/noises
            if len(out) == n:
                break
        return out
    noisy, clean = waves[:, 0].astype(np.float64), waves[:, 1].astype(np.float64)
    est = [as_written(enhance_channels(model, x[:, None], SAMPLE_RATE, device)[:, 0], SAMPLE_RATE)
           for x in noisy]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        base = [p for p, _, _ in pool.map(score_pair, zip(clean, noisy), chunksize=8)]
        mod = [p for p, _, _ in pool.map(score_pair, zip(clean, est), chunksize=8)]
    gain = np.array(mod) - np.array(base)
    order = np.argsort(gain)
    return [int(i) for i in (order[:n] if how == "worst" else order[::-1][:n])]


def page(items: list[dict], models: list[str], out: Path) -> None:
    css = ("body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}"
           "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;"
           "text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%}"
           ".clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}"
           ".clips div{border:1px solid #ddd;padding:6px}audio{width:100%}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Listening</title><style>{css}</style>",
             "<h1>모델 청취·스펙트로그램 비교</h1>",
             f"<p>모델: {html.escape(', '.join(models))} · 모든 음성은 같은 재생 게인 · "
             "학생 녹음이 들어 있으니 공유하지 마세요.</p>"]
    for it in items:
        parts.append(f"<h2>{html.escape(it['title'])}</h2>")
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v[0]}</td><td>{v[1]}</td><td>{v[2]}</td></tr>"
                       for k, v in it["table"].items())
        parts.append("<table><tr><th>음성</th><th>PESQ-WB</th><th>ΔPESQ</th><th>SNR (dB)</th></tr>"
                     f"{rows}</table>")
        parts.append(f"<img src='{it['png']}' alt='spectrograms'>")
        parts.append("<div class='clips'>" + "".join(
            f"<div><b>{html.escape(k)}</b><br><audio controls preload='none' src='{w}'></audio></div>"
            for k, w in it["wavs"].items()) + "</div>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", nargs="*", help="segment ids from samples.csv")
    ap.add_argument("--models", nargs="+", default=["weights/submission.pt"])
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--split", choices=["validation", "train"], default="validation")
    ap.add_argument("--pick", choices=["diverse", "worst", "best"], default="diverse")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--noisy", type=Path, help="any 16 kHz WAV instead of the dataset")
    ap.add_argument("--clean", type=Path, help="reference for --noisy (optional)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    resolved = [resolve_model(m) for m in args.models]
    labels = args.labels or [lab for _, lab in resolved]
    if len(labels) != len(resolved) or len(set(labels)) != len(labels):
        sys.exit("--labels: one distinct label per model")
    models = [load_model(p, device)[0] for p, _ in resolved]

    # inputs: (title, noisy, clean or None, file stem)
    inputs = []
    if args.noisy:
        x, sr = sf.read(str(args.noisy), dtype="float64", always_2d=True)
        if sr != SAMPLE_RATE:
            sys.exit("--noisy must be 16 kHz")
        c = None
        if args.clean:
            c, _ = sf.read(str(args.clean), dtype="float64", always_2d=True)
            c = c[: len(x), 0]
        inputs.append((args.noisy.name, x[:, 0], c, safe(args.noisy.stem)))
    else:
        root = data_root()
        rows = read_manifest(args.split, root)
        waves = load_waveforms(args.split, root)
        if args.segments:
            idx = {r["segment_id"]: i for i, r in enumerate(rows)}
            missing = [s for s in args.segments if s not in idx]
            if missing:
                sys.exit(f"not in {args.split}: {missing}")
            chosen = [idx[s] for s in args.segments]
        else:
            chosen = pick_files(rows, waves, args.pick, args.n, models[0], device, args.workers)
        for i in chosen:
            r = rows[i]
            title = (f"{r['segment_id']} | {r['noise_category']} ({r['noise_code']}, "
                     f"{r['noise_difficulty']}) | input SNR {float(r['target_snr_db']):g} dB")
            inputs.append((title, waves[i, 0].astype(np.float64), waves[i, 1].astype(np.float64),
                           r["segment_id"]))

    out = (args.out or OUT / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for title, noisy, clean, stem in inputs:
        clips = {"Noisy": noisy}
        if clean is not None:
            clips["Clean"] = clean
        for lab, model in zip(labels, models):
            clips[lab] = as_written(enhance_channels(model, noisy[:, None], SAMPLE_RATE, device)[:, 0],
                                    SAMPLE_RATE)
        table, scores = {}, {}
        base = pesq_score(clean, noisy)[0] if clean is not None else None
        for name, w in clips.items():
            if clean is None or name == "Clean":
                table[name] = ("-", "-", "-")
                continue
            p = pesq_score(clean, w)[0]
            s = snr_db(clean, w)
            table[name] = (f"{p:.3f}", f"{p - base:+.3f}" if name != "Noisy" else "", f"{s:.2f}")
            scores[name] = f"PESQ {p:.2f} / SNR {s:.1f} dB"
        figure(clips, scores, title, out / f"{stem}.png")
        gain = 0.99 / max(max(np.max(np.abs(w)) for w in clips.values()), 1e-8)
        wavs = {}
        for name, w in clips.items():
            fn = f"{stem}__{safe(name)}.wav"
            sf.write(str(out / fn), np.clip(w * gain, -1, 1), SAMPLE_RATE, subtype="PCM_16")
            wavs[name] = fn
        items.append({"title": title, "table": table, "png": f"{stem}.png", "wavs": wavs})
        print(title)
        for name, (p, d, s) in table.items():
            if p != "-":
                print(f"  {name:<20} PESQ {p}  {d:>7}  SNR {s} dB")
    page(items, labels, out)
    try:
        shown = out.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(out)
    print(f"-> {shown}/index.html (open in a browser; contains recordings, do not publish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
