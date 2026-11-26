"""Study how to combine several models' outputs: alignment, level, weights, domain.

    python tools/mix_study.py dump  --members L=weights/a.pt M=weights/b.pt --splits validation newnoise hard lowband
    python tools/mix_study.py align                       # lag, polarity, level of every member; lags between members
    python tools/mix_study.py gain  --member L [--score --const 1.48 --lo 1.2 --hi 1.8]
    python tools/mix_study.py mix   --grid [--gain L:proj:1.48:1.2:1.8]
    python tools/mix_study.py mix   --combos L0.6+M0.2+F0.2 --method mag --phase L
    python tools/mix_study.py report
    python tools/mix_study.py pcs   --targets LFM=L0.5+F0.2+M0.3 L=L1.0 --gain L:proj:1.484:1.35:1.65
    python tools/mix_study.py pcs-report

Each member's output is computed once per split (dump, on the GPU) and kept in
runs/mix_study/<split>/<member>.npy; everything else works on those files. Scores are of the
16-bit file denoise.py would write (clip, PCM_16, read back), as in evaluate.py.

Level correction ("gain"). A model trained with a scale-invariant loss can return the speech
at another level than the input's (TF-Locoformer DNS: about 3.4 dB too quiet). PESQ does not
see that, SNR and a waveform average do. Three corrections of one member are compared:
  const   one number for every file
  proj    per file, the least-squares gain of the output towards the NOISY input,
          g = <noisy, y> / <y, y>  (no clean needed: if y is c * speech and the noise is
          uncorrelated with it, this is 1 / c), clamped to [lo, hi], const if y is silent
  oracle  the same towards the clean reference (not available at run time; the yardstick)

Combination ("mix"). wave = weighted mean of the (level-corrected) waveforms. mag = weighted
mean of the STFT magnitudes (400/100, Hann) with the phase of one member. A mean of complex
spectra is the waveform mean, so it is not a separate method.

The splits contain students' recordings: runs/ is git-ignored and nothing here is published.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from se import SAMPLE_RATE  # noqa: E402
from se.data import load_waveforms, read_manifest  # noqa: E402

CACHE = ROOT / "runs" / "mix_study"
SPLITS = ("validation", "newnoise", "hard", "lowband")
SELECT_ON = ("validation", "newnoise")          # weights are chosen on these; the others only confirm
FULL_SCALE = 1.0 - 2.0 ** -15


# ---------------------------------------------------------------- scoring
def _score(pair):
    """(PESQ-WB, SNR dB) of the 16-bit file that would be written."""
    from pesq import pesq
    from se.inference import as_written
    from se.metrics import snr_db
    clean, y = pair
    out = as_written(np.clip(y, -1.0, FULL_SCALE), SAMPLE_RATE)
    try:
        p = float(pesq(SAMPLE_RATE, clean, out, "wb"))
    except Exception:  # noqa: BLE001
        p = float("nan")
    return p, snr_db(clean, out)


def score_all(pool, clean: np.ndarray, outs: np.ndarray) -> np.ndarray:
    """[N, 2] (PESQ, SNR)."""
    return np.array(list(pool.map(_score, zip(clean.astype(np.float64), outs.astype(np.float64)),
                                  chunksize=16)))


def load_split(split: str):
    w = load_waveforms(split)
    return w[:, 0].astype(np.float64), w[:, 1].astype(np.float64)


def member(split: str, name: str) -> np.ndarray:
    return np.load(CACHE / split / f"{name}.npy").astype(np.float64)


def members_available(split: str = "validation") -> list[str]:
    return sorted(p.stem for p in (CACHE / split).glob("*.npy"))


# ---------------------------------------------------------------- level correction
def proj_gain(x: np.ndarray, y: np.ndarray, const: float, lo: float, hi: float) -> float:
    """Least-squares gain of y towards the noisy input x, clamped; const when y is silent."""
    e = float(y @ y)
    if e <= 1e-10 * y.size:
        return const
    g = float(x @ y) / e
    return const if not np.isfinite(g) else float(min(max(g, lo), hi))


def apply_gain(spec: str | None, name: str, noisy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """spec: 'L:const:1.48' or 'L:proj:1.48:1.2:1.8' (member, method, const[, lo, hi]); other members unchanged."""
    if not spec:
        return y
    parts = spec.split(":")
    if parts[0] != name:
        return y
    if parts[1] == "const":
        return y * float(parts[2])
    const, lo, hi = (float(v) for v in parts[2:5])
    g = np.array([proj_gain(x, o, const, lo, hi) for x, o in zip(noisy, y)])
    return y * g[:, None]


# ---------------------------------------------------------------- dump
def cmd_dump(args) -> None:
    import torch
    from se.checkpoint import load_model
    from se.inference import enhance, pick_device
    device = pick_device(args.device)
    for item in args.members:
        name, path = item.split("=", 1)
        model, _ = load_model(ROOT / path, device)
        for split in args.splits:
            out = CACHE / split / f"{name}.npy"
            if out.exists() and not args.force:
                print(f"{split}/{name}: kept"); continue
            noisy, _ = load_split(split)
            if args.n:
                noisy = noisy[:args.n]
            y = np.stack([enhance(model, x, SAMPLE_RATE, device) for x in noisy]).astype(np.float32)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, y)
            print(f"{split}/{name}: {y.shape} from {path}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- align
def _lag_sign(a: np.ndarray, b: np.ndarray, max_lag: int = 800) -> tuple[int, int]:
    """Lag (samples, a later than b = positive) and sign at the largest |cross-correlation|."""
    from scipy.signal import correlate
    c = correlate(a, b, mode="full", method="fft")
    mid = b.size - 1
    seg = c[mid - max_lag:mid + max_lag + 1]
    k = int(np.argmax(np.abs(seg)))
    return k - max_lag, int(np.sign(seg[k]))


def pct(v, qs=(0, 1, 5, 50, 95, 99, 100)) -> str:
    return " ".join(f"{np.percentile(v, q):.3f}" for q in qs)


def cmd_align(args) -> None:
    for split in args.splits:
        _, clean = load_split(split)
        names = members_available(split)
        outs = {m: member(split, m)[:len(clean)] for m in names}
        clean = clean[:len(next(iter(outs.values())))]
        print(f"\n[{split}] {len(clean)} files   (lag in samples, + = output later than clean)")
        for m in names:
            ls = np.array([_lag_sign(y, c) for y, c in zip(outs[m], clean)])
            level = 10 * np.log10((outs[m] ** 2).mean(1) / (clean ** 2).mean(1))
            g = (clean * outs[m]).sum(1) / (outs[m] ** 2).sum(1)
            lags, counts = np.unique(ls[:, 0], return_counts=True)
            top = ", ".join(f"{a:+d}:{b}" for a, b in sorted(zip(lags, counts), key=lambda t: -t[1])[:5])
            print(f"  {m}: lag 0 in {np.mean(ls[:, 0] == 0) * 100:.1f} % (lag:files {top}), |lag| max {np.abs(ls[:, 0]).max()}, "
                  f"negative polarity {int(np.sum(ls[:, 1] < 0))}, level vs clean mean {level.mean():+.2f} dB "
                  f"(p5 {np.percentile(level, 5):+.2f}, p95 {np.percentile(level, 95):+.2f}), oracle gain median {np.median(g):.3f}")
        for a, b in itertools.combinations(names, 2):
            ls = np.array([_lag_sign(x, y) for x, y in zip(outs[a], outs[b])])
            print(f"  {a} vs {b}: lag 0 in {np.mean(ls[:, 0] == 0) * 100:.1f} %, |lag| max {np.abs(ls[:, 0]).max()}, "
                  f"negative polarity {int(np.sum(ls[:, 1] < 0))}")


# ---------------------------------------------------------------- gain
def cmd_gain(args) -> None:
    m = args.member
    const = args.const
    if (CACHE / "train" / f"{m}.npy").exists():
        noisy, clean = load_split("train")
        y = member("train", m)
        go = (clean * y).sum(1) / (y ** 2).sum(1)
        gp = (noisy * y).sum(1) / (y ** 2).sum(1)
        print(f"[train] {len(y)} files  oracle gain pct(0 1 5 50 95 99 100): {pct(go)}   proj: {pct(gp)}   "
              f"proj/oracle: {pct(gp / go)}")
        if const is None:
            const = float(np.median(go))
            print(f"const = median oracle gain on train = {const:.4f}")
    if const is None:
        sys.exit("give --const or dump the member on train")
    pool = ProcessPoolExecutor(args.workers) if args.score else None
    for split in args.splits:
        noisy, clean = load_split(split)
        y = member(split, m)
        noisy, clean = noisy[:len(y)], clean[:len(y)]
        rows = read_manifest(split)[:len(y)]
        snr = np.array([float(r["target_snr_db"]) for r in rows])
        go = (clean * y).sum(1) / (y ** 2).sum(1)
        gp = (noisy * y).sum(1) / (y ** 2).sum(1)
        print(f"\n[{split}] {len(y)} files   percentiles 0 1 5 50 95 99 100")
        print(f"  oracle gain      {pct(go)}")
        print(f"  proj gain        {pct(gp)}")
        print(f"  proj / oracle    {pct(gp / go)}")
        for s in sorted(set(snr)):
            k = snr == s
            print(f"    SNR {s:+5.0f} dB (n {k.sum():3d}): oracle median {np.median(go[k]):.3f} [{go[k].min():.3f}, {go[k].max():.3f}]"
                  f"   proj median {np.median(gp[k]):.3f} [{gp[k].min():.3f}, {gp[k].max():.3f}]"
                  f"   proj/oracle median {np.median(gp[k] / go[k]):.3f} max {np.max(gp[k] / go[k]):.3f}")
        gc = np.clip(gp, args.lo, args.hi)
        print(f"  proj outside [{args.lo}, {args.hi}]: {np.mean((gp < args.lo) | (gp > args.hi)) * 100:.2f} % of files")
        variants = {"raw": y, "const": y * const, "proj": y * gp[:, None], "proj_clamp": y * gc[:, None],
                    "oracle": y * go[:, None]}
        peak_in, peak_clean = np.abs(noisy).max(1), np.abs(clean).max(1)
        print(f"  files whose peak exceeds full scale: noisy input {np.mean(peak_in >= FULL_SCALE) * 100:.2f} %, "
              f"clean {np.mean(peak_clean >= FULL_SCALE) * 100:.2f} %")
        for k, v in variants.items():
            peak = np.abs(v).max(1)
            over = peak > FULL_SCALE
            clipped = np.mean(np.abs(v[over]) > FULL_SCALE) * 100 if over.any() else 0.0
            line = (f"  {k:10s} clipping files {over.mean() * 100:5.2f} % (peak max {peak.max():.3f}, "
                    f"clipped samples in those files {clipped:.4f} %)")
            if pool:
                sc = score_all(pool, clean, v)
                line += f"   PESQ {np.nanmean(sc[:, 0]):.4f}  SNR {sc[:, 1].mean():.2f} dB"
                for s in sorted(set(snr)):
                    line += f" | {s:+.0f}dB: {sc[snr == s, 1].mean():.2f}"
            print(line, flush=True)


# ---------------------------------------------------------------- mix
def parse_combo(label: str) -> dict[str, float]:
    out = {}
    for part in label.split("+"):
        name = part.rstrip("0123456789.")
        out[name] = float(part[len(name):])
    return out


def label_of(weights: dict[str, float]) -> str:
    return "+".join(f"{k}{v:.1f}" for k, v in weights.items())


def grid(names: list[str], anchor: str) -> list[dict[str, float]]:
    """Singles, and every subset containing `anchor` with weights in steps of 0.1 (anchor >= 0.4, others >= 0.1)."""
    combos = [{n: 1.0} for n in names]
    others = [n for n in names if n != anchor]
    for k in range(1, len(others) + 1):
        for sub in itertools.combinations(others, k):
            for w in itertools.product(range(1, 7), repeat=k):
                a = 10 - sum(w)
                if a >= 4:
                    combos.append({anchor: a / 10, **{n: v / 10 for n, v in zip(sub, w)}})
    return combos


def combine(outs: dict[str, np.ndarray], weights: dict[str, float], method: str, phase: str) -> np.ndarray:
    total = sum(weights.values())
    if method == "wave" or len(weights) == 1:
        return sum(outs[k] * (w / total) for k, w in weights.items())
    import torch
    win = torch.hann_window(400, dtype=torch.float64)
    spec = {k: torch.stft(torch.from_numpy(outs[k]), 400, 100, window=win, return_complex=True) for k in weights}
    mag = sum(spec[k].abs() * (w / total) for k, w in weights.items())
    y = torch.istft(torch.polar(mag, torch.angle(spec[phase])), 400, 100, window=win,
                    length=outs[phase].shape[-1])
    return y.numpy()


def cmd_mix(args) -> None:
    names = members_available()
    combos = grid(names, args.anchor) if args.grid else [parse_combo(c) for c in args.combos]
    tag = f"{args.method}|{args.gain or 'nogain'}" + (f"|phase={args.phase}" if args.method == "mag" else "")
    store = CACHE / "scores"
    store.mkdir(parents=True, exist_ok=True)
    print(f"{len(combos)} combinations, {tag}")
    with ProcessPoolExecutor(args.workers) as pool:
        for split in args.splits:
            noisy, clean = load_split(split)
            n = len(member(split, names[0]))
            noisy, clean = noisy[:n], clean[:n]
            outs = {m: apply_gain(args.gain, m, noisy, member(split, m)) for m in names}
            path = store / f"{split}.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            for w in combos:
                key = f"{label_of(w)}|{tag}"
                if key in data and not args.force:
                    continue
                sc = score_all(pool, clean, combine(outs, w, args.method, args.phase))
                data[key] = {"pesq": sc[:, 0].round(4).tolist(), "snr": sc[:, 1].round(3).tolist()}
                print(f"[{split}] {key}: PESQ {np.nanmean(sc[:, 0]):.4f} SNR {sc[:, 1].mean():.2f}", flush=True)
                path.write_text(json.dumps(data))


# ---------------------------------------------------------------- report
def boot_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_report(args) -> None:
    SPLITS = tuple(args.splits)                 # noqa: N806 - the splits scored (all four on the VM)
    data = {s: json.loads((CACHE / "scores" / f"{s}.json").read_text()) for s in SPLITS}
    tags = sorted({k.split("|", 1)[1] for k in data["validation"]})
    for tag in tags:
        keys = [k for k in data["validation"] if k.split("|", 1)[1] == tag and all(k in data[s] for s in SPLITS)]
        if not keys:
            continue
        mean = {k: {s: (float(np.nanmean(data[s][k]["pesq"])), float(np.mean(data[s][k]["snr"]))) for s in SPLITS}
                for k in keys}
        sel = {k: np.mean([mean[k][s][0] for s in SELECT_ON if s in SPLITS]) for k in keys}
        groups: dict[frozenset, list[str]] = {}
        for k in keys:
            groups.setdefault(frozenset(parse_combo(k.split("|")[0])), []).append(k)
        best = {g: max(ks, key=sel.get) for g, ks in groups.items()}
        print(f"\n## {tag}\n")
        print("Best weights of every member set (chosen on validation + newnoise). PESQ / SNR dB per split.\n")
        print("| members | weights | " + " | ".join(SPLITS) + " | chosen-on mean |")
        print("|---|---|" + "---|" * (len(SPLITS) + 1))
        order = sorted(best, key=lambda g: (len(g), -sel[best[g]]))
        for g in order:
            k = best[g]
            print(f"| {'+'.join(sorted(g))} | {k.split('|')[0]} | "
                  + " | ".join(f"{mean[k][s][0]:.3f} / {mean[k][s][1]:.2f}" for s in SPLITS) + f" | {sel[k]:.4f} |")
        if not args.anchor or frozenset([args.anchor]) not in best:
            continue
        print(f"\nPer-file PESQ difference with 95 % bootstrap interval [better / worse files]: against {args.anchor} alone, "
              "and against the best set with one member fewer (its subset with the highest chosen-on mean).\n")
        print("| members | against | " + " | ".join(SPLITS) + " |")
        print("|---|---|" + "---|" * len(SPLITS))
        for g in order:
            if len(g) < 2:
                continue
            refs = [("alone", frozenset([args.anchor]))]
            subs = [h for h in best if len(h) == len(g) - 1 and h < g and args.anchor in h]
            if subs and len(g) > 2:
                h = max(subs, key=lambda h: sel[best[h]])
                refs.append(("+".join(sorted(h)), h))
            for name, h in refs:
                cells = []
                for s in SPLITS:
                    d = np.array(data[s][best[g]]["pesq"]) - np.array(data[s][best[h]]["pesq"])
                    d = d[np.isfinite(d)]
                    lo, hi = boot_ci(d)
                    cells.append(f"{d.mean():+.3f} ({lo:+.3f}, {hi:+.3f}) [{int((d > 0).sum())}/{int((d < 0).sum())}]")
                print(f"| {'+'.join(sorted(g))} | {name} | " + " | ".join(cells) + " |")
        if args.top:
            print(f"\nTop {args.top} of all combinations by the chosen-on mean:\n")
            for k in sorted(keys, key=sel.get, reverse=True)[:args.top]:
                print(f"  {k.split('|')[0]:28s} " + "  ".join(f"{s} {mean[k][s][0]:.3f}/{mean[k][s][1]:.2f}" for s in SPLITS))


# ---------------------------------------------------------------- PCS
# Perceptual Contrast Stretching (Chao et al., Interspeech 2022). Band weights for a 400-point
# STFT (201 bins): the PCS400 table of https://github.com/RoyChao19477/SEMamba (models/pcs400.py,
# commit 750ff5d, MIT). The source computes  |X| -> expm1(w * log1p(|X|))  on a Hamming-window
# STFT of the waveform as it is and then sets the peak of the result to 1.
PCS400 = np.ones(201)
PCS400[3:5] = 1.070175439
PCS400[5:8] = 1.182456140
PCS400[8:10] = 1.287719298
PCS400[10:110] = 1.4
PCS400[110:130] = 1.322807018
PCS400[130:160] = 1.238596491
PCS400[160:190] = 1.161403509
PCS400[190:201] = 1.077192982
PCS_PEAK = 1.0          # with norm = "peak": level at which PCS is applied (log1p depends on the level)
PCS_EPS = 1e-8


def pcs_magnitude(mag, strength: float, mode: str):
    """mag [..., 201, frames] (torch). mode 'coef': weights 1 + strength * (w - 1);
    'blend': (1 - strength) * mag + strength * PCS(mag). Both are PCS400 at strength 1."""
    import torch
    w = torch.from_numpy(PCS400).to(mag.dtype)[:, None]
    if mode == "coef":
        return torch.expm1((1 + strength * (w - 1)) * torch.log1p(mag))
    return (1 - strength) * mag + strength * torch.expm1(w * torch.log1p(mag))


def _window(kind: str):
    import torch
    if kind == "hann":
        return torch.hann_window(400, dtype=torch.float64)
    from scipy.signal.windows import hamming
    return torch.from_numpy(hamming(400)).double()          # symmetric, as in the source


def pcs_apply(mag, phase, y0, strength: float, mode: str, norm: str, window):
    """PCS on a magnitude spectrum, one inverse STFT, then the level of y0 (the output without PCS).

    norm 'peak': the spectrum is scaled so that y0 would peak at PCS_PEAK, stretched, scaled back
    (the same stretch for a quiet and a loud copy of a file); 'asis': stretched at its own level,
    as the source does. A silent y0 is returned as it is (no division by a zero peak). The level
    is restored by the least-squares gain towards y0 (no reference needed), not by setting the
    peak to 1.
    """
    import torch
    peak = y0.abs().amax(-1)
    ok = peak > PCS_EPS
    if norm == "peak":
        c = torch.where(ok, PCS_PEAK / peak.clamp_min(PCS_EPS), torch.ones_like(peak))
    else:
        c = torch.ones_like(peak)
    stretched = pcs_magnitude(mag * c[:, None, None], strength, mode) / c[:, None, None]
    y1 = torch.istft(torch.polar(stretched, phase), 400, 100, window=window, length=y0.shape[-1])
    e = (y1 * y1).sum(-1)
    g = torch.where(e > PCS_EPS, (y0 * y1).sum(-1) / e.clamp_min(PCS_EPS), torch.ones_like(e))
    return torch.where(ok[:, None], y1 * g[:, None], y0)


def pcs_configs() -> list[tuple[str, str, str, float]]:
    """(site, norm, mode, strength); strength 1 is the same for both modes."""
    out = []
    for site in ("spec", "wave"):
        for norm in ("peak", "asis"):
            out += [(site, norm, "coef", v) for v in (0.25, 0.5, 0.75, 1.0)]
            out += [(site, norm, "blend", v) for v in (0.25, 0.5, 0.75)]
    return out


def cmd_pcs(args) -> None:
    import torch
    store = CACHE / "scores_pcs"
    store.mkdir(parents=True, exist_ok=True)
    targets = {t.split("=")[0]: parse_combo(t.split("=")[1]) for t in args.targets}
    hann, hamming = _window("hann"), _window("hamming")
    with ProcessPoolExecutor(args.workers) as pool:
        for split in args.splits:
            noisy, clean = load_split(split)
            names = sorted({m for w in targets.values() for m in w})
            n = len(member(split, names[0]))
            noisy, clean = noisy[:n], clean[:n]
            outs = {m: torch.from_numpy(apply_gain(args.gain, m, noisy, member(split, m))) for m in names}
            path = store / f"{split}.json"
            data = json.loads(path.read_text()) if path.exists() else {}

            def save(key, y):
                y = y.numpy()
                sc = score_all(pool, clean, y)
                peak = np.abs(y).max(1)
                data[key] = {"pesq": sc[:, 0].round(4).tolist(), "snr": sc[:, 1].round(3).tolist(),
                             "peak": peak.round(4).tolist()}
                path.write_text(json.dumps(data))
                print(f"[{split}] {key}: PESQ {np.nanmean(sc[:, 0]):.4f} SNR {sc[:, 1].mean():.2f} "
                      f"clip {np.mean(peak > FULL_SCALE) * 100:.2f} %", flush=True)

            for tname, weights in targets.items():
                total = sum(weights.values())
                spec = {m: torch.stft(outs[m], 400, 100, window=hann, return_complex=True) for m in weights}
                mag = sum(spec[m].abs() * (w / total) for m, w in weights.items())
                phase = torch.angle(spec[args.phase])
                y0 = torch.istft(torch.polar(mag, phase), 400, 100, window=hann, length=noisy.shape[-1])
                del spec
                if f"{tname}|none" not in data or args.force:
                    save(f"{tname}|none", y0)
                # the same output seen through the source's Hamming STFT, for PCS as a waveform post-step
                s0 = torch.stft(y0, 400, 100, window=hamming, return_complex=True, pad_mode="constant")
                mag_w, phase_w = s0.abs(), torch.angle(s0)
                del s0
                for site, norm, mode, strength in pcs_configs():
                    key = f"{tname}|{site}|{norm}|{mode}|{strength}"
                    if key in data and not args.force:
                        continue
                    if site == "spec":
                        y = pcs_apply(mag, phase, y0, strength, mode, norm, hann)
                    else:
                        y = pcs_apply(mag_w, phase_w, y0, strength, mode, norm, hamming)
                    save(key, y)


def cmd_pcs_report(args) -> None:
    splits = tuple(args.splits)
    data = {s: json.loads((CACHE / "scores_pcs" / f"{s}.json").read_text()) for s in splits}
    targets = sorted({k.split("|")[0] for k in data[splits[0]]})
    for t in targets:
        base = {s: data[s][f"{t}|none"] for s in splits}
        print(f"\n## {t}\n")
        print("Without PCS: " + ", ".join(
            f"{s} {np.nanmean(base[s]['pesq']):.3f} / {np.mean(base[s]['snr']):.2f} dB "
            f"(clip {np.mean(np.array(base[s]['peak']) > FULL_SCALE) * 100:.2f} %)" for s in splits))
        print("\nPESQ change with 95 % interval / SNR change (dB) / files over full scale (%). "
              "site: spec = on the averaged magnitude (Hann), wave = on the finished waveform (Hamming, as the source); "
              "norm: peak = stretched at peak 1 and scaled back, asis = at the file's own level.\n")
        print("| site | norm | mode | strength | " + " | ".join(splits) + " |")
        print("|---|---|---|---|" + "---|" * len(splits))
        keys = [k for k in data[splits[0]] if k.startswith(t + "|") and not k.endswith("|none")]
        for k in keys:
            cells = []
            for s in splits:
                d = np.array(data[s][k]["pesq"]) - np.array(base[s]["pesq"])
                d = d[np.isfinite(d)]
                lo, hi = boot_ci(d)
                cells.append(f"{d.mean():+.3f} ({lo:+.3f}, {hi:+.3f}) / "
                             f"{np.mean(data[s][k]['snr']) - np.mean(base[s]['snr']):+.2f} / "
                             f"{np.mean(np.array(data[s][k]['peak']) > FULL_SCALE) * 100:.2f}")
            print("| " + " | ".join(k.split("|")[1:]) + " | " + " | ".join(cells) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "align", "gain", "mix", "report", "pcs", "pcs-report"):
        p = sub.add_parser(name)
        p.add_argument("--splits", nargs="+", default=list(SPLITS))
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--force", action="store_true")
        if name == "dump":
            p.add_argument("--members", nargs="+", required=True, help="NAME=weights/file.pt ...")
            p.add_argument("--device", default="auto")
            p.add_argument("--n", type=int, default=0, help="only the first n files (testing)")
        if name == "gain":
            p.add_argument("--member", required=True)
            p.add_argument("--const", type=float)
            p.add_argument("--lo", type=float, default=1.0)
            p.add_argument("--hi", type=float, default=2.0)
            p.add_argument("--score", action="store_true")
        if name == "mix":
            p.add_argument("--grid", action="store_true")
            p.add_argument("--combos", nargs="*", default=[])
            p.add_argument("--anchor", default="L")
            p.add_argument("--gain", default="")
            p.add_argument("--method", choices=["wave", "mag"], default="wave")
            p.add_argument("--phase", default="L")
        if name == "report":
            p.add_argument("--anchor", default="L")
            p.add_argument("--top", type=int, default=10)
        if name == "pcs":
            p.add_argument("--targets", nargs="+", required=True, help="NAME=L0.5+F0.2+M0.3 ...")
            p.add_argument("--gain", default="")
            p.add_argument("--phase", default="L")
    args = ap.parse_args(argv)
    {"dump": cmd_dump, "align": cmd_align, "gain": cmd_gain, "mix": cmd_mix, "report": cmd_report,
     "pcs": cmd_pcs, "pcs-report": cmd_pcs_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
