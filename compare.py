"""Compare two evaluation runs file by file.

    python compare.py <runA> <runB> [--split validation]
    python compare.py results/submissions/round1 results/runs/<id>/validation

A and B are run ids under results/runs/ (then --split picks the split folder)
or folders holding per_file.csv. Each side uses its non-noisy method unless
--method-a/--method-b is given. Prints the mean change, win/loss counts, the
change by SNR, difficulty and noise category, and the most damaged/improved files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def load(run: str, split: str) -> pd.DataFrame:
    p = Path(run)
    if not (p / "per_file.csv").exists():
        p = HERE / "results" / "runs" / run / split
    return pd.read_csv(p / "per_file.csv", encoding="utf-8-sig")


def pick(df: pd.DataFrame, method: str | None):
    if method is None:
        method = [m for m in df.method.unique() if m != "noisy_input"][-1]
    return method, df[df.method == method].set_index("segment_id")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--method-a")
    ap.add_argument("--method-b")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    ma, a = pick(load(args.a, args.split), args.method_a)
    mb, b = pick(load(args.b, args.split), args.method_b)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    d = b[["noise_category", "noise_code", "noise_difficulty", "target_snr_db"]].copy()
    d["pesq_a"], d["pesq_b"] = a.pesq, b.pesq
    d["d_pesq"] = b.pesq - a.pesq
    d["d_snr"] = b.snr_db - a.snr_db

    pd.set_option("display.width", 200)
    print(f"A = {ma} ({args.a})\nB = {mb} ({args.b})\nfiles: {len(d)}")
    print(f"PESQ  A {a.pesq.mean():.4f}  B {b.pesq.mean():.4f}  diff {d.d_pesq.mean():+.4f}")
    print(f"SNR   A {a.snr_db.mean():.3f}  B {b.snr_db.mean():.3f}  diff {d.d_snr.mean():+.3f} dB")
    print(f"B better on {(d.d_pesq > 0.01).sum()} files, worse on {(d.d_pesq < -0.01).sum()}, "
          f"within ±0.01 on {(d.d_pesq.abs() <= 0.01).sum()}")
    for g in ("target_snr_db", "noise_difficulty", "noise_category"):
        t = d.groupby(g).agg(n=("d_pesq", "size"), pesq_a=("pesq_a", "mean"),
                             pesq_b=("pesq_b", "mean"), d_pesq=("d_pesq", "mean"))
        print(f"\nby {g}\n" + t.sort_values("d_pesq").round(3).to_string())
    cols = ["noise_category", "noise_code", "target_snr_db", "pesq_a", "pesq_b", "d_pesq"]
    print("\nmost damaged by B\n" + d.nsmallest(args.top, "d_pesq")[cols].round(3).to_string())
    print("\nmost improved by B\n" + d.nlargest(args.top, "d_pesq")[cols].round(3).to_string())


if __name__ == "__main__":
    main()
