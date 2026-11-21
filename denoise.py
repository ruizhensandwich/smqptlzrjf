"""Submission entry point: enhance every WAV in input/ and write it to output/.

    python denoise.py                                   # ./input -> ./output (next to this file)
    python denoise.py --input IN --output OUT
    python denoise.py --weights weights/other.pt --device cuda

Each output goes directly into output/ with the input's file name and keeps
the input's sample rate, channel count, sample count and sample format
(16 kHz mono PCM_16 for the challenge). The model and its settings come from
the weight file alone (weights/submission.pt), so nothing is downloaded. If a
file cannot be processed it is copied unchanged, so every input still has
exactly one output. Only input files ending in .wav are read; nothing else is
written to output/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from se.checkpoint import load_model  # noqa: E402
from se.inference import enhance_channels, pick_device  # noqa: E402

DEFAULT_WEIGHTS = HERE / "weights" / "submission.pt"


def output_path(src: Path, src_root: Path, out_root: Path, taken: set[str]) -> Path:
    """output/<file name>; a name already used (sub-folders) keeps its relative path."""
    if src.name.lower() not in taken:
        taken.add(src.name.lower())
        return out_root / src.name
    return out_root / src.relative_to(src_root)


def denoise_file(model, device, src: Path, dst: Path) -> None:
    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)    # (samples, channels)
    y = enhance_channels(model, x, sr, device)
    if info.channels == 1:
        y = y[:, 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype=info.subtype, format=info.format)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=HERE / "input")
    ap.add_argument("--output", type=Path, default=HERE / "output")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args(argv)

    src_root = args.input.resolve()
    if not src_root.is_dir():
        print(f"Input folder not found: {src_root}", file=sys.stderr)
        return 1
    files = sorted(p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        print(f"No WAV files in {src_root}", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    model = None
    try:
        model, ckpt = load_model(args.weights, device)
        print(f"model {ckpt['model']} from {args.weights.name} on {device}", flush=True)
    except Exception as e:  # noqa: BLE001 - outputs must exist even then
        print(f"[WARN] cannot load {args.weights}: {type(e).__name__}: {e} -> inputs copied",
              file=sys.stderr)

    taken: set[str] = set()
    copied = failed = 0
    for i, src in enumerate(files, 1):
        dst = output_path(src, src_root, args.output, taken)
        try:
            if model is None:
                raise RuntimeError("no model")
            denoise_file(model, device, src, dst)
        except Exception as e:  # noqa: BLE001
            if model is not None:
                print(f"[WARN] {src.name}: {type(e).__name__}: {e} -> copied unprocessed",
                      file=sys.stderr)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied += 1
            except Exception as e2:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {src.name}: {type(e2).__name__}: {e2}", file=sys.stderr)
        if i % 100 == 0 or i == len(files):
            print(f"{i}/{len(files)} done", flush=True)
    print(f"denoised={len(files) - copied - failed}  unprocessed={copied}  failed={failed}  "
          f"-> {args.output.resolve()}")
    return 1 if failed or model is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
