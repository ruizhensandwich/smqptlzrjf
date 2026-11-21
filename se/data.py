"""Paired noisy/clean data (train, validation) with the notebook's waveform cache.

Data location: the environment variable DATA_ROOT if set, otherwise
<repo>/data. That folder holds train/ and validation/ exactly as distributed
(each with samples.csv, clean/, noisy/). No absolute path is needed anywhere.

The first load reads every WAV, checks rate and length, and stores the split
as one float32 array [pairs, 2 (noisy, clean), samples] in
<data>/.waveform_cache/; later loads read that file (rebuilt when samples.csv
changes).
"""
from __future__ import annotations

import csv
import os
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from se import SAMPLE_RATE

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")
MANIFEST_COLUMNS = {"segment_id", "clean_path", "noisy_path", "sample_rate_hz", "num_samples",
                    "noise_category", "noise_collection", "noise_code", "noise_difficulty",
                    "target_snr_db", "split"}


def data_root() -> Path:
    root = Path(os.environ["DATA_ROOT"]).expanduser() if os.environ.get("DATA_ROOT") \
        else REPO / "data"
    if not all((root / s / "samples.csv").is_file() for s in SPLITS):
        raise FileNotFoundError(
            f"{root} has no train/samples.csv and validation/samples.csv. Copy the distributed "
            "train/ and validation/ folders into data/ or set DATA_ROOT to their parent folder.")
    return root


def read_manifest(split: str, root: Path | None = None) -> list[dict]:
    root = root or data_root()
    text = (root / split / "samples.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(StringIO(text)))
    if not rows or not MANIFEST_COLUMNS <= set(rows[0]):
        raise ValueError(f"{split}/samples.csv: missing columns {MANIFEST_COLUMNS - set(rows[0] if rows else [])}")
    ids = [r["segment_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(r["split"] != split for r in rows):
        raise ValueError(f"{split}/samples.csv: duplicated ids or wrong split column")
    return rows


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE or x.ndim != 1:
        raise ValueError(f"16 kHz mono WAV required: {path}")
    return x


def load_waveforms(split: str, root: Path | None = None, rebuild: bool = False) -> np.ndarray:
    root = root or data_root()
    rows = read_manifest(split, root)
    n = int(rows[0]["num_samples"])
    if n <= 0 or any(int(r["num_samples"]) != n for r in rows):
        raise ValueError(f"{split}: all files must have the same positive num_samples")
    key = sha256((root / split / "samples.csv").read_bytes()).hexdigest()[:20]
    cache = root / ".waveform_cache" / f"{split}_v1_{SAMPLE_RATE}_{key}.npy"
    shape = (len(rows), 2, n)
    if cache.is_file() and not rebuild:
        try:
            w = np.load(cache, allow_pickle=False)
            if w.shape == shape and w.dtype == np.float32:
                return w
        except (OSError, ValueError, EOFError):
            pass
    w = np.empty(shape, dtype=np.float32)
    for i, r in enumerate(rows):
        noisy, clean = read_wav(root / r["noisy_path"]), read_wav(root / r["clean_path"])
        if noisy.size != n or clean.size != n:
            raise ValueError(f"length mismatch: {r['segment_id']}")
        w[i, 0], w[i, 1] = noisy, clean
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as tmp:   # never leave a half-written cache
        t = Path(tmp) / cache.name
        np.save(t, w, allow_pickle=False)
        t.replace(cache)
    return w


class PairedSpeech(Dataset):
    """(noisy, clean) float32 tensors of one split, all in memory."""

    def __init__(self, split: str, root: Path | None = None, rebuild: bool = False):
        self.root = root or data_root()
        self.split = split
        self.manifest = read_manifest(split, self.root)
        self.waveforms = torch.from_numpy(load_waveforms(split, self.root, rebuild))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        noisy, clean = self.waveforms[i]
        return noisy, clean
