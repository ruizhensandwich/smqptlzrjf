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


# --------------------------------------------------------------------------
# dynamic remixing (training only)
# A train pair's noise is noisy - clean. That holds only if the noisy file is the clean file
# plus noise, sample for sample; tools/check_mixing.py reports these numbers for every file
# (plus the cross-correlation lag) and results/mixing_check/summary.json has their distribution.
REMIX_LIMITS = {"max_gain_error": 0.1, "max_abs_corr": 0.1, "max_snr_error_db": 1.0,
                "max_quiet_frac": 0.5, "max_clip_frac": 0.001}
_FRAME = 320                       # 20 ms
_QUIET_RMS = 10 ** (-70 / 20)      # a noise frame below -70 dBFS counts as silent


def mixing_stats(noisy: np.ndarray, clean: np.ndarray) -> dict:
    """How well  noisy = clean + noise  holds for one pair (see REMIX_LIMITS)."""
    noisy, clean = noisy.astype(np.float64), clean.astype(np.float64)
    noise = noisy - clean
    cc, nn = float(clean @ clean), float(noise @ noise)
    a, b = noise - noise.mean(), clean - clean.mean()
    den = float(np.sqrt((a @ a) * (b @ b)))
    frames = noise[: noise.size // _FRAME * _FRAME].reshape(-1, _FRAME)
    return {"gain": float(noisy @ clean) / cc if cc > 0 else float("nan"),
            "corr": float(a @ b) / den if den > 0 else float("nan"),
            "snr_db": float(10 * np.log10(cc / nn)) if cc > 0 and nn > 0 else float("nan"),
            "noise_rms": float(np.sqrt(nn / noise.size)),
            "quiet_frac": float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) < _QUIET_RMS)),
            "clip_frac": float(np.mean(np.abs(noisy) >= 0.999))}


def remix_problems(stats: dict, target_snr_db: float, limits: dict = REMIX_LIMITS) -> list[str]:
    """Reasons not to use this pair's noise as remixing material (empty = usable)."""
    if not all(np.isfinite(stats[k]) for k in ("gain", "corr", "snr_db")):
        return ["not_finite"]
    checks = [("gain", abs(stats["gain"] - 1) > limits["max_gain_error"]),
              ("corr", abs(stats["corr"]) > limits["max_abs_corr"]),
              ("snr", abs(stats["snr_db"] - target_snr_db) > limits["max_snr_error_db"]),
              ("quiet", stats["quiet_frac"] > limits["max_quiet_frac"]),
              ("clip", stats["clip_frac"] > limits["max_clip_frac"])]
    return [name for name, bad in checks if bad]


class RemixedSpeech(Dataset):
    """Train pairs as random segments, a share of them remixed in memory.

    Each item is pair i either as recorded (same random segment of noisy and
    clean) or, with probability remix_prob, its clean speech plus the noise of
    another train pair (noisy - clean of a random usable file, its own random
    segment) at a random SNR. SNR is defined like the dataset's: power of the
    whole clean file over power of the whole noise file. Nothing is written to
    disk; the draws come from torch's global generator, so train.py's seed
    reproduces them. Only train data is used: the speakers and noise recordings
    are the same, only their combinations and levels are new.
    """

    def __init__(self, base: PairedSpeech, segment_samples: int = 0, remix_prob: float = 0.0,
                 snr_db=(-5.0, 20.0), gain_db=(0.0, 0.0)):
        n = base.waveforms.shape[-1]
        if not 0 <= segment_samples <= n or not 0 <= remix_prob <= 1:
            raise ValueError(f"need 0 <= segment_samples <= {n} and 0 <= remix_prob <= 1")
        if snr_db[0] > snr_db[1] or gain_db[0] > gain_db[1]:
            raise ValueError("snr_db and gain_db are [low, high]")
        self.base, self.segment = base, segment_samples or n
        self.remix_prob, self.snr_db, self.gain_db = remix_prob, tuple(snr_db), tuple(gain_db)
        self.material, self.excluded = [], {}
        if remix_prob > 0:
            w = base.waveforms.numpy()
            for i, r in enumerate(base.manifest):
                why = remix_problems(mixing_stats(w[i, 0], w[i, 1]), float(r["target_snr_db"]))
                if why:
                    self.excluded[r["segment_id"]] = why
                else:
                    self.material.append(i)
            if not self.material:
                raise ValueError("no train pair is usable as remixing material")
        clean = base.waveforms[:, 1].double()
        noise = base.waveforms[:, 0].double() - clean
        self.clean_power = clean.pow(2).mean(-1)
        self.noise_power = noise.pow(2).mean(-1)

    def __len__(self):
        return len(self.base)

    def _start(self) -> int:
        room = self.base.waveforms.shape[-1] - self.segment
        return int(torch.randint(0, room + 1, (1,))) if room > 0 else 0

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand(()))

    def __getitem__(self, i):
        noisy, clean = self.base.waveforms[i]
        a = self._start()
        clean_seg = clean[a:a + self.segment]
        if self.remix_prob > 0 and float(torch.rand(())) < self.remix_prob \
                and float(self.clean_power[i]) > 0:
            j = self.material[int(torch.randint(0, len(self.material), (1,)))]
            b = self._start()
            if self.segment == noisy.shape[-1]:       # whole file: shift the noise in time instead
                b = int(torch.randint(0, self.segment, (1,)))
                noise = torch.roll(self.base.waveforms[j, 0] - self.base.waveforms[j, 1], b)
            else:
                noise = self.base.waveforms[j, 0, b:b + self.segment] \
                    - self.base.waveforms[j, 1, b:b + self.segment]
            snr = self._uniform(*self.snr_db)
            scale = torch.sqrt(self.clean_power[i] / (self.noise_power[j] * 10 ** (snr / 10)))
            noisy_seg = clean_seg + noise * scale.float()
        else:
            noisy_seg = noisy[a:a + self.segment]
        gain = 10 ** (self._uniform(*self.gain_db) / 20)
        peak = float(noisy_seg.abs().max()) * gain
        if peak > 0.99:                               # keep the mixture inside full scale
            gain *= 0.99 / peak
        return noisy_seg * gain, clean_seg * gain
