"""Listening tool: figure, WAVs at one playback gain, and the HTML page."""
import numpy as np
import soundfile as sf

from conftest import write_wav
from tools import spectrogram


def test_listening_page_for_any_wav(tmp_path, tiny_weights):
    noisy = write_wav(tmp_path / "rec.wav", n=24000, seed=5)
    clean = write_wav(tmp_path / "ref.wav", n=24000, seed=6)
    out = tmp_path / "out"
    code = spectrogram.main(["--noisy", str(noisy), "--clean", str(clean), "--models",
                             str(tiny_weights), str(tiny_weights), "--labels", "a", "b",
                             "--out", str(out), "--device", "cpu"])
    assert code == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["index.html", "rec.png", "rec__Clean.wav", "rec__Noisy.wav",
                     "rec__a.wav", "rec__b.wav"]
    peaks = [np.max(np.abs(sf.read(str(out / n))[0])) for n in names if n.endswith(".wav")]
    assert max(peaks) > 0.95                       # one common gain: the loudest clip near full scale
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "rec__a.wav" in page and "PESQ-WB" in page
