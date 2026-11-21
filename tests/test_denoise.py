"""The submission contract: every input WAV -> one output directly in output/, same format."""
import numpy as np
import soundfile as sf

import denoise
from conftest import write_wav
from se.inference import as_written


def run(tmp_path, weights):
    inp, out = tmp_path / "input", tmp_path / "output"
    write_wav(inp / "sample_001.wav", seed=1)
    write_wav(inp / "sample_002.wav", n=12345, seed=2)          # odd length
    write_wav(inp / "stereo.wav", channels=2, seed=3)
    write_wav(inp / "sub" / "nested.wav", seed=4)                # sub-folder -> flat output
    (inp / "broken.wav").write_bytes(b"not a wav")               # -> copied unprocessed
    (inp / "notes.txt").write_text("ignore me", encoding="utf-8")
    code = denoise.main(["--input", str(inp), "--output", str(out), "--weights", str(weights),
                         "--device", "cpu"])
    return code, inp, out


def test_outputs_flat_same_names_same_format(tmp_path, tiny_weights):
    code, inp, out = run(tmp_path, tiny_weights)
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == \
        ["broken.wav", "nested.wav", "sample_001.wav", "sample_002.wav", "stereo.wav"]
    for name, src in [("sample_001.wav", inp / "sample_001.wav"),
                      ("sample_002.wav", inp / "sample_002.wav"),
                      ("stereo.wav", inp / "stereo.wav"), ("nested.wav", inp / "sub" / "nested.wav")]:
        a, b = sf.info(str(src)), sf.info(str(out / name))
        assert (a.samplerate, a.channels, a.frames, a.subtype) == \
            (b.samplerate, b.channels, b.frames, b.subtype)
    assert (out / "broken.wav").read_bytes() == b"not a wav"
    y, _ = sf.read(str(out / "sample_001.wav"))
    x, _ = sf.read(str(inp / "sample_001.wav"))
    assert np.isfinite(y).all() and not np.allclose(x, y)     # really processed


def test_missing_weights_still_gives_every_output(tmp_path):
    code, inp, out = run(tmp_path, tmp_path / "nope.pt")
    assert code == 1
    assert len(list(out.iterdir())) == 5
    assert (out / "sample_001.wav").read_bytes() == (inp / "sample_001.wav").read_bytes()


def test_scoring_reads_back_what_denoise_writes():
    y = np.linspace(-1.2, 1.2, 1000)
    back = as_written(np.clip(y, -1, 1 - 2 ** -15), 16000, "PCM_16")
    assert np.all(np.abs(back) <= 1) and np.allclose(back, np.round(back * 32768) / 32768)
