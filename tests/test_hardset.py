import numpy as np
import soundfile as sf

from tools.make_hardset import babble, reverberate, room_response


def test_room_response_decays_by_60_db_in_rt60():
    rng = np.random.default_rng(0)
    for rt60 in (0.3, 0.8):
        h = room_response(rt60, rng)
        assert h[0] == 1.0 and h.size == int(rt60 * 16000)
        tail = h[1:] ** 2
        first, last = tail[: tail.size // 10].mean(), tail[-tail.size // 10:].mean()
        assert 40 < 10 * np.log10(first / last) < 70                       # about 54 dB over 90 %


def test_reverb_keeps_length_and_timing():
    rng = np.random.default_rng(1)
    clean = np.zeros(16000)
    clean[1000] = 1.0
    wet = reverberate(clean, room_response(0.5, rng))
    assert wet.size == clean.size and np.argmax(np.abs(wet)) == 1000      # direct sound not shifted
    assert np.sum(wet[1100:] ** 2) > 0.01                                 # a tail after it


def test_babble_sums_distinct_talkers_at_equal_level(tmp_path):
    rng = np.random.default_rng(2)
    files = []
    for k in range(5):
        p = tmp_path / f"{100 + k}_001_{k:04d}.flac"
        sf.write(str(p), (k + 1) * 0.05 * rng.standard_normal(16000 * 4), 16000)
        files.append(p)
    out, names = babble(files, 16000 * 3, 3, rng)
    assert out.shape == (48000,) and len(set(names.split("+"))) == 3
    assert abs(np.mean(out ** 2) - 3.0) < 0.3                             # three unit-power talkers
