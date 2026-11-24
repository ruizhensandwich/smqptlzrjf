import numpy as np
import pytest
import torch

from se.config import load_config
from se.data import EXTERNAL_SPEECH, RemixedSpeech, load_external_speech
from tests.test_external_noise import Base
from tools.make_external_speech import file_stats, judge_speakers, speech_windows


def recording(floor_db: float, seconds: int = 4, seed: int = 0) -> np.ndarray:
    """Half speech-level noise bursts at -20 dB, half room noise at floor_db."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(seconds * 16000) * 10 ** (floor_db / 20)
    for a in range(0, x.size, 16000):
        x[a:a + 8000] = rng.standard_normal(8000) * 0.1
    return x


def test_a_noisy_room_gives_a_small_range_and_its_speaker_is_left_out():
    quiet, noisy = file_stats(recording(-80)), file_stats(recording(-45))
    assert abs(quiet["floor_db"] + 80) < 1 and abs(quiet["range_db"] - 60) < 1
    assert abs(noisy["range_db"] - 25) < 1
    assert file_stats(np.zeros(100)) is None                             # too short to judge
    stats = [{**quiet, "speaker": "101", "path": "a"}, {**quiet, "speaker": "101", "path": "b"},
             {**noisy, "speaker": "102", "path": "c"},
             {**quiet, "speaker": "103", "path": "d", "peak": 1.0}]
    rows = {r["speaker"]: r for r in judge_speakers(stats, 45.0)}
    assert [rows[s]["used"] for s in ("101", "102", "103")] == [1, 0, 1]
    assert rows["101"]["files"] == 2 and rows["103"]["clipped_files"] == 1


def test_windows_hold_speech_and_do_not_overlap():
    x = np.zeros(4 * 1000)
    x[1000:2000], x[3000:4000] = 0.1, 0.2                                # windows 1 and 3 speak
    assert speech_windows(x, 1000) == [1000, 3000]
    assert speech_windows(x, 1000, most=1) == [3000]
    assert speech_windows(x[:500], 1000) == [] and speech_windows(np.zeros(3000), 1000) == []


def test_external_speech_replaces_the_speech_at_the_item_level():
    base = Base()
    voice = 0.01 * torch.sin(torch.arange(16000) * 0.2).repeat(3, 1)     # unlike any train speech
    torch.manual_seed(1)
    items = RemixedSpeech(base, 0, 1.0, (5.0, 5.0), external_speech=voice, speech_prob=1.0)
    for i in range(4):
        noisy, clean = items[i]
        noise = noisy - clean
        spectrum = torch.fft.rfft(clean).abs()
        assert spectrum.max() > 50 * spectrum.median()                   # the tone is the target
        assert abs(float(clean.pow(2).mean() / base.waveforms[i, 1].pow(2).mean()) - 1) < 1e-3
        assert abs(float(10 * torch.log10(clean.pow(2).mean() / noise.pow(2).mean())) - 5.0) < 1e-3


def test_no_external_speech_keeps_the_old_random_draws():
    base = Base()
    torch.manual_seed(3)
    old = [RemixedSpeech(base, 8000, 0.5)[i] for i in range(4)]
    torch.manual_seed(3)
    new = [RemixedSpeech(base, 8000, 0.5, external_speech=torch.ones(2, 16000), speech_prob=0.0)[i]
           for i in range(4)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(old, new))


def test_pool_loading_and_config_checks(tmp_path):
    path = tmp_path / EXTERNAL_SPEECH["zeroth"]
    with pytest.raises(FileNotFoundError):
        load_external_speech("zeroth", tmp_path)
    path.parent.mkdir(parents=True)
    np.save(path, np.full((2, 16000), 16384, dtype=np.int16))
    pool = load_external_speech("zeroth", tmp_path)
    assert pool.shape == (2, 16000) and pool.dtype == torch.float32 and float(pool[0, 0]) == 0.5
    with pytest.raises(ValueError):
        load_config(None, ['data.external_speech="nope"'])
    with pytest.raises(ValueError):                                      # needs remixing
        load_config(None, ['data.external_speech="zeroth"', "data.external_speech_prob=0.5"])
    cfg = load_config(None, ['data.external_speech="zeroth"', "data.external_speech_prob=0.5",
                             "data.remix_prob=0.5"])
    assert cfg["data"]["external_speech_prob"] == 0.5
