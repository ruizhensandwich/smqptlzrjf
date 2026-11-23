import numpy as np

from tools.make_newnoise import loudest_window, mix, plan, to_16k_mono


def test_mix_has_the_requested_snr_and_never_clips():
    rng = np.random.default_rng(0)
    clean, noise = 0.1 * rng.standard_normal(48000), rng.standard_normal(48000)
    noisy, c, gain_db = mix(clean, noise, 5.0)
    snr = 10 * np.log10(np.mean(c ** 2) / np.mean((noisy - c) ** 2))
    assert abs(snr - 5.0) < 1e-9 and gain_db == 0 and np.array_equal(c, clean)
    noisy, c, gain_db = mix(8 * clean, noise, 0.0)                  # would exceed full scale
    snr = 10 * np.log10(np.mean(c ** 2) / np.mean((noisy - c) ** 2))
    assert abs(snr) < 1e-9 and gain_db < 0 and np.max(np.abs(noisy)) <= 0.99 + 1e-12


def test_loudest_window_finds_the_event_and_resampling_gives_16k():
    x = np.zeros(80000)
    x[40000:44000] = 1.0
    w, start = loudest_window(x, 48000)
    assert w.size == 48000 and start <= 40000 and start + 48000 >= 44000 and w.sum() == 4000
    assert loudest_window(np.ones(1000), 48000)[0].size == 48000    # short clips are tiled
    assert to_16k_mono(np.zeros((44100, 2)), 44100).shape == (16000,)


def test_plan_spreads_classes_uses_no_clip_twice_and_is_reproducible():
    esc = [{"filename": f"5-{c}-{k}.wav", "category": f"c{c}"} for c in range(50) for k in range(8)]
    val = [{"segment_id": f"s{i}"} for i in range(164)]
    a, b = plan(val, esc, 0), plan(val, esc, 0)
    names = [r["filename"] for r in a]
    assert names == [r["filename"] for r in b] and len(set(names)) == 164
    counts = np.unique([r["category"] for r in a], return_counts=True)[1]
    assert len(counts) == 50 and counts.min() >= 3 and counts.max() <= 4
    assert names != [r["filename"] for r in plan(val, esc, 1)]
