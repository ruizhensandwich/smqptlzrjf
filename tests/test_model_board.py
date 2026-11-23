"""Model board: metrics from per-file scores, best-value marking, strengths/weaknesses."""
from tools import model_board as mb


def per_file(gains, snrs=(0, 15), noisy=1.5):
    rows = []
    for i, g in enumerate(gains):
        base = {"segment_id": f"s{i}", "snr_db": 10.0, "noise_category": f"cat{i % 2}",
                "noise_code": "x", "noise_difficulty": "hard" if i % 2 else "easy",
                "target_snr_db": str(snrs[i % len(snrs)])}
        rows.append({**base, "method": "noisy_input", "pesq": noisy})
        rows.append({**base, "method": "m", "pesq": noisy + g})
    return rows


def test_split_metrics():
    m = mb.split_metrics(per_file([0.5, -0.1, 0.5, -0.1, 0.5, -0.1]), "m")
    assert abs(m["d_pesq"] - 0.2) < 1e-9
    assert abs(m["worse_pct"] - 50) < 1e-9
    assert m["best_category"] == "cat0" and m["worst_category"] == "cat1"


def test_board_marks_best_and_judges(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "BOARD_CSV", tmp_path / "b.csv")
    monkeypatch.setattr(mb, "BOARD_MD", tmp_path / "b.md")
    good = mb.make_row({"validation": per_file([0.6] * 6), "train": per_file([0.7] * 6)}, "m",
                       {"date": "2026-11-22", "run_id": "r1", "tag": "big", "model": "a",
                        "params": 900000, "weights_sha": "aaa", "rtf": 0.01, "infer_device": "cpu"})
    weak = mb.make_row({"validation": per_file([0.1, -0.2] * 3), "train": per_file([0.4] * 6)}, "m",
                       {"date": "2026-11-21", "run_id": "r0", "tag": "small", "model": "a",
                        "params": 100000, "weights_sha": "bbb", "rtf": 0.001, "infer_device": "cpu"})
    mb.save_row(good)
    mb.save_row(weak)
    mb.save_row(good)                       # same weights again: replaced, not duplicated
    text = mb.show()
    assert len(mb.load_rows()) == 2
    assert "**2.100**" in text              # best validation PESQ in bold
    assert "검증 PESQ 최고" in text and "가장 가벼움 (100,000" in text
    assert "과적합 경향" in text and "입력보다 나빠진 파일 50.0%" in text
    assert text.index("small") < text.index("big")      # oldest first


def test_newnoise_scores_join_the_row_and_survive_a_validation_rescore(tmp_path, monkeypatch):
    from tools import model_board as mb
    monkeypatch.setattr(mb, "BOARD_CSV", tmp_path / "board.csv")
    monkeypatch.setattr(mb, "BOARD_MD", tmp_path / "board.md")

    def per_file(noisy, out):
        base = {"noise_category": "wind", "noise_difficulty": "easy", "target_snr_db": "5.0",
                "snr_db": "10"}
        return [{**base, "segment_id": "a", "method": "noisy_input", "pesq": str(noisy)},
                {**base, "segment_id": "a", "method": "m", "pesq": str(out)}]

    info = {"weights_sha": "abc", "tag": "t", "date": "2026-11-23", "model": "mpsenet"}
    assert not mb.set_newnoise("abc", per_file(1.2, 2.0), "m")          # no row yet
    mb.save_row(mb.make_row({"validation": per_file(1.4, 3.0)}, "m", info))
    assert mb.set_newnoise("abc", per_file(1.2, 2.0), "m")
    mb.save_row(mb.make_row({"validation": per_file(1.4, 3.1)}, "m", info))
    (row,) = mb.load_rows()
    assert float(row["val_pesq"]) == 3.1 and float(row["new_pesq"]) == 2.0
    assert abs(float(row["new_d_pesq"]) - 0.8) < 1e-9 and "새 잡음 PESQ" in mb.show()
    both = mb.make_row({"validation": per_file(1.4, 3.0), "newnoise": per_file(1.0, 1.5)}, "m", info)
    assert both["new_pesq"] == 1.5
