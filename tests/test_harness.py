"""Submission harness: fingerprint, secret scan, README rendering."""
import shutil

from tools import release
from tools.common import fingerprint
from tools.secret_scan import scan


def test_fingerprint_follows_weights_and_code(tmp_path):
    (tmp_path / "se").mkdir()
    (tmp_path / "weights").mkdir()
    (tmp_path / "denoise.py").write_bytes(b"x = 1\n")
    (tmp_path / "weights" / "submission.pt").write_bytes(b"\x00\x01")
    a = fingerprint(tmp_path)
    (tmp_path / "weights" / "submission.pt").write_bytes(b"\x00\x02")
    b = fingerprint(tmp_path)
    (tmp_path / "denoise.py").write_bytes(b"x = 1\r\n")   # line endings ignored
    assert a != b and fingerprint(tmp_path) == b


def test_secret_scan_blocks_keys_data_and_reports(tmp_path):
    files = {
        "ok.py": "print('hello')\n",
        "key.json": '{"type": "service_account", "project_id": "x"}',
        "notes.md": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n",
        "data/train/samples.csv": "a,b\n",
        "input/x.wav": "RIFF",
        "docs/ai_usage/round1.json": "{}",
        "results/listening/x/index.html": "<audio>",
        ".env": "TOKEN=1",
        "conf.py": "KEY = 'AIza" + "A" * 35 + "'\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    problems = scan(list(files), root=tmp_path)
    flagged = {p.split(":")[0] for p in problems}
    assert flagged == set(files) - {"ok.py"}


def test_reproduce_block_has_what_the_notice_asks(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    shutil.copy(release.README, readme)
    monkeypatch.setattr(release, "README", readme)
    data = {"team": "7조", "repository": "https://example.invalid/repo", "rounds": [{
        "round": 1, "tag": "submission-1", "commit": None, "submitted_at": None, "leaderboard": None,
        "validation": {"pesq": 1.69, "snr_db": 13.0}, "train": {"pesq": 1.8, "snr_db": 12.0},
        "weights": {"model": "mask_cnn", "model_args": {"n_fft": 512}, "parameters": 1000,
                    "loss": "log_mag_l1", "seed": 42, "epochs_completed": 109, "best_epoch": 89,
                    "stop_reason": "early_stopping", "best_val_pesq_training": 1.6922,
                    "config_path": "configs/mask_cnn_baseline.json", "overrides": [],
                    "train_settings": {"lr": 0.001}, "weights_mb": 0.43, "torch": "2.15.0",
                    "device": "cpu", "source": "x"}}]}
    release.write_readme(data)
    text = readme.read_text(encoding="utf-8")
    for need in ("seed", "42", "학습 epoch", "89 epoch 가중치 선택", "선택 가중치의 검증 점수",
                 "`submission-1`", "weights/submission.pt", "python train.py --config "
                 "configs/mask_cnn_baseline.json"):
        assert need in text
    assert "시작 가중치" not in text                  # rounds recorded before the field: no row

    w = data["rounds"][0]["weights"]
    w.update(init_weights="", init_source=None)
    release.write_readme(data)
    assert "무작위 초기값" in readme.read_text(encoding="utf-8")
    w.update(init_weights="weights/pretrained/x.pt", init_source="authors' DNS checkpoint")
    release.write_readme(data)
    text = readme.read_text(encoding="utf-8")
    assert "`weights/pretrained/x.pt`에서 이어 학습" in text and "authors' DNS checkpoint" in text
