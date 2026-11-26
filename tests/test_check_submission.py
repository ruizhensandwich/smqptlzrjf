"""Pre-submission check: the notice's items that do not need a network or a venv."""
from tools.check_submission import (inference_code_hints, is_lfs_pointer, readme_problems,
                                    requirement_problems)


def test_requirements_need_versions():
    fails, warns = requirement_problems("# comment\ntorch==2.15.0\nnumpy>=1.26,<3\nscipy\n\n-r other.txt\n")
    assert fails == ["scipy: 버전 없음"]
    assert len(warns) == 1 and warns[0].startswith("numpy")
    assert requirement_problems("torch==2.15.0  # cpu is enough\n") == ([], [])


def test_lfs_pointer_is_not_weights(tmp_path):
    pointer = tmp_path / "a.pt"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 9\n")
    real = tmp_path / "b.pt"
    real.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    assert is_lfs_pointer(pointer) and not is_lfs_pointer(real)


def test_inference_code_hints(tmp_path):
    (tmp_path / "se" / "arch").mkdir(parents=True)
    (tmp_path / "denoise.py").write_text('"""Source: https://x\nmore text, curl\n"""\n'
                                         "HERE = Path(__file__).parent\n# see https://x\n", encoding="utf-8")
    (tmp_path / "se" / "arch" / "m.py").write_text(
        "W = 'C:\\\\Users\\\\me\\\\w.pt'\nsd = torch.hub.load_state_dict_from_url(u)\n", encoding="utf-8")
    downloads, paths = inference_code_hints(tmp_path)
    assert [d.split(":")[0] for d in downloads] == ["se/arch/m.py"]
    assert [p.split(":")[0] for p in paths] == ["se/arch/m.py"]


README = """팀 7조 · `submission-2` · Python 3.12 · pip install -r requirements.txt · python denoise.py
weights/submission.pt · seed · 학습 설정 · 학습 epoch · 제출 가중치의 선택 근거 · 선택 가중치의 검증 점수
## 사용한 공개 코드 | `weights/pretrained/x.pt` | MIT 라이선스 |
외부 데이터 | ESC-50 | 사용 범위 fold 1~4 |"""


def test_readme_items_of_the_notice():
    w = {"model": "mpsenet", "init_weights": "weights/pretrained/x.pt",
         "train_settings": {"data.external_noise": "esc50", "data.external_noise_prob": 0.5}}
    assert readme_problems(README, "7조", 2, w) == []
    short = README.replace("Python 3.12", "").replace("ESC-50", "")
    problems = readme_problems(short, "7조", 2, w)
    assert any("Python 버전" in p for p in problems) and any("esc50" in p for p in problems)
    w2 = {**w, "train_settings": {"data.external_speech": "zeroth"}}
    assert any("zeroth" in p for p in readme_problems(README, "7조", 2, w2))
    assert any("submissions.json" in p for p in readme_problems(README, "7조", 3, None))
