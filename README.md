# 딥러닝 음성 잡음 제거 — 2026 신호 및 시스템 2차 챌린지 (7조)

16 kHz 모노 음성의 배경 잡음을 PyTorch 딥러닝 모델로 줄입니다.
`input/`의 모든 WAV를 처리해 `output/` 바로 아래에 같은 파일명으로 저장합니다.

## 실행 방법 (추론)

**실행 환경:** Python 3.10 이상 (3.12에서 확인), Windows·macOS·Linux. GPU가 없어도 CPU로 실행됩니다.

```bash
pip install -r requirements.txt
python denoise.py
```

- 저장소 최상위 폴더에서 실행합니다. `input/`의 `*.wav`를 모두 읽어 `output/<같은 파일명>.wav`로 씁니다.
- 출력은 입력과 같은 표본률(16 kHz), 채널 수(모노), 표본 수, 포맷(PCM_16)입니다.
- **필요한 패키지:** `requirements.txt` (torch, numpy, scipy, soundfile). 추론 중 추가 다운로드는 없습니다.
- **모델·가중치 경로:** `weights/submission.pt`. 이 파일에 모델 이름과 설정, 가중치가 함께 들어 있어 다른 설정 파일이 필요 없습니다.
- **경로 수정:** 개인 PC의 절대 경로를 고칠 곳은 없습니다. 모든 경로는 `denoise.py` 위치 기준입니다. 다른 폴더를 쓰려면 `python denoise.py --input <입력 폴더> --output <출력 폴더>`를 쓰면 됩니다.
- 처리할 수 없는 파일이 있으면 원본을 그대로 복사해, 입력마다 출력이 하나씩 반드시 생깁니다.
- `--device cuda`로 GPU를 쓸 수 있고, 기본값 `auto`는 GPU가 있으면 GPU를 씁니다.

## 제출 가중치의 학습 정보

<!-- reproduce:start -->
_아직 제출 기록이 없습니다._
<!-- reproduce:end -->

## 제출 기록

<!-- submissions:start -->
팀명: **7조** · 저장소: https://github.com/ruizhensandwich/smqptlzrjf

| 회차 | 태그 | 커밋 | 런어스 제출일 | 모델 | Validation PESQ | Validation SNR (dB) | Train PESQ | Train SNR (dB) | 히든 테스트 PESQ |
|---|---|---|---|---|---|---|---|---|---|

평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 (`git checkout <태그>`).
<!-- submissions:end -->

## 학습과 평가 (재현)

데이터(`train/`, `validation/`)는 수강생 녹음이라 저장소에 올리지 않습니다. 배포 폴더의 두 폴더를 `data/` 아래에 복사하거나, 환경 변수 `DATA_ROOT`를 두 폴더가 들어 있는 위치로 지정하세요.

```bash
pip install -r requirements-dev.txt        # Windows는 pesq를 conda-forge로: conda install -c conda-forge pesq=0.0.4
python train.py --config configs/mask_cnn_baseline.json --tag baseline
python evaluate.py --weights runs/<run_id>/best.pt --split all --tag baseline
python tools/promote.py runs/<run_id>      # 그 가중치를 weights/submission.pt로
```

- `train.py`: 매 epoch 학습 1,296쌍 → 검증 164개 전체의 손실과 평균 PESQ-WB. **검증 PESQ가 가장 높은 epoch의 가중치**를 고릅니다. 검증 PESQ가 정체되면 학습률을 낮추고, 일정 기간 개선이 없으면 조기 종료합니다. 기록은 `runs/<run_id>/`와 `results/training_runs.csv`에 남습니다.
- `evaluate.py`: `denoise.py`와 같은 코드로 만든 16비트 출력을 clean과 비교해 PESQ-WB와 SNR을 잽니다. 결과는 `results/experiments.csv`에 기록합니다.
- `tools/spectrogram.py`: 같은 파일에 대해 Noisy·Clean·각 모델 출력의 스펙트로그램(같은 축·색 범위)과 PESQ/SNR, 같은 재생 게인의 WAV를 `results/listening/<시각>/index.html` 한 페이지로 만듭니다. 결과에는 수강생 녹음이 들어 있어 저장소에 올리지 않습니다.
- `tools/model_board.py`: 지금까지 평가한 모든 모델의 비교표(PESQ 기록, SNR·조건별 점수, 일반화 차, 학습·추론 비용, 장점·단점)를 [`results/model_board.md`](results/model_board.md)에 씁니다. `train.py`·`evaluate.py` 출력 끝에도 표시됩니다.
- 리눅스 GPU 서버(GCP, RunPod): `bash setup.sh` 한 번이면 학습 환경이 준비됩니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `denoise.py` | 제출용 추론: `input/` → `output/` |
| `weights/submission.pt` | 제출 가중치 (모델 설정 포함) |
| `se/` | 모델(`models.py`), 손실(`losses.py`), 데이터(`data.py`), 추론(`inference.py`), 채점(`metrics.py`) |
| `train.py`, `evaluate.py`, `compare.py` | 학습, 채점, 두 실험 비교 |
| `configs/` | 실험 설정 (모델 구조, 학습 하이퍼파라미터) |
| `results/` | 실험 기록(`experiments.csv`, `training_runs.csv`), 모델 비교표(`model_board.md`), 회차별 제출 결과(`submissions/roundN/`) |
| `tools/` | 제출 기록·태그(`release.py`), 제출 전 점검(`check_submission.py`), 가중치 지정(`promote.py`), 모델 비교표(`model_board.py`), 스펙트로그램·청취(`spectrogram.py`), 공개 전 보안 검사(`secret_scan.py`) |
