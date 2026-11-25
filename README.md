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
| 항목 | 값 |
|---|---|
| 제출 회차 | 1회차 (`submission-1`) |
| 모델 | `mpsenet` {"n_fft": 400, "hop": 100, "win": 400, "dense_channel": 64, "num_tsblocks": 4, "n_heads": 4, "beta": 2.0, "compress_factor": 0.3}, 파라미터 2,263,372개 |
| 가중치 파일 | `weights/submission.pt` (8.71 MB, 설정 포함) |
| 손실 함수 | `mpsenet` |
| seed | 1234 |
| 학습 epoch | 103 epoch 실행 (max_epochs), **100 epoch 가중치 선택** (팀 결정: 25 epoch마다 남긴 저장본 중 마지막인 100 epoch. validation과 새 잡음 점수는 85 epoch 무렵부터 거의 제자리이고 train 점수만 올랐다. 학습 중 validation PESQ 최고는 103 epoch의 3.0432) |
| 선택 가중치의 검증 점수 | 학습 중 Validation PESQ-WB 3.0224 (float 출력) · 제출 경로(16비트 WAV) 3.0225 |
| 학습 설정 | batch_size=4, lr=0.0005, adam_betas=[0.8, 0.99], weight_decay=0.01, max_epochs=150, min_epochs=40, patience=25, min_delta=0.01, lr_factor=0.5, lr_patience=5, min_lr=1e-05, optimizer=adamw, scheduler=exponential, lr_gamma=0.99, discriminator=metric, data.segment_seconds=2.0, data.remix_prob=0.5, data.remix_snr_db=[-5.0, 20.0], data.gain_db=[0.0, 0.0], data.speed_factors=[], data.speed_prob=0.0 |
| 학습 명령 | `python train.py --config configs/mpsenet_esc.json` |
| 학습 환경 | PyTorch 2.14.1+cu130, cuda |
| 출처 | trained here: runs/20261124-030714_mpsenet_esc/epoch_0100.pt |
<!-- reproduce:end -->

## 제출 기록

<!-- submissions:start -->
팀명: **7조** · 저장소: https://github.com/ruizhensandwich/smqptlzrjf

| 회차 | 태그 | 커밋 | 런어스 제출일 | 모델 | Validation PESQ | Validation SNR (dB) | Train PESQ | Train SNR (dB) | 히든 테스트 PESQ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `submission-1` | 태그 참조 | - | MP-SENet (2.26M), 처음부터 학습: 동적 재혼합 50 % + 외부 잡음 ESC-50 fold 1~4, 100 epoch 가중치 | 3.022 | 18.01 | 3.197 | 18.12 | - |

평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 (`git checkout <태그>`). 점수는 `denoise.py`가 쓰는 16비트 WAV 그대로 채점한 Validation 164개·Train 1,296개 전체의 파일별 평균이며, 상세 결과는 [`results/submissions/`](results/submissions/) (train은 `roundN/train/`).
<!-- submissions:end -->

## 학습과 평가 (재현)

데이터(`train/`, `validation/`)는 수강생 녹음이라 저장소에 올리지 않습니다. 배포 폴더의 두 폴더를 `data/` 아래에 복사하거나, 환경 변수 `DATA_ROOT`를 두 폴더가 들어 있는 위치로 지정하세요.

```bash
pip install -r requirements-dev.txt        # Windows는 pesq를 conda-forge로: conda install -c conda-forge pesq=0.0.4
python train.py --config configs/mask_cnn_baseline.json --tag baseline
python evaluate.py --weights runs/<run_id>/best.pt --split all --tag baseline
python tools/promote.py runs/<run_id>      # 그 가중치를 weights/submission.pt로

python train.py --config configs/mpsenet.json --tag mpsenet     # MP-SENet (GPU 권장)
python train.py --resume runs/<run_id>                          # 중단된 학습 이어서
```

- `train.py`: 매 epoch 학습 1,296쌍 → 검증 164개 전체의 손실과 평균 PESQ-WB. **검증 PESQ가 가장 높은 epoch의 가중치**를 고르고, 일정 기간 개선이 없으면 조기 종료합니다. 기록은 `runs/<run_id>/`와 `results/training_runs.csv`에 남습니다. 모델, 손실, 옵티마이저, 학습률 일정, 판별기 사용 여부, 데이터 옵션은 모두 `configs/*.json` 한 파일에 적습니다.
- **동적 재혼합**(`data.remix_prob`): 학습 쌍의 잡음(`noisy − clean`)을 다른 학습 발화에 무작위 SNR로 다시 섞어, 같은 데이터에서 화자·잡음·SNR 조합을 늘립니다. 학습 데이터만 쓰고, 메모리에서만 섞으며 파일로 저장하지 않습니다. `tools/check_mixing.py`가 `noisy − clean`이 실제로 깨끗한 잡음인지(위치 어긋남, 음량 차, SNR 일치, 무음) 파일마다 확인하고, 기준을 벗어난 파일은 재료에서 뺍니다. 결과는 [`results/mixing_check/`](results/mixing_check/)에 있습니다.
- `evaluate.py`: `denoise.py`와 같은 코드로 만든 16비트 출력을 clean과 비교해 PESQ-WB와 SNR을 잽니다. 결과는 `results/experiments.csv`에 기록합니다.
- **새 잡음 검증셋**(`tools/make_newnoise.py --fetch` → `evaluate.py --split newnoise`): 배포 validation의 잡음 녹음은 모두 train에도 들어 있어(화자만 새로움), validation PESQ로는 처음 듣는 잡음에서의 성능을 알 수 없습니다. 그래서 validation의 clean 164개와 SNR은 그대로 두고 잡음만 과제 데이터에 없는 것(ESC-50 fold 5)으로 바꾼 검증셋을 따로 채점합니다. 비교표의 "새 잡음 PESQ" 열입니다.
- `tools/spectrogram.py`: 같은 파일에 대해 Noisy·Clean·각 모델 출력의 스펙트로그램(같은 축·색 범위)과 PESQ/SNR, 같은 재생 게인의 WAV를 `results/listening/<시각>/index.html` 한 페이지로 만듭니다. 결과에는 수강생 녹음이 들어 있어 저장소에 올리지 않습니다.
- `tools/model_board.py`: 지금까지 평가한 모든 모델의 비교표(PESQ 기록, SNR·조건별 점수, 일반화 차, 학습·추론 비용, 장점·단점)를 [`results/model_board.md`](results/model_board.md)에 씁니다. `train.py`·`evaluate.py` 출력 끝에도 표시됩니다.
- 리눅스 GPU 서버(GCP, RunPod): `bash setup.sh` 한 번이면 학습 환경이 준비됩니다.

## 사용한 공개 코드

| 이 저장소의 파일 | 출처 | 라이선스 | 사용 방식 |
|---|---|---|---|
| `se/arch/mpsenet.py`, `se/arch/metric_discriminator.py`, `se/losses.py`의 `mpsenet`, `se/trainers.py`의 `MetricGANTrainer` | [yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet) (Lu, Ai, Ling, Interspeech 2023) | MIT | 모델·판별기·손실·학습 순서를 이식. 공개된 학습 가중치는 쓰지 않고 이 과제의 train 데이터로 처음부터 학습. 바꾼 점은 각 파일 첫머리에 기록 |
| `weights/pretrained/mpsenet_dns.pt` | 같은 저장소의 `best_ckpt/g_best_dns` (저자 공개 가중치, DNS Challenge 데이터로 학습) | MIT (`weights/pretrained/LICENSE.MP-SENet`) | 가중치 값은 그대로 두고 이 저장소의 파일 형식으로만 변환(`tools/promote.py --state-dict`). 이어 학습의 출발점(`configs/mpsenet_pretrained_dns.json`의 `init_weights`)으로 쓴다. 이 가중치는 원본 코드의 축 순서로 학습됐으므로 모델 설정 `source_axes: true`가 필요하다 |
| `se/arch/xlstm_senet/` (`generator.py`, `codec_module.py`, `xlstm_block.py`, `vision_lstm.py`, `vision_lstm_util.py`, `lsigmoid.py`, `stfts.py`, `mamba_init.py`) | [NikolaiKyhne/xLSTM-SENet](https://github.com/NikolaiKyhne/xLSTM-SENet) 커밋 `e195b64` (Kühne, Østergaard, Jensen, Tan, Interspeech 2025). `mamba_init.py`는 같은 저장소에 들어 있는 mamba_ssm의 `_init_weights` | MIT (`se/arch/xlstm_senet/LICENSE`), `mamba_init.py`는 Apache-2.0 (`LICENSE_mamba`) | 모델 코드를 그대로 두고(import 한 줄만 수정) `__init__.py`에서 이 저장소의 `enhance()` 인터페이스로 감쌌다. 공개된 학습 가중치는 쓰지 않았다. 가중치 `weights/candidates/xlstm_ep57.pt`는 팀원이 과제 train 데이터만으로 무작위 초기값에서 학습한 것(57 epoch, 재혼합 50 %, 외부 데이터 없음)을 `tools/promote.py --state-dict … --model xlstm_senet`으로 형식만 바꾼 것이다 |

외부 데이터 (저장소에 올리지 않음, 받는 방법은 각 도구의 `--fetch`)

| 데이터 | 출처 | 라이선스 | 사용 방식 |
|---|---|---|---|
| ESC-50 fold 1~4 (환경음 1,600개, 50종) | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, ACM MM 2015) | CC BY-NC 3.0 | **학습에** 사용. `tools/make_external_noise.py --fetch`가 잡음 풀을 만들고, 설정 `data.external_noise: "esc50"`, `data.external_noise_prob`만큼의 재혼합 항목이 과제 잡음 대신 이 잡음을 쓴다(`configs/mpsenet_esc.json`) |
| Zeroth-Korean (한국어 낭독 음성, 화자 115명, 16 kHz) | [OpenSLR 40](https://www.openslr.org/40/) (Jo & Lee, Atlas Guide·Gridspace) | CC BY 4.0 | **학습용 clean 음성**으로 준비. `tools/make_external_speech.py`가 배경 잡음이 있는 화자(말소리와 바닥 잡음의 간격 중앙값 45 dB 미만, 10명)와 넘친 파일을 빼고 화자당 최대 60개의 3초 구간을 모은다(105명, 6,096개). 설정 `data.external_speech: "zeroth"`, `data.external_speech_prob`만큼의 재혼합 항목이 과제 말소리 대신 이 음성을 쓴다 |
| ESC-50 fold 5 (환경음 400개, 50종) | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, ACM MM 2015) | CC BY-NC 3.0 | **채점에만** 사용. `tools/make_newnoise.py`가 validation 말소리 164개에 같은 SNR로 섞어 "새 잡음" 검증셋(`data/newnoise/`)을 만든다. 학습에는 쓰지 않는다 |

참고 문헌 목록은 [`docs/references.json`](docs/references.json)에 있습니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `denoise.py` | 제출용 추론: `input/` → `output/` |
| `weights/submission.pt` | 제출 가중치 (모델 설정 포함) |
| `se/` | 모델 등록(`models.py`), 손실(`losses.py`), 학습 단계(`trainers.py`), 데이터(`data.py`), 추론(`inference.py`), 채점(`metrics.py`) |
| `se/arch/` | 공개 코드에서 이식한 모델 (파일 하나에 모델 하나, 첫머리에 출처와 라이선스) |
| `train.py`, `evaluate.py`, `compare.py` | 학습, 채점, 두 실험 비교 |
| `configs/` | 실험 설정 (모델 구조, 학습 하이퍼파라미터) |
| `results/` | 실험 기록(`experiments.csv`, `training_runs.csv`), 모델 비교표(`model_board.md`), 회차별 제출 결과(`submissions/roundN/`) |
| `tools/` | 제출 기록·태그(`release.py`), 제출 전 점검(`check_submission.py`), 가중치 지정(`promote.py`), 모델 비교표(`model_board.py`), 스펙트로그램·청취(`spectrogram.py`), 공개 전 보안 검사(`secret_scan.py`) |
