# 딥러닝 음성 잡음 제거 — 2026 신호 및 시스템 2차 챌린지 (7조)

PyTorch 딥러닝 모델로 16 kHz 모노 음성의 잡음을 줄인다. 순위 = 히든 테스트셋 파일별 **PESQ-WB 평균**
(재제출 시 검증 완료된 제출 중 최고 점수). 데이터·분할은 1차와 같다 (train 1,296쌍·화자 35명,
validation 164쌍·화자 5명, 3초, 잡음 범주 23개, SNR 0/5/10/15 dB).
저장소: https://github.com/ruizhensandwich/smqptlzrjf (조교가 clone해서 실행. 지금은 비공개이고 제출 직전에 공개로 전환한다). 배포 노트북:
`~/Downloads/deep_learning_noise_reduction_challenge/` (MaskCNN 예제, 배포 가중치 validation 1.692).

## 절대 규칙
- **딥러닝으로 푼다. 1차 챌린지의 전통 기법(1차 저장소 `nr/`의 OM-LSA·셸프·범주 라우팅 등)은 쓰지도 참고하지도 않는다** (사용자 결정, 2026-11-21). 1차에서 가져온 것은 제출·기록·내역서 도구뿐.
- **제출 입출력:** `input/`의 모든 WAV → `output/` **바로 아래** 같은 파일명, 16 kHz 모노, 입력과 같은 표본 수(포맷도 PCM_16 유지). 누락·추가 WAV·형식 오류·PESQ 오류 = 점수 보류. 처리 실패 시 원본 복사로라도 출력. 절대 경로 금지(`Path(__file__)` 기준).
- **추론은 `weights/submission.pt` 하나로.** 추론 중 다운로드 금지. 가중치 파일 95 MB 이하(GitHub 100 MB 한도).
- **런타임(denoise.py)은 noisy만 읽는다.** clean은 학습(`train.py`)과 채점(`evaluate.py`, `tools/`)에서만.
- **공개 금지:** `data/`(수강생 녹음), 모든 음성 파일, `docs/ai_usage/`(내역서), 인증 정보(서비스 계정 키, SSH 키, 토큰, `.env`). `.gitignore` + `tools/secret_scan.py`(pre-push 훅, `git config core.hooksPath .githooks`)로 막는다.
- **외부 공개 데이터(음성·잡음)를 학습에 쓸 수 있다** (사용자 통보, 2026-11-23). 학습에만 쓰고, 출처·라이선스를 README에 적고, 저장소에는 올리지 않는다. **공개 pretrained 가중치도 쓸 수 있다**(사용자 통보, 2026-11-23): 학습의 출발점으로만 쓰고 추론은 여전히 `weights/submission.pt` 하나로, 출처·라이선스·학습 데이터를 README에 적는다.
- **validation은 새 화자만 잰다.** validation의 잡음 원본(NOISEX-92 14개, Nonspeech 47개)은 전부 train에도 있다(2026-11-23 확인). 새 잡음에 대한 일반화는 validation PESQ로 알 수 없으므로 따로 잰다.
- **가중치 선택은 validation PESQ 최고 epoch** (공지 권장). 대신 train 점수도 함께 기록하고, validation에만 맞춘 설정 선택이 반복되지 않게 본다(히든 테스트는 새 화자·잡음).
- **AI 참고 문헌은 실재·내용을 직접 확인한 것만.** `docs/references.json`의 `checked_by`는 사람이 채운다.
- push: 일반 커밋과 제출 태그는 Claude가 push한다(사용자 요청). 런어스 비밀글 제출은 팀장이 직접.

## 환경 / 명령
로컬(Windows): conda 환경 `se2` (Python 3.12, conda-forge pesq 0.0.4, torch 2.15.0+cpu — 배포 노트북과 같은 버전). numpy BLAS는 **OpenBLAS**로 둔다: conda-forge MKL은 LLVM OpenMP(libomp)를 불러 torch의 libiomp5md와 충돌한다(OMP Error #15, matplotlib 그림에서 발생).
`PYTHONIOENCODING=utf-8 conda run --no-capture-output -n se2 python <script>`
GPU 서버(GCP 서울 리전 L4/T4 할당량 1 — 프로젝트는 로컬 `gcloud config get project`, 또는 RunPod): `bash setup.sh` → `source .venv/bin/activate`. 긴 학습은 `tmux` 안에서. Spot이 끊기면 `python train.py --resume runs/<id>`.

| 목적 | 명령 |
|---|---|
| 테스트 | `python -m pytest -q` |
| 학습 (끝나면 자동 채점 + 모델 비교표) | `python train.py --config configs/X.json --tag T [--set train.lr=5e-4] --note "..."` |
| 이어서 학습 | `python train.py --resume runs/<run_id>` |
| 재혼합 재료 확인 (noisy − clean이 깨끗한 잡음인지) | `python tools/check_mixing.py` → `results/mixing_check/` |
| 채점 (16비트 출력 기준) | `python evaluate.py --weights runs/<id>/best.pt --split all --tag T` |
| 모델 비교표 (모든 모델의 PESQ·장단점) | `python tools/model_board.py` → `results/model_board.md` |
| 스펙트로그램·청취 비교 | `python tools/spectrogram.py --models runs/<A> runs/<B> [--pick diverse\|worst\|best --n 4]` / `<segment_id> ...` / `--noisy any.wav [--clean ref.wav]` → `results/listening/<time>/index.html` (학생 녹음 포함: git 제외·공개 금지) |
| 두 실행 비교 | `python compare.py <runA> <runB> [--split train]` |
| 제출 가중치 지정 | `python tools/promote.py runs/<run_id>` → `weights/submission.pt` |
| 공개 전 보안 검사 | `python tools/secret_scan.py` |
| N회차 제출 결과 | `python evaluate.py --round N` → `results/submissions/roundN/` (+ `train/`) |
| 제출 기록·README | `python tools/release.py prepare N --summary "..."` → 커밋 → `release.py tag N` → `release.py push N` |
| 제출 전 확인 | `python tools/check_submission.py --round N` |
| AI 활용 내역서 | `python tools/build_ai_usage.py docs/ai_usage/roundN.json [--check]` |
| 참고 문헌 DOI 확인 | `python tools/check_refs.py --write` |

## 구조
- `se/`: `models.py`(MODELS 레지스트리, 모델은 `enhance(noisy[B,T]) -> [B,T]`), `losses.py`(LOSSES, `loss(model, noisy, clean)`), `config.py`(기본값 = 배포 노트북, 모르는 키는 오류), `data.py`(DATA_ROOT 또는 `data/`, 파형 캐시), `checkpoint.py`(가중치+모델 설정+학습 정보를 한 파일에), `inference.py`(denoise와 evaluate가 공유, 16비트로 쓰고 다시 읽은 값으로 채점), `metrics.py`.
- `se/arch/`: 공개 코드에서 이식한 모델(파일 하나에 모델 하나, 첫머리에 출처·라이선스·바꾼 점). `se/trainers.py`: 한 epoch 학습(`Trainer` = 손실 하나, `MetricGANTrainer` = PESQ 예측 판별기 포함). 설정의 `discriminator`·`data.*`(구간 자르기, 동적 재혼합)·`train.optimizer/scheduler`로 고른다. 기본값은 배포 노트북 그대로.
- 새 모델: `se/arch/<이름>.py` → `se/models.py`의 MODELS에 한 줄(손실이 새로 필요하면 `se/losses.py`) → `configs/<이름>.json` → train → evaluate. MP-SENet 계열(analysis/forward/synthesis 인터페이스)은 손실 `mpsenet`과 판별기를 그대로 쓴다.
- 동적 재혼합(`se/data.py` `RemixedSpeech`): train 쌍의 잡음(noisy − clean)을 다른 train 발화에 무작위 SNR로 섞는다. train 데이터만, 메모리에서만. 재료 기준은 `REMIX_LIMITS`(2026-11-22 확인: 1,296개 모두 지연 0·SNR 오차 0.002 dB 이내, 잡음이 대부분 무음인 6개만 제외).
- `results/experiments.csv`(채점 기록), `results/training_runs.csv`(학습 기록), `results/model_board.csv`·`.md`(모델 비교표), `results/submissions/roundN/` 커밋함. `runs/`는 git 제외.
- **모델 비교표**(`tools/model_board.py`): validation 전체를 채점한 모든 모델을 한 표에 (Val PESQ·ΔPESQ·SNR, 0 dB·15 dB·hard, 입력보다 나빠진 파일 비율, Train PESQ, 일반화 차 = train 개선 − validation 개선, 학습 epoch·시간, 추론 속도) + 모델별 장점·단점(다른 모델 대비 자동 판정, 잡음 범주는 train 기준). evaluate.py와 train.py 출력 끝에 찍힌다. 같은 가중치는 한 줄.
- `submissions.json`과 README의 `<!-- submissions -->`·`<!-- reproduce -->` 블록은 `release.py`가 쓴다. 직접 편집 금지.
- `docs/ai_usage/`: 1차 양식 `template.hwpx` 그대로(2차 새 양식 없음). Part 3~6 제목만 딥러닝에 맞춤.

## 실험 프로토콜
1. 설정 파일 하나 = 실험 하나. `--note`에 무엇을 바꿨는지.
2. `train.py`(validation PESQ로 epoch 선택) → 끝나면 자동으로 `evaluate.py --split all`(train 1,296 + validation 164 전체, 16비트 출력) → 모델 비교표.
3. **사용자에게 새 모델 결과를 보고할 때는 응답 맨 끝에 `results/model_board.md`의 비교표 전체를 붙인다** (사용자 요청, 2026-11-21).
4. `compare.py`로 SNR·난이도·잡음 범주별로 손해 보는 그룹이 없는지 본다(평균만 보지 말 것). `tools/spectrogram.py --pick worst`로 손해 본 파일을 보고 듣는다(공지: 잡음 감소와 말소리 보존을 소리·스펙트로그램으로 확인).
5. `pytest` 통과 후 커밋.

## 제출 절차 (팀당 주 1회, 최초 제출은 횟수 제외)
1. `pytest` 통과 → `tools/promote.py runs/<id>` → `weights/submission.pt`
2. `evaluate.py --round N` → `release.py prepare N --summary "..."`
3. `docs/ai_usage/roundN.json`(최초: 접근 방법·AI 활용·검증 / 이후: 이전 대비 방법론 변화·AI 활용·검증 결과, results.rows에 `round{N-1}` vs `roundN`) → `build_ai_usage.py`
4. 커밋 → `release.py tag N` → `release.py push N` (secret scan 후 main + 태그 push)
5. `check_submission.py --round N` → READY
6. 팀장이 런어스 → 질의응답 비밀글: 저장소 링크 + 태그 `submission-N`(커밋 해시) + 내역서 hwpx
7. `release.py submitted N`, 리더보드 공개 후 `release.py leaderboard N --pesq X`

## 현재 기록
- 배포 예제 MaskCNN(seed 42, 109 epoch 중 89 epoch 선택): validation 1.692 (노트북 float 채점).
- 이 저장소의 `train.py`로 CPU 재현(2026-11-21, `runs/20261121-175901_baseline_repro`, 9.8분): 109 epoch 모두 손실·PESQ 차이 0, 선택 가중치가 배포 가중치와 비트 단위로 같음. 현재 `weights/submission.pt`.
  16비트 출력 채점: **validation 1.6922 / SNR 13.06 dB, train 1.8776 / 13.11 dB** (입력 1.395 / 1.470). train 개선 +0.41 대 validation +0.30.
  약점(스펙트로그램): 저역 잡음(바람·차량 실내)을 지울 때 300 Hz 아래 말소리 배음까지 지운다 — validation에서 가장 손해 본 3개 파일(traffic 15 dB −0.61, wind 15 dB −0.56, volvo 5 dB −0.45)이 모두 이 경우.
- MP-SENet(`se/arch/mpsenet.py`, 2.26M, 크기는 마스킹·위상은 매핑)을 처음부터 학습(2026-11-22, GCP L4 11시간,
  `configs/mpsenet.json` + 재혼합 50 %, 116 epoch 중 115 epoch 선택): **validation 3.132 / SNR 18.40 dB,
  train 3.303** (일반화 차 +0.094). validation 164개 전부 베이스라인보다 좋고 입력보다 나빠진 파일 0개.
  약점: 넓은 대역의 정상 잡음(hf_channel 2.35, broadband 2.45)과 0 dB(2.37).
  가중치는 `weights/candidates/mpsenet_remix50.pt`에 보관. **`weights/submission.pt`는 아직 베이스라인**(1회차 미제출).
- 같은 설정에 말소리 속도 변환(0.9~1.1배)만 추가(`configs/mpsenet_speed.json`): validation 3.113 / train 3.284,
  일반화 차 +0.095. epoch 간 흔들림(±0.02) 안이라 **효과 없음**.
- MP-SENet 저자 공개 가중치(DNS Challenge 학습, `weights/pretrained/mpsenet_dns.pt`, 모델 설정 `source_axes: true` 필요)를
  그대로 채점(2026-11-23): **validation 3.176 / train 2.980**. 과제 데이터를 본 적 없는 모델이 train보다 validation에서
  0.27 더 개선한다(validation이 더 쉬운 묶음).
- 그 가중치를 과제 train으로 이어 학습(`configs/mpsenet_pretrained_dns.json`, lr 1e-4, L4 2.8시간, 29 epoch에서 조기 종료,
  29 epoch 선택): **validation 3.318 / SNR 19.82 dB, train 3.457** (일반화 차 +0.063). 현재 최고.
  가중치는 `weights/candidates/mpsenet_dns_ft.pt`. 공개 가중치 대비 validation +0.14, train +0.48: 오른 폭의 대부분이
  train의 화자·잡음에 맞춘 것이라, 새 잡음에서 이득인지는 아직 재지 않았다(새 잡음 검증셋 필요).
- 학습 곡선의 모양: 60 epoch쯤 3.0, 80 epoch 이후 시간당 +0.03 이하, 막바지에 train만 오르고 validation이 멈춤.
  중단 조건은 `min_delta 0.01`, `patience 25`, `min_epochs 40`, `max_epochs 150`(사용자와 합의), 끝나면 VM 정지.

## 작업 방식 (사용자 요청, 2026-11-22)
- 실험은 한 번에 하나만 바꾼다.
- 로컬 CPU에서는 pytest와 파일 몇 개짜리 확인만. 학습과 전체 채점은 GPU VM에서. CPU 추론 속도는 모델 선택 기준이 아니다.
- 일반화가 최우선이다(히든 테스트는 새 화자·잡음). validation 0.0x 차이보다 과적합 여부를 본다.
- GPU VM은 Python 3.10이다. 학습을 시작하기 전에 VM에서 pytest와 `py_compile`을 돌린다.
