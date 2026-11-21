# 딥러닝 음성 잡음 제거 — 2026 신호 및 시스템 2차 챌린지 (7조)

PyTorch 딥러닝 모델로 16 kHz 모노 음성의 잡음을 줄인다. 순위 = 히든 테스트셋 파일별 **PESQ-WB 평균**
(재제출 시 검증 완료된 제출 중 최고 점수). 데이터·분할은 1차와 같다 (train 1,296쌍·화자 35명,
validation 164쌍·화자 5명, 3초, 잡음 범주 23개, SNR 0/5/10/15 dB).
저장소: https://github.com/ruizhensandwich/smqptlzrjf (public, 조교가 clone해서 실행). 배포 노트북:
`~/Downloads/deep_learning_noise_reduction_challenge/` (MaskCNN 예제, 배포 가중치 validation 1.692).

## 절대 규칙
- **딥러닝으로 푼다. 1차 챌린지의 전통 기법(1차 저장소 `nr/`의 OM-LSA·셸프·범주 라우팅 등)은 쓰지도 참고하지도 않는다** (사용자 결정, 2026-11-21). 1차에서 가져온 것은 제출·기록·내역서 도구뿐.
- **제출 입출력:** `input/`의 모든 WAV → `output/` **바로 아래** 같은 파일명, 16 kHz 모노, 입력과 같은 표본 수(포맷도 PCM_16 유지). 누락·추가 WAV·형식 오류·PESQ 오류 = 점수 보류. 처리 실패 시 원본 복사로라도 출력. 절대 경로 금지(`Path(__file__)` 기준).
- **추론은 `weights/submission.pt` 하나로.** 추론 중 다운로드 금지. 가중치 파일 95 MB 이하(GitHub 100 MB 한도).
- **런타임(denoise.py)은 noisy만 읽는다.** clean은 학습(`train.py`)과 채점(`evaluate.py`, `tools/`)에서만.
- **공개 금지:** `data/`(수강생 녹음), 모든 음성 파일, `docs/ai_usage/`(내역서), 인증 정보(서비스 계정 키, SSH 키, 토큰, `.env`). `.gitignore` + `tools/secret_scan.py`(pre-push 훅, `git config core.hooksPath .githooks`)로 막는다.
- **가중치 선택은 validation PESQ 최고 epoch** (공지 권장). 대신 train 점수도 함께 기록하고, validation에만 맞춘 설정 선택이 반복되지 않게 본다(히든 테스트는 새 화자·잡음).
- **AI 참고 문헌은 실재·내용을 직접 확인한 것만.** `docs/references.json`의 `checked_by`는 사람이 채운다.
- push: 일반 커밋과 제출 태그는 Claude가 push한다(사용자 요청). 런어스 비밀글 제출은 팀장이 직접.

## 환경 / 명령
로컬(Windows): conda 환경 `se2` (Python 3.12, conda-forge pesq 0.0.4, torch 2.15.0+cpu — 배포 노트북과 같은 버전).
`PYTHONIOENCODING=utf-8 conda run --no-capture-output -n se2 python <script>`
GPU 서버(GCP 서울 리전 L4/T4 할당량 1 — 프로젝트는 로컬 `gcloud config get project`, 또는 RunPod): `bash setup.sh` → `source .venv/bin/activate`. 긴 학습은 `tmux` 안에서. Spot이 끊기면 `python train.py --resume runs/<id>`.

| 목적 | 명령 |
|---|---|
| 테스트 | `python -m pytest -q` |
| 학습 | `python train.py --config configs/X.json --tag T [--set train.lr=5e-4] --note "..."` |
| 이어서 학습 | `python train.py --resume runs/<run_id>` |
| 채점 (16비트 출력 기준) | `python evaluate.py --weights runs/<id>/best.pt --split all --tag T` |
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
- 새 모델·손실: `se/models.py`·`se/losses.py`에 등록 → `configs/`에 설정 파일 → train → evaluate.
- `results/experiments.csv`(채점 기록), `results/training_runs.csv`(학습 기록), `results/submissions/roundN/` 커밋함. `runs/`는 git 제외.
- `submissions.json`과 README의 `<!-- submissions -->`·`<!-- reproduce -->` 블록은 `release.py`가 쓴다. 직접 편집 금지.
- `docs/ai_usage/`: 1차 양식 `template.hwpx` 그대로(2차 새 양식 없음). Part 3~6 제목만 딥러닝에 맞춤.

## 실험 프로토콜
1. 설정 파일 하나 = 실험 하나. `--note`에 무엇을 바꿨는지.
2. `train.py`(validation PESQ로 epoch 선택) → `evaluate.py --split all`로 train 1,296 + validation 164 전체를 16비트 출력으로 채점.
3. `compare.py`로 SNR·난이도·잡음 범주별로 손해 보는 그룹이 없는지 본다(평균만 보지 말 것).
4. `pytest` 통과 후 커밋.

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
