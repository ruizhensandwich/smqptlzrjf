# 딥러닝 음성 잡음 제거 — 2026 신호 및 시스템 2차 챌린지 (7조)

PyTorch 딥러닝 모델로 16 kHz 모노 음성의 잡음을 줄인다. 순위 = 히든 테스트셋 파일별 **PESQ-WB 평균**
(재제출 시 검증 완료된 제출 중 최고 점수). 데이터·분할은 1차와 같다 (train 1,296쌍·화자 35명,
validation 164쌍·화자 5명, 3초, 잡음 범주 23개, SNR 0/5/10/15 dB).
저장소: https://github.com/ruizhensandwich/smqptlzrjf (공개, 조교가 clone해서 실행). 배포 노트북:
`~/Downloads/deep_learning_noise_reduction_challenge/` (MaskCNN 예제, 배포 가중치 validation 1.692).

## 일정·평가 (2차 챌린지 공지, 사용자 전달 2026-11-26)
- 챌린지 11-18 ~ **12-13** (KST, `tools/common.py` `DEADLINE`), **최종 보고서 12-14 마감**, 결과 이의 신청 12-14부터.
- 히든 테스트 **483개**, 배포 train·validation에 **없는 잡음도 포함**. 순위 = 파일별 PESQ-WB 평균(파일마다 같은 가중치), 보조 = SNR.
  리더보드는 검증을 마친 제출 중 팀별 최고 점수.
- 최종 성적 = 순위 점수 + 최종 보고서(알고리즘 이해, 실험·검증의 충실성, 결과 해석의 타당성).
- **최종 보고서:** A4 10쪽 내외, 본문 맑은 고딕·나눔고딕 같은 글꼴 11 pt 통일. 담을 것:
  1. 진행 과정(초기 접근, 회차별 변경과 이유, 기대에 못 미친 실험에서 확인한 점)
  2. 알고리즘(Noisy → 복원 음성의 처리 과정, 입력 표현·모델 구조·손실·후처리 중 중요했던 선택과 이유, 흐름도·구조도)
  3. 데이터와 학습 설정(학습·검증 데이터 구성, 전처리·증강, 주요 설정, 모델·가중치 선택 기준, 외부 자료 출처·범위)
  4. 실험 결과와 분석(같은 Validation 조건에서 예제 모델·이전 제출·주요 실험·최종 모델의 PESQ-WB·SNR 표, 대표 사례의 처리 전후 스펙트로그램과 청취: 줄어든 잡음, 말소리 보존, 성능이 떨어진 조건)
  5. 한계와 개선 방향(취약한 잡음 조건, 다음 개선 방법), 최종 제출 커밋·태그, 참고 문헌
  → 그래서 모든 실험은 `--note`에 무엇을 왜 바꿨는지 남기고, 기대에 못 미친 실험도 결과를 기록한다(보고서 1·4절 재료).
- **매 제출 전 확인 8항목**은 `tools/check_submission.py`가 공지 번호 그대로 점검한다(아래 "제출 절차").

## 절대 규칙
- **PESQ를 높이는 수단은 가리지 않는다** (조교 답변, 사용자 전달 2026-11-25): 공개 pretrained 가중치, 후처리, 외부 데이터 모두 허용. 지킬 것은 제출 입출력 계약, 추론 중 다운로드 금지(가중치는 저장소 안에), 출처·라이선스 표기(README "사용한 공개 코드"·외부 데이터 표와 내역서).
- 후처리에 쓸 기법은 사용자가 정한다(2026-11-25). **PCS를 적용·검증하기로 했다**(사용자 결정 2026-11-26). 검증셋 전체에서 고르게 오르는지 강도별로 확인한 뒤에 제출 경로에 넣는다. 그 밖의 후처리는 넣지 않는다.
- **제출 입출력:** `input/`의 모든 WAV → `output/` **바로 아래** 같은 파일명, 16 kHz 모노, 입력과 같은 표본 수(포맷도 PCM_16 유지). 누락·추가 WAV·형식 오류·PESQ 오류 = 점수 보류. 처리 실패 시 원본 복사로라도 출력. 절대 경로 금지(`Path(__file__)` 기준).
- **추론은 `weights/submission.pt` 하나로.** 추론 중 다운로드 금지. 가중치 파일 95 MB 이하(GitHub 100 MB 한도).
- **런타임(denoise.py)은 noisy만 읽는다.** clean은 학습(`train.py`)과 채점(`evaluate.py`, `tools/`)에서만.
- **공개 금지:** `data/`(수강생 녹음), 모든 음성 파일, `docs/ai_usage/`(내역서), 인증 정보(서비스 계정 키, SSH 키, 토큰, `.env`). `.gitignore` + `tools/secret_scan.py`(pre-push 훅, `git config core.hooksPath .githooks`)로 막는다.
- **외부 공개 데이터(음성·잡음)를 학습에 쓸 수 있다** (조교 답변 2026-11-23, 공지 2026-11-26). 저장소에는 올리지 않고, README 외부 데이터 표에 **출처·버전, 라이선스, 사용 범위(분할·폴드), 학습에 쓴 방식, 검증에 쓴 방식**을 각각 적는다(공지). 새 외부 데이터를 쓰면 그 표에 한 줄을 추가하고 `tools/check_submission.py`의 `OUTSIDE_DATA`에 설정 이름 → README 표기 이름을 넣는다.
- **공개 코드·사전학습 가중치는 출처, 버전(커밋·파일), 적용 방식**을 README "사용한 공개 코드" 표에 적는다(공지).
- **실행 안내(공지):** README에 Python 버전(3.12), 패키지와 버전, 설치·추론 명령. `requirements.txt`는 확인한 버전을 `==`로 고정한다(torch 2.15.0, numpy 2.5.5, scipy 1.18.1, soundfile 0.14.0). 제출 모델이 다른 패키지(예: `einops`)를 쓰면 버전과 함께 추가하고 `check_submission.py`로 새 환경 설치를 확인한다. 가중치는 100 MB 미만이라 Git LFS를 쓰지 않는다(쓰게 되면 점검 3이 LFS 포인터를 잡는다).
- **공개 pretrained 가중치는 제출에 써도 된다** (조교 답변, 2026-11-25. 그 전 11-23에는 미확인이라 제출에서 뺐다). 쓰면 출처·라이선스·그 가중치의 학습 데이터를 README의 "사용한 공개 코드" 표와 내역서에 적는다. 이어 학습한 가중치는 `release.py`가 README 재현 블록에 "시작 가중치"(설정의 `init_weights`와 그 파일의 `source`)를 자동으로 적고, `check_submission.py` 7c가 README에 그 파일의 출처가 있는지 본다. 받은 원본 가중치는 이 저장소 형식으로 바꿔(`tools/promote.py --state-dict`) `weights/pretrained/`에 둔다(95 MB 이하).
- **validation은 새 화자만 잰다.** validation의 잡음 원본(NOISEX-92 14개, Nonspeech 47개)은 전부 train에도 있다(2026-11-23 확인). 새 잡음에 대한 일반화는 **새 잡음 검증셋**(`data/newnoise/`: validation clean 164개 + 같은 SNR + ESC-50 fold 5 잡음, `tools/make_newnoise.py`)으로 따로 잰다. **ESC-50 fold 5는 학습에 절대 쓰지 않는다**(외부 잡음으로 학습할 때는 fold 1~4나 다른 출처).
- **학습은 매 epoch의 가중치를 남긴다** (`train.save_every_epoch`, 기본값 true, `runs/<id>/epoch_NNNN.pt`; 사용자 요청 2026-11-25: 87 epoch를 고르고 싶었지만 저장본이 없었다).
- **가중치 선택은 validation PESQ 최고 epoch** (공지 권장). validation과 어려운 검증셋(`data/hard/`)을 둘 다 주요 지표로 본다(사용자 결정 2026-11-26). 히든 테스트는 483개이고 배포 데이터에 없는 잡음도 들어 있다(공지 2026-11-26). 화자 구성은 알려지지 않았고, 조교는 "1차보다 어렵다"고 했다. 그래서 새 잡음 검증셋·어려운 검증셋의 점수를 validation과 함께 본다. README에는 제출 가중치의 **선택 근거**를 적는다(`release.py`가 설정의 `selection` 또는 "Validation PESQ 최고 epoch"를 재현 블록에 쓴다).
- **AI 참고 문헌은 실재·내용을 직접 확인한 것만.** `docs/references.json`의 `checked_by`는 사람이 채운다.
- push: 일반 커밋과 제출 태그는 Claude가 push한다(사용자 요청). 런어스 비밀글 제출은 팀장이 직접.

## 환경 / 명령
로컬(Windows): conda 환경 `se2` (Python 3.12, conda-forge pesq 0.0.4, torch 2.15.0+cpu — 배포 노트북과 같은 버전). numpy BLAS는 **OpenBLAS**로 둔다: conda-forge MKL은 LLVM OpenMP(libomp)를 불러 torch의 libiomp5md와 충돌한다(OMP Error #15, matplotlib 그림에서 발생).
`PYTHONIOENCODING=utf-8 conda run --no-capture-output -n se2 python <script>`
GPU 서버(GCP 서울 리전 L4/T4 할당량 1 — 프로젝트는 로컬 `gcloud config get project`, 또는 RunPod): `bash setup.sh` → `source .venv/bin/activate`. 긴 학습은 `tmux` 안에서. Spot이 끊기면 `python train.py --resume runs/<id>`. `--resume`은 run 폴더의 `config.json`만 읽는다(`--config`·`--set`과 함께 쓰면 오류). 손으로 멈춘 실행을 마무리하려면 그 `config.json`의 `max_epochs`를 끝난 epoch로 바꾸고 `--resume`.

| 목적 | 명령 |
|---|---|
| 테스트 | `python -m pytest -q` |
| 학습 (끝나면 자동 채점 + 모델 비교표) | `python train.py --config configs/X.json --tag T [--set train.lr=5e-4] --note "..."` |
| 이어서 학습 | `python train.py --resume runs/<run_id>` |
| 재혼합 재료 확인 (noisy − clean이 깨끗한 잡음인지) | `python tools/check_mixing.py` → `results/mixing_check/` |
| 채점 (16비트 출력 기준) | `python evaluate.py --weights runs/<id>/best.pt --split all --tag T` |
| 새 잡음 검증셋 만들기 / 채점 | `python tools/make_newnoise.py [--fetch]` → `data/newnoise/` / `python evaluate.py --weights W --split newnoise --tag T` (`--split all`은 있으면 함께 채점) |
| 어려운 검증셋 만들기 / 채점 | `python tools/make_hardset.py` → `data/hard/` (validation 말소리 164개 × 4묶음: lowsnr·newlow·babble·reverb, ESC-50 fold 5와 Zeroth 제외 화자 10명 필요) / `python evaluate.py --weights W --split hard --tag T --no-board` (묶음별 한 줄) |
| 외부 잡음 풀(학습용, ESC-50 fold 1~4) | `python tools/make_external_noise.py [--fetch]` → `data/external/esc50/train_noise_16k.npy` (설정 `data.external_noise`, `data.external_noise_prob`) |
| 외부 음성 풀(학습용 clean, Zeroth-Korean) | `python tools/make_external_speech.py [--min-range-db 45 --per-speaker 60]` → `data/external/zeroth/train_speech_16k.npy` + `speakers.csv` (설정 `data.external_speech`, `data.external_speech_prob`). 원본은 VM의 `data/external/zeroth/full/`에만 있다 |
| 모델 비교표 (모든 모델의 PESQ·장단점) | `python tools/model_board.py` → `results/model_board.md` |
| 스펙트로그램·청취 비교 | `python tools/spectrogram.py --models runs/<A> runs/<B> [--pick diverse\|worst\|best --n 4]` / `<segment_id> ...` / `--noisy any.wav [--clean ref.wav]` → `results/listening/<time>/index.html` (학생 녹음 포함: git 제외·공개 금지) |
| 두 실행 비교 | `python compare.py <runA> <runB> [--split train]` |
| 섞기 연구 (멤버 출력 저장 → 정렬·음량 진단 → 조합·결합 방식 비교, VM) | `python tools/mix_study.py dump --members L=W1 M=W2 …` → `align` → `gain --member L [--score]` → `mix --grid --gain L:proj:1.484:1.35:1.65` → `mix --combos … --method mag` → `report` (`runs/mix_study/`) |
| 여러 모델을 한 가중치 파일로 (출력 파형의 고정 비율 평균) | `python tools/make_ensemble.py --members A.pt B.pt --weights 0.7 0.3 --out weights/candidates/E.pt` (모델 `ensemble`, 비율은 validation으로 큰 단위만 고른다) |
| 다른 곳에서 학습한 가중치 들여오기 | `python tools/promote.py --state-dict X.pt --model xlstm_senet --source "..." --out weights/candidates/Y.pt` |
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
- `se/arch/semamba/`: SEMamba(공개 코드 + 순수 PyTorch Mamba `mamba_torch.py` + `enhance()` 래퍼, `einops` 필요). 추론만 한다.
- `se/arch/xlstm_senet/`: xLSTM-SENet(공개 코드 그대로 + `__init__.py`의 `enhance()` 래퍼, `einops` 필요). 이 저장소에서는 추론만 한다(학습은 팀원의 별도 하네스에서 했고 그 학습 코드는 이 저장소에 없다). `se/models.py`의 `ensemble`은 여러 모델을 한 파일에 담아 출력을 평균한다.
- 새 모델: `se/arch/<이름>.py` → `se/models.py`의 MODELS에 한 줄(손실이 새로 필요하면 `se/losses.py`) → `configs/<이름>.json` → train → evaluate. MP-SENet 계열(analysis/forward/synthesis 인터페이스)은 손실 `mpsenet`과 판별기를 그대로 쓴다.
- 동적 재혼합(`se/data.py` `RemixedSpeech`): train 쌍의 잡음(noisy − clean)을 다른 train 발화에 무작위 SNR로 섞는다. train 데이터만, 메모리에서만. 재료 기준은 `REMIX_LIMITS`(2026-11-22 확인: 1,296개 모두 지연 0·SNR 오차 0.002 dB 이내, 잡음이 대부분 무음인 6개만 제외).
- `train.py`는 `data/newnoise/`가 있으면 매 epoch 새 잡음 PESQ도 재서 `history.csv`의 `newnoise_pesq`에 적는다. 보기만 하고, 가중치 선택은 여전히 validation PESQ.
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
3. `docs/ai_usage/roundN.json` → `build_ai_usage.py`. 공지가 요구하는 내용: 사용한 AI 모델과 목적, 핵심 질문과 답변 요약(cases), 모델 구조·학습 방법의 핵심 원리와 선택 이유(algorithm; 최초는 선택한 이유, 이후는 이전 회차 대비 달라진 점), AI 제안을 직접 검증·수정한 부분(cases.verify), 변경 전후 **Validation** PESQ-WB·SNR(results.rows, Validation 행 2개 이상: `round{N-1}` vs `roundN`), 직접 확인한 문헌·외부 코드·사전학습 가중치·데이터의 출처와 활용 부분(references, external_code). AI를 쓰지 않은 회차는 미사용 사실과 직접 한 실험·검증 결과.
4. 커밋 → `release.py tag N` → `release.py push N` (secret scan 후 main + 태그 push)
5. `check_submission.py --round N` → READY. **공지의 "매 제출 전 확인" 8항목을 같은 번호로 점검한다:**
   1 requirements.txt의 패키지·버전 표기 + 새 가상환경 설치 · 2 제출 태그를 **GitHub에서 새로 clone**해 README 명령(`python denoise.py`)으로 실행 ·
   3 가중치·설정 포함, 95 MB 이하, Git LFS 포인터가 아닌 실제 파일 · 4 추론 코드에 개인 PC 절대 경로·다운로드 없음 ·
   5 `input/`의 모든 WAV → `output/` 바로 아래 같은 이름 하나씩 · 6 16 kHz·모노·48,000 표본, 입력과 같은 형식(6+ 재채점 = 저장 결과) ·
   7 GitHub 공개 접근, 태그 GitHub = 로컬, README의 팀명·회차·태그, Python·패키지 버전, 가중치 경로, 학습 설정·선택 근거·Validation 결과, 외부 코드·데이터·가중치의 출처·범위 ·
   8 이번 버전의 Validation·Train 결과(코드 지문 일치)와 AI 활용 내역서. 그 밖에 공개 범위(보안 검사)와 주 1회 제출 횟수.
6. 팀장이 런어스 → 질의응답 비밀글: 저장소 링크 + 태그 `submission-N`(커밋 해시) + 내역서 hwpx
7. `release.py submitted N`, 리더보드 공개 후 `release.py leaderboard N --pesq X`

## 현재 기록
- 배포 예제 MaskCNN(seed 42, 109 epoch 중 89 epoch 선택): validation 1.692 (노트북 float 채점).
- 이 저장소의 `train.py`로 CPU 재현(2026-11-21, `runs/20261121-175901_baseline_repro`, 9.8분): 109 epoch 모두 손실·PESQ 차이 0, 선택 가중치가 배포 가중치와 비트 단위로 같음(2026-11-25까지 `weights/submission.pt`였음).
  16비트 출력 채점: **validation 1.6922 / SNR 13.06 dB, train 1.8776 / 13.11 dB** (입력 1.395 / 1.470). train 개선 +0.41 대 validation +0.30.
  약점(스펙트로그램): 저역 잡음(바람·차량 실내)을 지울 때 300 Hz 아래 말소리 배음까지 지운다 — validation에서 가장 손해 본 3개 파일(traffic 15 dB −0.61, wind 15 dB −0.56, volvo 5 dB −0.45)이 모두 이 경우.
- MP-SENet(`se/arch/mpsenet.py`, 2.26M, 크기는 마스킹·위상은 매핑)을 처음부터 학습(2026-11-22, GCP L4 11시간,
  `configs/mpsenet.json` + 재혼합 50 %, 116 epoch 중 115 epoch 선택): **validation 3.132 / SNR 18.40 dB,
  train 3.303** (일반화 차 +0.094). validation 164개 전부 베이스라인보다 좋고 입력보다 나빠진 파일 0개.
  약점: 넓은 대역의 정상 잡음(hf_channel 2.35, broadband 2.45)과 0 dB(2.37).
  가중치는 `weights/candidates/mpsenet_remix50.pt`에 보관. (`weights/submission.pt`는 2026-11-25에 mpsenet_esc 100 epoch로 바뀜, 아래 참고.)
- 같은 설정에 말소리 속도 변환(0.9~1.1배)만 추가(`configs/mpsenet_speed.json`): validation 3.113 / train 3.284,
  일반화 차 +0.095. epoch 간 흔들림(±0.02) 안이라 **효과 없음**.
- MP-SENet 저자 공개 가중치(DNS Challenge 학습, `weights/pretrained/mpsenet_dns.pt`, 모델 설정 `source_axes: true` 필요)를
  그대로 채점(2026-11-23): **validation 3.176 / train 2.980**. 과제 데이터를 본 적 없는 모델이 train보다 validation에서
  0.27 더 개선한다(validation이 더 쉬운 묶음).
- 그 가중치를 과제 train으로 이어 학습(`configs/mpsenet_pretrained_dns.json`, lr 1e-4, L4 2.8시간, 29 epoch에서 조기 종료,
  29 epoch 선택): **validation 3.318 / SNR 19.82 dB, train 3.457** (일반화 차 +0.063). 현재 최고.
  가중치는 `weights/candidates/mpsenet_dns_ft.pt`. 공개 가중치 대비 validation +0.14, train +0.48: 오른 폭의 대부분이
  train의 화자·잡음에 맞춘 것이라, 새 잡음에서 이득인지는 아직 재지 않았다(새 잡음 검증셋 필요).
- 새 잡음 검증셋 첫 채점(2026-11-23, 입력 1.519): baseline 1.543(+0.02, 62개가 입력보다 나쁨), mpsenet_remix50 **2.827**,
  mpsenet_speed 2.888, DNS 공개 가중치 그대로 **3.336**, DNS 이어 학습 3.332. 과제 데이터만으로 학습한 모델은 새 잡음에서
  DNS 계열보다 0.5 낮다(validation에서는 0.04~0.19 차). 이어 학습은 새 잡음에서 이득 0(순간 잡음 −0.25, 0 dB 2.75 → 2.42):
  validation +0.14는 아는 잡음에 맞춘 몫이었다. DNS 계열 점수는 ESC-50과 DNS 잡음이 둘 다 Freesound 출처라 조금 유리할 수 있다.
- 외부 잡음을 섞어 처음부터 학습(`configs/mpsenet_esc.json`, 2026-11-24, L4 10.5시간, 103 epoch에서 사용자 지시로 중단,
  103 epoch 선택): 재혼합 항목의 절반(전체의 25 %)이 ESC-50 fold 1~4 잡음. **validation 3.043 / 새 잡음 3.055 / train 3.212**.
  mpsenet_remix50 대비 새 잡음 **+0.228**(95 % 구간 +0.16~+0.30, 109개 개선·46개 악화), validation **−0.089**(164개 중 135개 악화,
  모든 SNR·난이도에서 고르게). DNS 공개 가중치(새 잡음 3.336)까지는 아직 0.28. 학습 내내 validation과 새 잡음 곡선이 붙어 갔다
  (상관 0.97, validation 최고 epoch를 골라도 새 잡음 최고와 0.005 차). 가중치는 `weights/candidates/mpsenet_esc.pt`.
  주의: 학습 잡음과 검증 잡음이 같은 ESC-50의 다른 fold(같은 50종)라 새 잡음 이득이 실제보다 크게 보일 수 있다.
- 외부 음성 풀 준비(2026-11-25, 아직 학습에 쓰지 않음): Zeroth-Korean(OpenSLR 40, CC BY 4.0) 22,720개 전체를 VM에서 측정.
  말소리−바닥 잡음 간격의 화자별 중앙값이 45 dB 미만인 10명(174, 183, 189, 194, 197, 203, 204, 206, 209, 217)을 제외(사용자 결정),
  넘친 파일 340개 제외, 화자당 최대 60개 → **105명, 3초 구간 6,096개**(VM에만 있음, 558 MB). 재혼합 항목의 말소리를 이 풀로 바꾸고
  크기는 원래 항목의 clean에 맞춘다. 비율과 섞을 잡음은 사용자 결정 대기.
- Zeroth 음성을 섞어 처음부터 학습(`configs/mpsenet_zeroth.json`, 2026-11-25, L4 4시간, **39 epoch에서 사용자 지시로 조기 중단**,
  39 epoch 선택): 재혼합 항목의 절반(전체의 25 %)이 Zeroth 말소리. **validation 2.824 / 새 잡음 2.826 / train 2.922**.
  다 돌린 모델과 견줄 수 없다. 같은 시점의 mpsenet_esc와 비교(35~39 epoch 평균): validation 2.805 대 2.806, 새 잡음 2.791 대 2.809,
  train 2.821 대 2.780. **39 epoch까지는 효과가 보이지 않는다**(차이가 epoch 간 흔들림 안). 결론을 내려면 60 epoch 이상이 필요하다
  (`runs/…_mpsenet_zeroth/last.pt`가 VM과 로컬에 있어 이어 학습 가능, 단 run의 config.json에서 max_epochs·min_epochs를 39로 바꿔 둠).
  가중치는 `weights/candidates/mpsenet_zeroth.pt`. 사용자가 38 epoch에서 중단시켰고, 실행을 마무리하다 Claude의 실수로 1 epoch가 더 돌았다.
- **1회차 제출 준비(2026-11-25, 사용자 결정): `weights/submission.pt` = mpsenet_esc의 100 epoch** (`weights/candidates/mpsenet_esc_ep100.pt`).
  처음에는 과적합 징후(약 88 epoch부터 train PESQ만 오름) 전의 저장본인 75 epoch로 준비했다가(validation 2.969 / train 3.126 / 새 잡음 2.998,
  `mpsenet_esc_ep75.pt`), 87 epoch 저장본이 없음을 확인한 뒤 사용자가 100 epoch로 바꿨다(25 epoch마다 저장한 것 중 마지막).
  16비트 채점: **validation 3.023 / SNR 18.01 dB, train 3.197, 새 잡음 3.052**.
  `results/submissions/round1/` 기록, README 블록 갱신, 내역서 `docs/ai_usage/round1.json` 작성, 태그 `submission-1`(`b922a40`) push,
  저장소 공개, `check_submission.py --round 1` READY(14개 항목 PASS, 재채점 3.0225 = 저장값)까지 끝(2026-11-25).
  남은 것: `docs/references.json`의 `checked_by`(사람), 리더보드 공개 후 `release.py leaderboard 1 --pesq X`.
- **2회차 후보(pretrained 허용 후):** `weights/candidates/mpsenet_dns_ft.pt`(DNS 공개 가중치에서 이어 학습) validation 3.318 / 새 잡음 3.333 / train 3.457,
  공개 가중치 그대로 validation 3.176 / 새 잡음 3.336. 1회차(3.023 / 3.052)보다 validation +0.30, 새 잡음 +0.28.
- 팀원의 xLSTM-SENet(과제 train만, 재혼합 50 %, 57 epoch, 사전학습 없음)을 들여와 채점(2026-11-25): validation 2.849(팀원 값 2.8493 재현),
  새 잡음 **2.586**. 같은 조건의 MP-SENet(57 epoch 2.945)보다 낮고 추론은 약 6배 느리다. mpsenet_esc와 출력 평균(7:3)은
  75 epoch 기준 validation 3.006(+0.04) / 새 잡음 2.922(−0.08), 103 epoch 기준(8:2) 3.060(+0.02) / 2.998(−0.06): **새 잡음에서 손해라 제출에 쓰지 않는다**(사용자 결정).
  가중치 `weights/candidates/xlstm_ep57.pt`는 기록용(평균 파일은 `tools/make_ensemble.py`로 다시 만든다).
- 히든 테스트(1회차, mpsenet_esc 100 epoch): 조교가 테스트셋을 어렵게 고치기 **전 3.12, 고친 뒤 약 2.4**(사용자 전달 2026-11-26).
  수정 전 점수는 validation 3.02와 비슷하다. validation은 원래 테스트셋을 잘 예측했고, −0.7은 조교가 더한 어려움 몫이다.
  round1 기준 어려운 검증셋(1.96)이 validation보다 1.06 낮고 수정 후 히든은 0.72 낮다. 수정 후 히든은 그 사이, validation 쪽에서 약 2/3 지점이다(한 점뿐이라 대략).
- **어려운 검증셋**(`tools/make_hardset.py`, 2026-11-25): validation 말소리 164개마다 한 가지만 바꾼 4묶음 656개.
  lowsnr = 같은 쌍의 잡음을 −5/0 dB로, newlow = ESC-50 fold 5를 −5/0 dB로, babble = Zeroth 제외 화자 10명 중 3~6명의 동시 발화(원래 SNR),
  reverb = 합성 잔향(RT60 0.3~0.8 s) + 원래 잡음, 기준은 잔향 없는 clean. 입력 1.206. 결과(전체 / lowsnr / newlow / babble / reverb):
  round1 1.959 / 1.99 / 2.10 / 2.18 / 1.58, mpsenet_esc(103) 1.968, remix50 1.886, zeroth(39) 1.773,
  **DNS 공개 가중치 2.308 / 2.27 / 2.50 / 2.83 / 1.63, DNS 이어 학습 2.322 / 2.37 / 2.44 / 2.85 / 1.63**.
  DNS 계열이 잔향 말고 모든 묶음에서 round1보다 +0.29~+0.65(babble 최대, 164개 중 152개). 이어 학습 대 공개 가중치: lowsnr +0.10(과제 잡음),
  newlow −0.06(새 잡음, 95 % 구간 −0.12~−0.01), babble·reverb 차이 없음. **잔향은 모든 모델이 1.6 근처로 못 다룬다.**
  round1은 어느 묶음에서도 2.4보다 낮아 이 검증셋은 히든 테스트보다 어렵다(순위 비교용).
- **MossFormerGAN_SE_16K**(ClearerVoice, Apache-2.0, `weights/pretrained/mossformergan_se16k.pt`, 2026-11-26): 원본 추론과 출력 차 16비트 0.006단계 이하.
  그대로 채점 validation 3.135 / 새 잡음 3.273 / train 2.946 / 어려운 검증셋 2.314(lowsnr 2.20, newlow 2.47, babble 2.80, **reverb 1.79 = 전 모델 최고**).
- **출력 평균 섞기**(2026-11-26, 비율은 TF-Locoformer / DNS 이어 학습 / MossFormerGAN / MP-SENet DNS): 단독 최고 TF-Locoformer 3.297 / 3.424 / 2.463.
  0.7/0.3/0/0 = 3.437 / 3.532 / 2.514, **0.6/0.2/0.2/0 = 3.435 / 3.545 / 2.534**(reverb 1.69), 0.6/0.2/0/0.2 = 3.440 / 3.551 / 2.530.
  셋 이상 섞기는 둘보다 +0.01~0.02(흔들림 수준). TF-Locoformer 가중치(`weights/candidates/tflocoformer_v8.pt`, 팀원 v8 패키지에서 꺼냄)는 **MERL 공식 공개 가중치와 같다**(2026-11-26 확인):
  merlresearch/tf-locoformer(Apache-2.0)의 `egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth`(Git LFS, 59,972,116 B, sha256 6ffa518f…)를 받아 비교, 텐서 174개(14.97M 파라미터) 모두 비트 단위로 같음.
  학습 데이터는 DNS Challenge (Interspeech 2020). 파일 안 `source` 문구도 이 내용으로 고쳤다(파일 지문 e918dc511e4e → 890edb443399, 채점 기록의 지문도 바꿈).
- **형식 점검**(`tools/format_check.py`, `--n 0`은 VM에서): 음량(30 dB 작게, 최대), 직류 성분, 12·30초 이어 붙이기는 영향 없음(TF-Locoformer만 30초 −0.05).
  **앞뒤 무음은 크게 손해**: MP-SENet DNS 0.25초 −0.34, 1초 −0.54, 3초 −1.01, TF-Locoformer 1초 −0.34, MossFormerGAN 1초 −0.09. −70 dBFS 잡음을 붙여도 비슷.
  그 파일의 실제 잡음을 앞뒤에 붙이면 영향 없음. 과제 noisy 1,460개에는 조용한 시작 구간이 하나도 없다(가장 조용한 20 ms 중앙값 −37 dB).
  테스트 파일도 같은 방식으로 만들었다면 이 문제는 히든 점수와 무관하다. **대비책을 넣었다**(사용자 결정 2026-11-26, `se/inference.py` `quiet_edges`):
  앞뒤로 **그 파일의 큰 소리(프레임 크기 90 % 지점)보다 40 dB 넘게 조용한** 20 ms 프레임이 0.1초 이상 이어지면 그 부분은 모델에 넣지 않고 0으로 쓴다.
  처음에는 절대 기준(−55 dBFS)이었는데 VM 확인에서 무음·−70 dBFS 잡음은 완전히 회복(세 모델 모두 base와 같음)했지만 30 dB 작게 녹음한 파일의 진짜 앞뒤를 잘라 −0.15~−0.18 손해라 상대 기준으로 바꿨다.
  과제 파일의 앞뒤는 큰 소리보다 최대 25 dB 아래라 1,460개 중 걸리는 파일 0개(30 dB 작게 해도 0개, 출력 그대로).
- **저역 잡음 검증셋**(`tools/make_lowband.py`, `data/lowband/`, 2026-11-26): validation 말소리 164개 × 3묶음. esc_low = ESC-50 fold 5 중 300 Hz 아래 에너지 50 % 이상인 95개(헬리콥터·천둥·세탁기·엔진·바람 등)를 원래 SNR로,
  synth_low = 인공 저역 잡음(브라운·저역 통과·엔진 웅웅거림)을 원래 SNR로, esc_low_snr = esc_low를 −5/0 dB로. 입력 1.494.
  결과(전체 / esc_low / synth_low / esc_low_snr): TF-Locoformer 3.131 / 3.395 / 3.476 / 2.522, DNS 이어 학습 2.975 / 3.178 / 3.506 / 2.242, MossFormerGAN 2.932 / 3.208 / 3.333 / 2.253, round1 2.670 / 2.939 / 3.204 / 1.868,
  **섞기 0.6/0.2/0.2 = 3.207 / 3.459 / 3.624 / 2.538**, 0.7/0.3 = 3.197 / 3.439 / 3.633 / 2.519.
  **처음 듣는 저역 잡음 자체는 약점이 아니다**(같은 SNR에서 validation보다 높다. PESQ-WB는 100 Hz 아래를 거의 보지 않는다). 약점은 낮은 SNR: −5 dB 2.27, 0 dB 2.80(섞기 기준).
- **잔향 묶음 점검**(2026-11-26, 로컬, 모델 없이): `room_response`는 직접음이 0번 표본이라 지연이 없다(같은 RT60로 다시 만든 164개에서 상호상관 최대 위치 0이 155개, 나머지 9개는 ±4표본 이내 = 무작위 꼬리 탓).
  **잡음 없이 잔향만 있는 말소리를 마른 clean과 비교한 PESQ = 평균 1.687**(RT60 0.30~0.45 s 1.93, 0.45~0.60 s 1.75, 0.60~0.80 s 1.47; 직접음 대 잔향 비 +4.9~+2.4 dB), 잡음까지 더한 입력 1.226.
  즉 모델들의 1.58~1.79는 "잡음은 다 지웠지만 잔향은 못 지운" 천장과 같다. 낮은 점수는 지연이 아니라 잔향 제거 능력(과 마른 clean 기준) 몫이고, MossFormerGAN(1.79)만 천장을 조금 넘는다. 검증셋은 그대로 둔다(사용자 결정).
- **무음 처리 보완 시도 → 되돌림**(2026-11-26): 기준(프레임 크기 90 % 지점 − 40 dB)은 백분위라 파일의 10 %보다 짧은 충격음에 움직이지 않는다(테스트 추가, 이것만 남김).
  잘린 쪽에 조용한 구간 40 ms를 남겨 모델에 넣고 출력에 10 ms 페이드를 넣어 봤으나 **손해라 뺐다**. VM `format_check`(164개, 앞뒤 무음 1초 / −70 dBFS 잡음 1초):
  여유 40 ms가 있으면 MP-SENet DNS −0.22 / −0.21, TF-Locoformer −0.39 / −0.31, SEMamba −0.32 / −0.36, MossFormerGAN −0.07 / −0.04 (여유 없으면 모두 0.00). **입력 앞에 무음이 40 ms만 있어도 이 모델들은 크게 손해 본다.**
  페이드만 남긴 경우도 −0.02(MP-SENet, 로컬 12개): 자르는 자리가 첫 큰 프레임이라 페이드가 실제 신호를 깎는다. `se/inference.py`는 원래 자르기 그대로다.
  같은 측정에서 상대 기준은 확인됐다: 30 dB 작게 녹음한 파일이 +0.04~+0.06(절대 기준 때 −0.15~−0.18).
- **2회차 이후 방향**(사용자 결정 2026-11-26): DNS 계열 사전학습 가중치 기반, **PCS(Perceptual Contrast Stretching) 적용·검증**(후처리 기법 결정), Zeroth 음성을 섞은 낮은 SNR 이어 학습(+MetricGAN, EMA, WiSE-FT). VM의 Zeroth 원본 파일(10 GB)은 지우지 않는다.
- **SEMamba**(2026-11-26, `se/arch/semamba/`, 모델 이름 `semamba`, `weights/pretrained/semamba_vd.pt` 8.70 MB): RoyChao19477/SEMamba(MIT, 커밋 `750ff5d`)의 `ckpts/vd.pth`
  (9,142,810 B, sha256 e0c2db3f…, **VCTK-Demand + DNS-2020 학습**, 순수 DNS 가중치는 없음. 가중치 라이선스는 따로 적혀 있지 않다)를 값 그대로 변환. STFT 400/100/400, TFMamba 4개, 채널 64.
  `mamba-ssm`은 CUDA·nvcc가 있어야 설치되어(우리 VM에도 nvcc가 없어 빌드 실패) 제출에 쓸 수 없다 → **Mamba를 순수 PyTorch로 새로 썼다**(`mamba_torch.py`, 가중치 이름·모양 동일).
  대조(VM, validation 164개): 원 저장소 코드 + 번들 `mamba_ssm` 1.2.2 소스(참조 스캔 `selective_scan_ref`, RMSNorm은 triton 커널)와 **PESQ 3.0850 대 3.0850**(파일별 최대 차 0.0013), 파형 차 16비트 평균 2단계·최대 10단계. CUDA 커널 자체와는 대조하지 못했다.
  CPU 속도(로컬 8스레드, 3초 파일): `torch.jit.script` 순차 루프 3.9초, Python 루프 4.9초, 병렬 접두 스캔은 스캔만 약 12배 느림(연산량 log₂L배, 캐시 초과) → 순차 루프 채택. GPU 0.5초.
  **그대로 채점: validation 3.085 / 새 잡음 3.233 / train 2.966 / 어려운 2.241(lowsnr 2.25, newlow 2.52, babble 2.68, reverb 1.52) / 저역 2.791(esc_low 3.11, synth_low 2.95, esc_low_snr 2.31).**
  DNS 계열 넷 중 validation·새 잡음·저역에서 가장 낮다(MP-SENet DNS 3.176 / 3.336, MossFormerGAN 3.135 / 3.273, TF-Locoformer 3.297 / 3.424). newlow만 MP-SENet DNS(2.50)·MossFormerGAN(2.47)과 같은 수준. 인공 저역 잡음(synth_low 2.95)이 특히 약하다(다른 모델 3.33~3.51). 섞기에서의 가치는 아직 재지 않았다(3단계).
  `ckpts/pretrained_discriminator.pth`(747,574 B, 텐서 34개)는 `se/arch/metric_discriminator.py`의 `MetricDiscriminator`에 strict로 그대로 읽힌다(스크래치패드에만 있음, 학습 데이터 미표기).
  SEMamba++(Eddieyjlee/SEMambapp, Interspeech 2026)는 VCTK + LibriTTS로 학습한 음성 복원 모델이라 DNS 계열이 아니다(보류).
- **VM Zeroth 원본 확인**(2026-11-26): `data/external/zeroth/full/`에 flac 22,720개(압축 파일 안의 개수와 같음)·화자 115명, 무작위 300개 모두 열림. 압축 파일(9.7 GB)과 학습 풀(559 MB)도 그대로. 디스크 여유 38 GB.
- **3단계: 정렬·음량·섞기 연구**(2026-11-26~27, `tools/mix_study.py`, 표 전체는 `results/mix_study/report.md`, 파일별 점수는 `runs/mix_study/scores/`). 멤버 L = TF-Locoformer, M = MP-SENet DNS, F = MossFormerGAN, S = SEMamba, 모두 공개 가중치 그대로.
  - **지연·극성 보정은 필요 없다**: 4종 검증셋 1,476개에서 네 모델 모두 clean과 지연 0·극성 같음(어려운 검증셋만 3 %가 ±1표본, 잔향·동시 발화 묶음), 모델끼리는 99.8~100 % 지연 0.
  - **TF-Locoformer는 출력이 약 3.5 dB 작다**(clean에 맞추는 배율의 중앙값 1.48, train 1~99 % 구간 1.43~1.67). 비교표의 Val SNR 9.12 dB는 이 때문이고 PESQ는 음량을 보지 않아 그대로였다.
    보정(SNR validation / 새 잡음 / 어려운 / 저역): 없음 9.12 / 9.19 / 7.56 / 8.64, 고정 1.484배 20.16 / 21.41 / 13.02 / 17.76, **파일별 자동 배율(g = <noisy, y> / <y, y>, [1.35, 1.65]로 제한, 출력이 무음이면 1.484) 20.66 / 22.07 / 13.09 / 17.98**, clean 기준(상한) 20.72 / 22.11 / 13.58 / 18.06. PESQ 변화 0.0001 이하.
    자동 배율은 SNR이 낮을수록 위로 치우친다(−5 dB 중앙값 +5 %, 어려운 검증셋 최대 2.38배) → 제한이 필요하다. 클리핑: 고정 배율은 어려운 검증셋 3개(0.46 %), 자동 배율 1개(0.15 %, 피크 1.003), 나머지 검증셋 0개(최대 피크 0.89).
    **아직 `se/arch/tflocoformer`에 넣지 않았다**(연구 도구에서만 적용, 넣을지는 사용자 결정). 이전 섞기 "0.6/0.2/0.2"는 L이 0.68배로 들어가 실제로는 약 0.5/0.25/0.25였다.
  - 저역 검증셋 빈칸: MP-SENet DNS 공개 가중치 2.940(esc_low 3.22, synth_low 3.32, esc_low_snr 2.28).
  - **조합(L 음량 보정 후 파형 평균, 0.1 단위 87개 조합, 비율은 validation + 새 잡음으로 선택; PESQ validation / 새 잡음 / 어려운 / 저역)**: L 단독 3.297 / 3.424 / 2.463 / 3.131,
    L0.5+M0.5 3.383 / 3.506 / 2.515 / 3.172, L0.6+F0.4 3.360 / 3.495 / 2.522 / 3.182, L0.6+S0.4 3.350 / 3.482 / 2.498 / 3.104,
    **L0.5+F0.2+M0.3 3.394 / 3.528 / 2.537 / 3.199**(SNR 20.14 / 21.91 / 12.84 / 17.66), L0.4+F0.2+M0.3+S0.1 3.394 / 3.530 / 2.535 / 3.184.
    L+M 대비 F 추가: +0.011 / +0.021 / +0.021 / +0.027(새 잡음·어려운·저역에서 95 % 구간이 0 위). F+L+M 대비 S 추가: +0.000 / +0.003 / −0.002 / **−0.016**(저역 구간 −0.023~−0.008) → SEMamba는 이득이 없고 저역에서 손해.
    섞으면 SNR은 L 단독보다 0.2~0.5 dB 낮다(L의 SNR이 가장 높아서). 음량 보정 전후로 같은 실효 비율이면 PESQ는 같다(보정 없는 0.6/0.2/0.2 = 3.392 / 3.528 / 2.537 / 3.200).
  - **결합 방식: STFT(400/100) 크기 가중 평균 + L의 위상이 파형 평균보다 일관되게 낫다**(같은 비율). L0.5+F0.2+M0.3: **3.407 / 3.537 / 2.558 / 3.220**, 차이 +0.014 / +0.009 / +0.021(구간 +0.017~+0.025) / +0.021(+0.013~+0.029), SNR도 +0.01~+0.17 dB.
    네 조합 모두 같은 방향(+0.009~+0.026). 0 dB·−5 dB에서 이득이 크다(어려운 −5 dB 2.356 → 2.390). 크기 평균용 비율은 따로 찾지 않았다. 4개 조합 크기 평균은 3.413 / 3.544 / 2.561 / 3.210.
    잔향 묶음은 섞어도 1.68~1.70(천장 1.69 근처), L 단독 1.57.
- 학습 곡선의 모양: 60 epoch쯤 3.0, 80 epoch 이후 시간당 +0.03 이하, 막바지에 train만 오르고 validation이 멈춤.
  중단 조건은 `min_delta 0.01`, `patience 25`, `min_epochs 40`, `max_epochs 150`(사용자와 합의), 끝나면 VM 정지.

## 작업 방식 (사용자 요청, 2026-11-22)
- 실험은 한 번에 하나만 바꾼다.
- 로컬 CPU에서는 pytest와 파일 몇 개짜리 확인만. 학습과 전체 채점은 GPU VM에서.
- **모델·섞기 조합의 판단 기준**(사용자 결정 2026-11-26): **GPU 기준 성능(PESQ, 보조 SNR)과 가중치 95 MB 한도만 본다.** CPU 추론 시간(예: SEMamba 3초 파일당 3.9초)은 제약이 아니다(채점 환경은 GPU로 본다). 득실을 따질 때 CPU 속도를 단점으로 들지 않는다.
  단, **설치에 CUDA 커널 컴파일(`nvcc`)이 필요한 패키지(`mamba-ssm`, `causal-conv1d` 등)는 계속 쓰지 않는다**(`pip install -r requirements.txt`가 깨질 위험). 그런 모델은 순수 PyTorch 구현으로만 들인다. 코드는 CPU에서도 돌아야 한다(README 약속, 느려도 됨).
- 일반화가 최우선이다(히든 테스트는 1차보다 어렵다). validation 0.0x 차이보다 과적합 여부를 본다.
- **큰 결정은 사용자가 내린다** (사용자 요청, 2026-11-24). 학습 시작, 비율·설정 변경, 새 기법·데이터 도입 전에는 "시작할까요?"만 묻지 말고, 사용자가 판단할 핵심 쟁점과 대안 1~2개(각각의 득실)와 Claude의 권고를 먼저 보여 주고 결정을 기다린다. 사용자가 질문하면 코드를 고치기 전에 왜 그렇게 돌아가는지부터 쉬운 말로 설명한다.
- GPU VM은 Python 3.10이다. 학습을 시작하기 전에 VM에서 pytest와 `py_compile`을 돌린다.
