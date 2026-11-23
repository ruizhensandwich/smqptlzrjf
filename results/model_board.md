### 모델 비교표 (Validation 164개 · 16비트 출력 기준, **굵게** = 가장 좋은 값)

| # | 태그 | 모델 | 파라미터 | Val PESQ | ΔPESQ | Val SNR | 0 dB | 15 dB | hard | 나빠진 파일 | Train PESQ | 일반화 차 | 학습 | 추론 속도 | 날짜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline_repro | mask_cnn | 111k | 1.692 | +0.298 | 13.06 | 1.297 | 1.957 | 1.674 | 8.5% | 1.878 | +0.110 | 109ep/89 · 9.8분 · cpu | **0.001** (cpu) | 2026-11-21 |
| 2 | mpsenet_remix50 | mpsenet | 2263k | 3.132 | +1.738 | 18.40 | 2.369 | 3.503 | 3.417 | **0.0%** | 3.303 | +0.094 | 116ep/115 · 657.6분 · cuda | 0.031 (cuda) | 2026-11-22 |
| 3 | mpsenet_speed | mpsenet | 2263k | 3.113 | +1.718 | 18.15 | 2.365 | 3.506 | 3.384 | **0.0%** | 3.284 | +0.095 | 115ep/111 · 661.0분 · cuda | 0.031 (cuda) | 2026-11-22 |
| 4 | mpsenet_dns_ft | mpsenet | 2263k | **3.318** | **+1.924** | **19.82** | **2.677** | **3.673** | **3.653** | **0.0%** | **3.457** | +0.063 | 29ep/29 · 165.7분 · cuda | 0.031 (cuda) | 2026-11-23 |
| 5 | mpsenet_dns_pretrained | mpsenet | 2263k | 3.176 | +1.781 | 18.25 | 2.498 | 3.531 | 3.600 | 0.6% | 2.980 | **-0.272** | - | 2.541 (cpu) | 2026-11-23 |

| # | 태그 | 장점 | 단점 | 메모 |
|---|---|---|---|---|
| 1 | baseline_repro | 가장 가벼움 (111,361 파라미터)<br>가장 잘 줄이는 잡음: alarm_siren (+0.76) | 최고 모델보다 -1.626<br>입력보다 나빠진 파일 8.5%<br>과적합 경향: train 개선이 validation보다 +0.110 큼<br>가장 약한 잡음: vehicle_interior (-0.07)<br>0 dB 개선이 작음 (+0.218) | 배포 노트북 예제 재현 (CPU, seed 42) |
| 2 | mpsenet_remix50 | 입력보다 나빠지는 파일이 거의 없음 (0.0%)<br>가장 잘 줄이는 잡음: bell (+2.69) | 최고 모델보다 -0.186<br>과적합 경향: train 개선이 validation보다 +0.094 큼<br>가장 약한 잡음: vehicle_interior (+1.02) | MP-SENet 처음부터 학습 (원본 설정 + 재혼합 50 %, L4 11시간, 116 epoch에서 수동 중단). 같은 가중치의 CPU 채점도 3.1323 |
| 3 | mpsenet_speed | 입력보다 나빠지는 파일이 거의 없음 (0.0%)<br>가장 잘 줄이는 잡음: bell (+2.68) | 최고 모델보다 -0.205<br>과적합 경향: train 개선이 validation보다 +0.095 큼<br>가장 약한 잡음: vehicle_interior (+1.02) | One change against mpsenet_remix50: the speech of remixed items is played at 0.9 / 0.95 / 1 / 1.05 / 1.1 speed (remix stays 0.5, so 40 % of all items get a changed voice) | stopped by hand after epoch 115: validation PESQ flat for 2 h while train PESQ kept rising |
| 4 | mpsenet_dns_ft | 검증 PESQ 최고 (3.318)<br>0 dB에서 가장 강함 (+1.597)<br>어려운 잡음(hard)에서 가장 강함 (3.653)<br>입력보다 나빠지는 파일이 거의 없음 (0.0%)<br>가장 잘 줄이는 잡음: clap (+2.86) | 가장 약한 잡음: vehicle_interior (+0.99) | Fine-tune of the MP-SENet authors' DNS checkpoint (yxlu-0102/MP-SENet best_ckpt/g_best_dns, MIT) on the course train set. Against mpsenet_remix50: start weights (and the source's axis order they need), lr 1e-4 instead of 5e-4, 40 epochs at most. Data settings are the same (remix 0.5). The discriminator starts from random weights (none is published). |
| 5 | mpsenet_dns_pretrained | 입력보다 나빠지는 파일이 거의 없음 (0.6%)<br>일반화 차가 가장 작음 (-0.272)<br>가장 잘 줄이는 잡음: clap (+2.67) | 최고 모델보다 -0.142<br>CPU 추론이 느림 (실시간의 2.54배 시간)<br>가장 약한 잡음: vehicle_interior (+0.67) | MP-SENet 저자 공개 가중치(DNS Challenge 학습, yxlu-0102/MP-SENet best_ckpt/g_best_dns, MIT)를 그대로 채점. 이 과제 데이터로는 전혀 학습하지 않음 |

ΔPESQ = 입력(noisy) 대비 개선 · 0 dB/15 dB/hard = 해당 조건의 Val PESQ · 나빠진 파일 = 입력보다 PESQ가 낮아진 validation 파일 비율 · 일반화 차 = train 개선 − validation 개선 (클수록 학습 화자에 치우침) · 학습 = 실행 epoch/선택 epoch · 추론 속도 = 추론 시간 ÷ 음성 길이 (낮을수록 빠름) · 범주 장단점은 train 1,296개 기준
