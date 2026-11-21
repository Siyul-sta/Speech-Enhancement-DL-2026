# 2026 신호 및 시스템 2차 챌린지 — Deep Learning Speech Enhancement

PyTorch 기반 **3층 MaskCNN 베이스라인**으로 잡음이 섞인 음성을 복원하는 프로젝트입니다. 교수님 배포 노트북으로 학습한 모델과 2026-11-20 Validation 평가 결과를 정리했습니다.

## 데이터와 제출 규격

- Train: **1,296개** noisy/clean 쌍
- Validation: **164개** noisy/clean 쌍
- 입력: 16 kHz, 모노 WAV (배포 데이터는 3초)
- 추론: **noisy WAV만 사용**하며 clean 음성이나 잡음 참조 파일을 필요로 하지 않습니다.
- 출력: `output/` 바로 아래에 입력과 **같은 파일명**, 16 kHz/모노, **입력과 같은 표본 수**로 저장합니다.
- 원본 음성, Train/Validation 데이터셋, 개인 식별 정보가 포함된 녹음은 저장소에 포함하지 않았습니다.

## 실제 Validation 결과 (전체 164개 평균)

| 지표 | Noisy | MaskCNN | 개선량 |
| --- | ---: | ---: | ---: |
| PESQ-WB | 1.3947 | **1.6961** | **+0.3014** |
| SNR (dB) | 9.6341 | **13.0847** | **+3.4506** |

- 선택 가중치: `checkpoints/best_mask_cnn.pt`
- 최고 epoch: **54**
- 총 74 epoch 학습 후 early stopping
- 평가 기록: `results/validation_summary.csv`, `results/validation_per_file.csv`
- 학습 기록: `results/training_history.csv`, `results/training_summary.json`
- **위 점수는 공개 Validation 결과이며 비공개 Test 점수가 아닙니다.**

## 폴더 구조

```text
Speech-Enhancement-DL-2026/
├─ notebooks/
│  └─ deep_learning_noise_reduction.ipynb
├─ src/
│  └─ inference.py
├─ checkpoints/
│  └─ best_mask_cnn.pt
├─ results/
│  ├─ training_history.csv
│  ├─ training_summary.json
│  ├─ validation_summary.csv
│  └─ validation_per_file.csv
├─ requirements.txt
└─ README.md
```

## 실행 환경

- Python 3 (Google Colab GPU 또는 일반 PC의 CPU/GPU)
- 학습 기록: PyTorch `2.14.1+cu130`, 장치 `cuda`
- 추론은 CPU에서도 동작하도록 작성되어 있습니다.
- 필요한 패키지는 다음 명령으로 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 제출용 추론 (추가 다운로드 불필요)

저장소 루트에 `input/` 폴더를 만들고 테스트용 WAV 파일을 넣은 뒤 다음 명령을 실행합니다.

```bash
python src/inference.py --input input --output output --checkpoint checkpoints/best_mask_cnn.pt
```

파일마다 복원 WAV가 `output/입력파일명.wav`에 저장됩니다. 기존 `output/`에 입력과 무관한 다른 WAV가 있으면 오류를 반환하므로 먼저 정리해야 합니다. 실행 중에는 가중치나 다른 자료를 추가로 다운로드하지 않습니다.

## Colab에서 학습 및 평가 재현

1. Google Drive의 `MyDrive/deep_learning_noise_reduction_challenge/` 아래에 배포된 `train/`, `validation/`, `figures/` 폴더를 넣습니다. 각 데이터 분할에 `samples.csv`, `clean/`, `noisy/`가 필요합니다.
2. `notebooks/deep_learning_noise_reduction.ipynb`를 Google Colab에서 엽니다.
3. 노트북의 1–3절을 실행해 Drive와 데이터를 불러오고 CNN 모델을 정의합니다.
4. **4.1절 → 4.2절** 순서로 학습합니다. 가장 높은 전체 Validation PESQ-WB를 기록한 가중치가 Drive의 `outputs/best_mask_cnn.pt`에 저장됩니다.
5. 4.3–6절에서 epoch 비교, Wiener 비교, 전체 Validation 성능 평가를 진행합니다.

**학습 설정:** seed 42, batch size 16, Adam 초기 LR 0.001, `N_FFT=512`, `HOP=128`, CNN 중간 채널 `(64, 64)`, 커널 `(5, 3, 1)`, 최대 200 epoch, 최소 30 epoch, patience 20, `min_delta=0.001`. 예제 epoch 비교: 10·20·40.

본 저장소의 노트북은 교수님 배포 노트북 및 실행한 학습 설정을 기반으로 하며, GitHub 업로드를 위해 저장된 셀 출력(음성·이미지 포함)을 제거했습니다. 학습/검증 데이터 파일은 제공하지 않으므로 원본 데이터셋은 별도로 준비해야 합니다.

## 검증 현황 및 주의사항

- 저장된 checkpoint가 `MaskCNN` 구조에 정상적으로 로드되는지 확인했습니다.
- 임의로 생성한 16 kHz/모노 3초 WAV를 추론해 입력과 동일한 48,000표본을 출력하는 테스트를 통과했습니다.
- **비공개 채점 환경의 전체 파일에 대한 실행은 아직 검증하지 않았습니다.**
- 챌린지 제출 시 **GitHub 링크 외에 회차별 AI 활용 내역서**도 별도로 제출해야 합니다.

## Kaggle 무료 GPU 실험 자동화 (준비 완료, 실제 학습 미실행)

- [Kaggle CLI 설정·실행 안내](docs/KAGGLE_SETUP.md)
- [U-Net 학습 스크립트](src/kaggle_train.py)
- [GPU 실험 큐·자동 결과 수집](tools/kaggle_loop.py)
- [Codex 실험 규칙](AGENTS.md)

기본 CNN PESQ-WB 1.6961은 실제 검증 점수입니다. E02/E03 U-Net은 **아직 Kaggle에서 실행·검증되지 않은 실험 코드**입니다. 수업 음성 데이터의 제3자 Kaggle 업로드 허용 여부를 확인한 뒤 실행하세요.

## Codex Ralph Loop (로컬 PC 유지)

- [반복 실험 컨트롤러](tools/ralph_loop.py)
- [실행 및 한도 설정 안내](docs/RALPH_LOOP.md)

기존 E02/E03 연속 실험이 완료된 후 별도로 실행해야 합니다. 새 코드는 아직 Codex→Kaggle 통합 실험이 검증되지 않았으며, 무료 GPU/AI 호출 한도와 Private Kaggle 데이터 권한을 준수해야 합니다.

## 밤새 자동 개선 (로컬 PC 8시간)

- [8시간 Ralph Loop 설정 및 안전한 대기 실행](docs/OVERNIGHT.md)
- [기존 실험 종료 후 이어서 실행하는 PowerShell 스크립트](tools/start_overnight.ps1)

현재 3회 실험이 실행 중이면 **다른 PowerShell 창**에서 준비 스크립트를 실행하세요. 같은 소스 파일을 동시에 수정하지 않도록 기존 Python 프로세스 종료를 기다립니다. 사용자 설정 목표 PESQ 3.0에 도달해도 중단하지 않으며, 8시간 시간 예산과 무료 GPU 사용 한도를 지킵니다.

## GPU 한도까지 지속 개선 (8시간 종료 없이)

- [시간 제한 없는 로컬 Ralph Loop 스크립트](tools/start_until_quota.ps1)
- 기존 8시간 `start_overnight.ps1` 실행 중이면 **8시간 대기 창만 Ctrl+C로 중단**하고, 기존 3회 학습은 유지한 채 새 PowerShell 창에서 `& .\\tools\\start_until_quota.ps1`를 실행합니다.
- Kaggle 무료 GPU 잔여 시간이 2시간 아래로 떨어지거나 안전 문제가 생기면 신규 학습을 중단합니다. 3.0점 도달이나 8시간 경과는 중단 사유가 아닙니다.

## V2: 복소수 마스크 + 실험 이력 기반 전략 전환

기존 Ralph Loop를 안전하게 중단한 다음 [V2 전환 실행 가이드](docs/V2_STRATEGY.md)를 따르세요.

- [V2 복소수 마스크 U-Net 학습 코드](src/kaggle_train_v2.py)
- [V2 Kaggle GPU 순차 실험 제어기](tools/kaggle_v2.py)
- [V2 실험 164개 파일별 PESQ 비교](tools/compare_v2.py)
- [V2 오프라인 테스트](tests/test_kaggle_v2.py)

V1 학습 및 체크포인트는 유지되며, V2는 별도 비공개 Kaggle 작업과 결과 폴더를 사용합니다. V2는 아직 Kaggle 실행 검증 전이므로 2 Epoch 스모크를 먼저 수행해야 합니다.

## Adaptive Ralph Loop V2: 성능 정체 시 자동 모델 전환

이전 V1 자동화와 다르게, **검증 PESQ가 연속 2회 개선되지 않으면** 모델 계열을 자동으로 전환합니다.

- 새로운 `ResidualPhaseUNet` → 기존 V1 진폭 마스크 → V2 복소수 마스크 순환
- 계열별 우승 소스/체크포인트 보존, 이전 실험의 성공·실패 및 파일별 PESQ 퇴행 반영
- Kaggle 비공개 데이터·전체 164개 Validation 평가 유지, GPU 잔여 2시간 보호
- 무제한 시간 실행 가능하지만 GPU 사용량/반복 오류/안전 가드에 도달하면 중단
- 이전 `start_until_quota.ps1`은 **자동 모델 전환을 지원하지 않습니다.** 새로운 제어기를 사용해야 합니다.

**실행 가이드:** [Ralph Loop V2](docs/RALPH_V2.md)

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
# 기존 로컬 Ralph 및 Kaggle 원격 작업이 모두 종료된 것 확인 후:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob
```
