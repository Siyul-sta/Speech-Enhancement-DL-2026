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
