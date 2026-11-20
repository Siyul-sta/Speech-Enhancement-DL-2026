# Kaggle 무료 GPU 자동 실험 설정

이 문서는 2026 2차 챌린지의 **Kaggle GPU 배치 실험**을 위한 실행 안내입니다.
Kaggle 계정 연결/실제 GPU 학습은 아직 진행하지 않았습니다.
현재 코드는 첫 U-Net(E02)과 Multi-Resolution STFT Loss(E03)의 **학습 큐**만 자동 실행합니다.
Codex가 새 아키텍처를 스스로 만들고 수정하는 단계는 별도로 확장해야 합니다.

## 0. 데이터 반출 허가 먼저 확인

배포 데이터는 학생 음성(개인 식별 가능성 있음)을 포함할 수 있습니다.
**수업 담당자가 Kaggle이라는 제3자 플랫폼으로 업로드하는 것을 허용했는지 확인하기 전에는 업로드하지 마세요.**
Kaggle의 private dataset도 Kaggle 서비스로 원본 음성이 이전되는 것이며, private 설정이 업로드 허가를 대신하지 않습니다.
허용되지 않는다면 기존 Google Drive + Colab 경로로 학습을 계속 진행합니다.

## 1. 로컬 Windows 준비

PowerShell:

    git clone https://github.com/Siyul-sta/Speech-Enhancement-DL-2026.git
    cd Speech-Enhancement-DL-2026
    py -m pip install --upgrade kaggle
    kaggle auth login
    kaggle quota

- Kaggle 사용자명은 **GitHub의 Siyul-sta와 다를 수 있습니다**.
- 브라우저에서 로그인하거나 Kaggle API 토큰을 발급받되, 토큰/비밀번호는 GitHub 또는 채팅에 올리지 마세요.
- Kaggle 계정은 전화번호 인증이나 GPU 사용 자격 확인이 필요할 수 있습니다.

## 2. 데이터셋 준비 (허가를 받은 경우에만)

Google Drive의 기존 폴더:

    deep_learning_noise_reduction_challenge/
      train/samples.csv
      train/noisy/*.wav
      train/clean/*.wav
      validation/samples.csv
      validation/noisy/*.wav
      validation/clean/*.wav

그 중 `train/`과 `validation/`만 하나의 `speech-data.zip`으로 압축합니다.
ZIP 최상위에는 `train/`과 `validation/`이 존재해야 합니다.
`.waveform_cache/`와 `outputs/`는 넣지 마세요.

Kaggle의 Datasets → New Dataset에서 `speech-data.zip`을 업로드하고 **Private**으로 설정합니다.
생성된 데이터셋 식별자를 확인하세요: `KAGGLE_USERNAME/DATASET_SLUG`.
데이터셋 이름과 Kaggle 사용자명은 실제 계정에서 확인해야 하며 가상의 값으로 실행하면 실패합니다.
URL 예시: `https://www.kaggle.com/datasets/<USERNAME>/<SLUG>`.

## 3. GPU 전 준비: Dry Run

먼저 설정 파일을 생성하고 출력만 확인합니다. **이 명령은 GPU를 사용하지 않습니다.**

    py tools/kaggle_loop.py --username YOUR_KAGGLE_USERNAME --dataset YOUR_KAGGLE_USERNAME/YOUR_PRIVATE_DATASET --smoke

준비된 폴더:
- `kaggle_run/e02_unet/run.py`: Kaggle에서 실행할 단독 학습 스크립트
- `kaggle_run/e02_unet/kernel-metadata.json`: T4 GPU, private kernel, private dataset 연결, 인터넷 활성화 (pesq 설치용)

데이터셋을 Kaggle에 실제 업로드하기 전까지는 **절대 `--execute`를 추가하지 마세요.**

## 4. 2 Epoch 빠른 검증

    py tools/kaggle_loop.py --username YOUR_KAGGLE_USERNAME --dataset YOUR_KAGGLE_USERNAME/YOUR_PRIVATE_DATASET --smoke --execute

처리 순서: Kaggle에 GPU 작업 제출 → 종료 상태 폴링 → 결과 다운로드 → Validation 파일 164개 점검.
문제가 생기면 자동으로 다음 실험을 시작하지 않습니다.

주의: 2 Epoch은 GPU/데이터/코드 연동 테스트만 위한 것이므로, 본 학습 74 Epoch CNN의 PESQ 1.6961과 직접 비교해 모델 우열을 판단하지 않습니다.

## 5. 두 가지 실험 배치 실행

    py tools/kaggle_loop.py --username YOUR_KAGGLE_USERNAME --dataset YOUR_KAGGLE_USERNAME/YOUR_PRIVATE_DATASET --experiments E02_UNET E03_MRSTFT --epochs 25 --max-total-gpu-minutes 180 --execute

- E02: 시간·주파수 영역의 소형 U-Net + waveform/STFT 손실함수
- E03: 같은 U-Net, 256/512/1024 포인트 Multi-Resolution STFT 손실함수
- 모델 선택: **Validation 164개 전체 PESQ-WB 평균**
- Seed: 42, batch size: 4, patience: 7
- 기준 CNN PESQ-WB: 1.6961209534
- 결과는 `runs/e02_unet/`, `runs/e03_mrstft/`에 각각 저장됩니다.
- `runs/experiments.jsonl`에 실험 결과가 누적됩니다.

실행 횟수, 실행당 시간, Kaggle 주간 GPU 사용량에는 제한이 있습니다.
`--max-total-gpu-minutes`는 CLI의 실험 제출부터 결과 다운로드까지 경과 시간을 기준으로 확인하는 **안전 장치**이지 Kaggle의 정확한 GPU 청구/쿼터 측정치는 아닙니다.
주간 GPU 잔량은 실제 Kaggle 계정에서 확인하세요.
중단된 Kaggle 실행의 중간 가중치가 항상 복구되는 것은 아닙니다.

## 6. GPU 작업 종료 후

    kaggle kernels status USERNAME/speech-enhancement-e02-unet
    kaggle kernels output USERNAME/speech-enhancement-e02-unet -p runs/e02_unet -o

출력 파일:
- `best_model.pt` 최적 모델 (CNN이 아닌 U-Net 가중치)
- `result.json` 실험/최적 Epoch/평균 PESQ
- `history.csv` Epoch별 기록
- `validation_per_file.csv` 각 Validation 파일의 PESQ

**학습 결과를 바로 GitHub 기본 제출 가중치로 덮어쓰지 마세요.**
CNN보다 검증 성능이 우수하고 U-Net 전용 inference가 준비된 경우에만 다음 제출 모델을 변경해야 합니다.

## 7. Ralph Loop의 다음 단계

지금은 **실험 큐 → GPU 학습 → PESQ 비교 → 파일 수집** 단계가 자동화되었습니다.
Codex AI가 결과를 분석하고 코드를 수정하는 Ralph 반복은 아직 자동 실행되지 않습니다.
그 단계는 `AGENTS.md`의 안전 규칙을 적용하고, 새 모델의 CPU 자기검사 및 GPU smoke test가 통과한 뒤 확장합니다.

추가로 원본 녹음/가중치를 외부 AI 서비스에 전달하지 마세요.


## GPU 요청은 True인데 CUDA가 False인 경우

`kaggle kernels pull ... -m`의 `enable_gpu=true`는 GPU 요청이 저장됐다는 의미이며, 실행 환경의 CUDA 인식까지 보장하지는 않습니다.

**학습 데이터를 사용하지 않는 비공개 GPU 진단 작업**을 먼저 실행하세요:

    git pull
    py tools/kaggle_gpu_probe.py --username siyul1122 --execute

결과 로그에서 다음 값을 확인합니다:
- `nvidia-smi`: GPU 장치 및 드라이버가 실제 런타임에서 보이는지
- `torch.version.cuda`: PyTorch에 CUDA 지원이 포함됐는지
- `torch.cuda.is_available`: PyTorch가 GPU를 실제 사용할 수 있는지
- `GPU_PROBE_RESULT`: 최종 진단 요약

`GPU_PROBE_RESULT = NOT_AVAILABLE`이면 본 학습을 재시도하지 말고 Kaggle 웹 편집기의 Settings → Accelerator 설정과 계정 GPU 사용 자격 및 가용성을 점검하세요. `GPU_PROBE_RESULT = AVAILABLE`이면 이전 작업의 일시적 런타임/할당 문제 가능성이 있으므로 본 학습을 2 Epoch으로 다시 시험합니다. 진단 작업 역시 소량의 무료 GPU 시간을 사용할 수 있습니다.
