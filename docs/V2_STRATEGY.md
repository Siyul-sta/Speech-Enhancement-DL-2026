# V2 — 모델 구조·실험 전략 교체

## 무엇이 바뀌나

V1의 `SpectrogramUNet`은 STFT 진폭 기반 0~1 마스크만 예측하므로 노이즈 입력의 위상을 그대로 사용합니다. V2는 **복소수 마스크**를 예측해 **실수부/허수부(크기·위상)를 함께 보정**할 수 있게 했습니다. 이것이 반드시 더 높은 PESQ를 보장하지는 않습니다.

- V2 모델 파일: `src/kaggle_train_v2.py` (V1 파일은 변경 없음)
- V2 실험 제어: `tools/kaggle_v2.py`
- 단위 테스트: `tests/test_kaggle_v2.py`
- 비교 도구: `tools/compare_v2.py`
- 데이터: 기존 비공개 Kaggle `siyul1122/speech-data-private` 재사용, 16kHz mono 3s
- Validation clean은 **164개 PESQ 평가용으로만 사용**, 학습에 사용하지 않음
- Noisy-only 추론 및 48000-sample 길이 유지
- 기존 E02/E03 및 Ralph V1 체크포인트는 덮어쓰지 않음
- 사용자 지정 PESQ=3.0에 도달했다는 이유로 실험을 종료하지 않음

### 두 실험의 차이

| 실험 | 모델 구조 | 학습 목적 |
|---|---|---|
| V2_CRM_PHASE | 복소수 마스크 U-Net | 파형 L1 + MR-STFT 진폭 손실 + 복소 STFT 손실 |
| V2_CRM_SISDR | 동일 | 위 세 손실 + 소량 SI-SDR 손실 |

**모델 아키텍처는 동일하게 고정하고 학습 손실의 변경 효과만 비교**합니다. 이후 가장 성능이 높은 V2 후보와 기존 V1을 공정하게 비교합니다. 옵티마이저는 V2 전체에 AdamW + cosine learning-rate schedule을 공통 적용합니다.

참고 논문: [DCCRN: Deep Complex Convolution Recurrent Network](https://arxiv.org/abs/2008.00264), [Phase-aware Speech Enhancement with Deep Complex U-Net](https://arxiv.org/abs/1903.03107). **논문 모델을 그대로 재현한 것이 아니라 해당 접근에서 아이디어를 취한 경량 실험용 U-Net**입니다.

## 1. 먼저 기존 무제한 반복 중단 (중요)

현재 `start_until_quota.ps1`이 실행 중인 PowerShell 창에서 **Ctrl+C**를 눌러 로컬 제어 프로세스를 종료합니다. 같은 명령을 다시 실행하지 마세요.

새 PowerShell 창에서 로컬 Ralph Loop가 종료됐는지 확인:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'ralph_loop\.py|start_until_quota\.ps1' } |
  Select-Object ProcessId, Name, CommandLine
```

이미 제출한 Kaggle GPU 작업은 여전히 실행 중일 수 있습니다. 가장 최근 Ralph 작업의 참조값과 상태:

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
$job = Get-ChildItem .\kaggle_run -Recurse -Filter kernel-metadata.json |
  Where-Object { $_.FullName -match 'ralph_r\d+' } |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ref = (Get-Content $job.FullName -Raw | ConvertFrom-Json).id
$ref
kaggle kernels status $ref
```

- **RUNNING/QUEUED**: 새 V2 GPU 작업을 제출하지 않습니다. Kaggle 노트북의 웹 페이지에서 실행 버전을 정지하거나 현재 작업 완료를 기다립니다. 기존 Kaggle CLI에는 일반적인 `kernels cancel <slug>` 명령이 없습니다.
- **COMPLETE**: 기존 원격 작업 완료. 원하면 결과를 수동 다운로드한 뒤 V2 전환합니다.
- **ERROR/CANCELLED**: 종료 상태를 확인한 후 V2를 시작합니다.

## 2. V2 코드 받기 / 오프라인 테스트

기존 원격 GPU 실행이 정리되면:

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_kaggle_v2.py" -v
```

이 테스트는 5개이며, `torch`가 로컬에 설치돼 있지 않으면 GPU가 필요 없는 정적 검사 4개는 통과하고 모델 forward/backward 검사는 `skipped`로 표시됩니다. 최종 GPU 훈련 검증은 Kaggle에서 수행합니다.

새 GPU를 사용하지 않고 작업 메타데이터만 생성하는 **드라이런**:

```powershell
py tools/kaggle_v2.py --username siyul1122 --dataset siyul1122/speech-data-private --smoke
```

## 3. GPU 2 Epoch 스모크 테스트

```powershell
py tools/kaggle_v2.py --username siyul1122 --dataset siyul1122/speech-data-private --smoke --execute
```

이 단계는 V2 네트워크의 forward, backward, 데이터 로딩 및 전체 164개 PESQ 평가가 정상인지 확인합니다. **2 Epoch 점수는 25 Epoch V1 최고와 직접 경쟁시키지 않습니다.**

## 4. 스모크 통과 후 비교 실험

```powershell
py tools/kaggle_v2.py --username siyul1122 --dataset siyul1122/speech-data-private --experiments V2_CRM_PHASE V2_CRM_SISDR --epochs 25 --execute
```

두 실험은 **순차 실행**합니다. Kaggle 비공개 커널에 제출하고 완료되면 `runs/v2/<run_tag>/<variant>/`에 `result.json`, `history.csv`, `validation_per_file.csv`, `best_model.pt`을 다운로드합니다.

마지막으로:

```powershell
py tools/compare_v2.py
```

이 프로그램은 기존 최고 V1/이전 Ralph 모델과 V2 두 후보를 같은 Validation 164개 인덱스로 대응시켜 **PESQ 평균, 개선폭, 향상된 파일 비율, 가장 악화된 16개 파일의 평균 변화**를 보여줍니다.

## 5. V2 이후 Ralph Loop V2로 확장할 때

본 비교가 끝나면 최우수 V2를 기준으로 **전체 실험 이력/실패 후보를 Codex 프롬프트에 반영하는 제어기**를 설계할 예정입니다. V1의 제한된 3개 함수 수정 루프를 무작정 재시작하지 마세요.

보호 기준:
- Noisy-only 추론, Clean 사용은 Train supervised / Validation 평가 구분
- 비공개 데이터셋과 토큰 유출 금지
- PESQ 전체 164개 평가 코드 변경 금지
- 모델 코드 변경이 정당한 개선인지 전체 검증으로 판단
- Kaggle 무료 GPU 한도 2시간 보호
- 데이터셋/테스트셋에 과도한 반복 최적화 금지

**주의:** 이 코드는 아직 사용자 Kaggle 환경에서 실험이 실행되지 않은 새 구현입니다. 스모크가 통과하기 전에는 V2의 성능 향상을 주장할 수 없습니다.
