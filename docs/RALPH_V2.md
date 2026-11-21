# Ralph Loop V2 — 성능 정체 시 자동 모델 구조 전환

V1에서는 동일한 `SpectrogramUNet` 내부의 일부 코드만 계속 수정했습니다. Adaptive V2에서는 **PESQ 개선이 정체될 때 모델 계열을 바꾸고**, 각 계열에 대한 실험 이력과 최고 모델을 유지합니다.

## 자동 순환 구조

```text
현재 우승 모델 2.4043 (V1, 검증 164개)
              |
     Codex: 1가지 가설 제안
              |
       모델 코드 수정/보호 검사
              |
     Kaggle 비공개 GPU 학습
              |
    Validation 164개 PESQ-WB
              |
       +0.005 이상 갱신?
         /           \
       YES            NO
       |              |
  우승 모델 저장    정체 카운트 +1
                      |
               연속 2회 정체?
                /        \
               NO        YES
               |          |
            현 계열 유지   모델 계열 전환
                            |
             residual_phase → magnitude → complex
                            |             |
                      각 계열 최고 모델/소스 저장
                            |
                       실험 이력 누적
                            |
                       다음 Codex 반복
```

**모델 계열**
- `residual_phase` (첫 번째): `ResidualPhaseUNet`, 진폭 마스크 + 별도 위상 보정 헤드. **신규 V3**.
- `magnitude` (두 번째): 검증된 최고 V1 `SpectrogramUNet`의 **우승 소스**를 출발점으로 반복 탐색.
- `complex` (세 번째): `ComplexMaskUNet`, 실수부/허수부 복소수 마스크 기반의 **V2**. 기존 V2 2.3125 기록을 계열 기준선으로 사용.

각 계열에 들어가면 Codex가 해당 아키텍처의 핵심 네트워크와 손실함수에 대해 **한 번에 하나의 새로운 가설**을 코드로 적용합니다. 단, 모델의 층·헤드 구성과 함수 내부 연산은 변경할 수 있어도 새로운 최상위 클래스/함수, 학습 데이터·PESQ 평가·검증 샘플 수는 변경할 수 없습니다.

처음 학습하는 모델 계열은 **2 Epoch smoke**를 통과해야 정식 25 Epoch 학습이 시작됩니다. Smoke 점수는 우승 후보에 포함하지 않습니다.

**실험 이력:** `runs/ralph_v2/run_*/experiments.jsonl`에 라운드별 가설, 점수, 성공·실패를 누적하고, 다음 Codex가 최근 15회 이력을 참고합니다.

**우승 기록:** `runs/ralph_v2/run_*/global_best.json`, `families/*/best.json`, `best_source.py`. 기존 `runs/e02_unet`, `runs/e03_mrstft`, `runs/ralph`, `runs/v2`는 덮어쓰지 않습니다.

> 주의: 새로운 아키텍처로 바꿀 때 **기존 가중치를 항상 이어서 학습하는 것은 아닙니다.** 계열별 최상위 코드를 이어받지만 각 GPU 실험은 현재 학습 스크립트대로 새 가중치에서 학습합니다. 이는 체크포인트의 네트워크 형태가 다를 때 잘못된 로딩을 막기 위한 선택입니다.

## 실행 전

**기존 `start_until_quota.ps1`이 실행 중이면 창에서 Ctrl+C로 종료**하세요. 예전 제어기를 그대로 두면 예전 V1 실험을 계속 제출합니다.

로컬 구버전 프로세스 확인:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match '(ralph_loop|ralph_v2)\.py' } |
  Select-Object ProcessId, Name, CommandLine
```

**기존 Kaggle GPU Job도 확인**합니다. 가장 최근 노트북이 `RUNNING` 또는 `QUEUED`라면 완료되거나 안전하게 중단한 후 V2를 시작하세요. 로컬 Ctrl+C가 기존 Kaggle GPU 작업을 자동 취소하지는 않습니다.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
$job = Get-ChildItem .\kaggle_run -Recurse -Filter kernel-metadata.json |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ref = (Get-Content $job.FullName -Raw | ConvertFrom-Json).id
$ref
kaggle kernels status $ref
kaggle quota
```

`$job`이 어떤 노트북인지 실제 출력된 ref로 확인하세요. 다른 계정/작업의 GPU도 동시에 실행 중인 경우 Kaggle 내 목록을 확인하세요.

## 설치·테스트

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
py -m unittest discover -s tests -p "test_kaggle_v2.py" -v
```

모두 `OK`여야 합니다. 로컬에 PyTorch가 없으면 별도 테스트가 `skipped`로 출력될 수 있습니다. **새 제어기는 아직 사용자 Windows/Kaggle 환경에서 실행 검증 전**입니다.

GPU 소비 없는 초기 기준선 확인:

```powershell
py tools/ralph_v2.py --username siyul1122 --dataset siyul1122/speech-data-private
```

## 자동 실행 (확인 후 딱 한 번)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob
```

`-ConfirmedNoRemoteJob`는 **기존 Kaggle 원격 작업이 실행 중이지 않다는 것을 직접 확인한 뒤에만** 붙이세요. 이 옵션이 원격 작업을 자동 취소하거나 상태를 검증하는 것은 아닙니다.

제어기는 테스트 통과 후 **첫 번째 residual_phase**로 시작하고, 최고 PESQ를 두 번 연속 갱신하지 못하면 모델 계열을 전환합니다. `max-rounds 0`이므로 8시간 시간 제한이나 PESQ 3.0 종료 기준은 없습니다. **GPU 잔여시간이 2시간 미만**이면 새 학습을 제출하지 않습니다.

기본 중단 보호 조건:
- Kaggle GPU 잔여시간 확인 불가/부족
- GPU 작업 타임아웃(기본 80분), 상태 불명확, 중복 작업 우려
- 연속 3회 코드 변경/평가 오류
- 로컬 Ctrl+C 또는 PC/인터넷 중단
- 다른 Ralph 제어기와의 충돌
- 모델이 허용된 범위를 벗어나 학습/평가 코드를 건드린 경우

동시에 두 제어기를 실행하지 마세요. `runs/ralph_v2/active.lock`가 남아 있으면 실제 프로세스와 원격 GPU 작업부터 확인하세요. **프로세스가 살아 있는데 락 파일만 삭제해서는 안 됩니다.**

## 진척도 확인

```powershell
$run = Get-ChildItem .\runs\ralph_v2 -Directory -Filter "run_*" |
  Sort-Object Name -Descending | Select-Object -First 1
Get-Content (Join-Path $run.FullName "global_best.json")
Get-Content (Join-Path $run.FullName "experiments.jsonl") -Tail 5
```

`MODEL SWITCH: residual_phase -> magnitude`와 같은 출력이 뜨면 **실제로 모델 계열이 자동 전환된 것**입니다. 최대 성능 개선은 보장되지 않으며 과도한 Validation 반복 최적화에 따른 일반화 위험은 남습니다.
