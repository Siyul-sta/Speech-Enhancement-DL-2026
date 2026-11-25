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

## 기존 V1 파일이 수정된 상태인 경우

기존 Ralph V1 실행 후 `src/kaggle_train.py`가 Git 추적 변경으로 남을 수 있습니다. Adaptive V2는 **이 파일 하나만 변경된 경우**, 기존 내용의 SHA-256 해시를 기록하고 `runs/ralph_v2/run_*/preserved_v1_trainer.py`로 바이트 단위 백업합니다. 파일 내용을 변경하거나 `git restore`를 실행하지 않습니다.

Codex가 실행 중 이 파일이나 다른 Git 추적 파일을 수정하면 안전상 중단합니다. `src/kaggle_train.py` 이외의 추적 파일도 변경되었다면 이전과 동일하게 실행을 거부합니다. 과거 우승 2.4043 기록은 검증된 `runs/ralph`의 checkpoint와 `best_source.py`를 기반으로 복원합니다.

**따라서 로컬에서 수동으로 `git restore src/kaggle_train.py` 또는 `git reset --hard`를 실행할 필요가 없습니다.**

## 2026-11-21 실제 오류: 노트북 ID와 Kaggle URL Slug 불일치

원인 확정: 기존 adaptive 제어기가 제출한 `kernel-metadata.json`의 `id`는 `speech-enhancement-auto-v220261121052503-j0001`이었지만, Kaggle에 생성된 **실제** 노트북은 제목에서 정규화한 `adaptive-speech-v2-v220261121052503-job-1`이었습니다. 따라서 잘못된 ref에 대한 `kernels status` 조회가 `kernels.get denied`로 실패했으며, 이 메시지만으로 인증/권한 문제가 있다고 결론내리면 안 됩니다.

이후 신규 제출의 `id`를 실제 제목 유래 URL slug와 일치하도록 고쳤습니다. 기존 잘못 기록된 `remote_state.json`에 대해서는 `kaggle kernels list --mine`에 해당 정규 ID가 **정확히 1개 존재**함을 검증한 경우에만 교정하고 실제 작업을 조회합니다.

```powershell
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
kaggle kernels status siyul1122/adaptive-speech-v2-v220261121052503-job-1
py tools/reconcile_ralph_v2.py
```

실제 ID가 `COMPLETE` / `ERROR` / `CANCELLED`이고 해당 상태가 공식 상태 조회로 확인되면:

```powershell
py tools/reconcile_ralph_v2.py --resolve
py tools/reconcile_ralph_v2.py
```

- `COMPLETE`: 164파일 점수 + 체크포인트 다운로드 후 기록을 `COMPLETE`로 전환. **2 Epoch 스모크 점수는 챔피언에 포함하지 않음.**
- `ERROR/CANCELLED`: 안전한 종료 상태 확인 후 `ERROR_TERMINAL`로 기록.
- `RUNNING/QUEUED/UNKNOWN`: 새 GPU 작업을 제출하지 않음. 조회만 수행.
- `--resolve`는 GPU 제출이 아니라 기존 작업 상태 복구 작업입니다.

복구된 후 다른 미해결 원격 작업이 없고 기존 controller 프로세스도 완전히 중단되었다면 `& .\\tools\\start_adaptive.ps1 -ConfirmedNoRemoteJob`로 **새 연구 반복**을 시작할 수 있습니다.

## Kaggle 상태 조회 오류 발생 시 복구 (새 실험 금지)

`kaggle kernels push` 뒤 `kaggle kernels status`가 HTTP 404/연결 오류 등으로 실패할 수 있습니다. 새 Ralph Loop V2는 **일시적 상태 조회 실패 5회 재시도** 후에도 상태가 불명확할 경우 중복 GPU 제출을 방지하기 위해 중단합니다. **404가 곧 학습 중단을 의미하지는 않습니다.**

기존 커널을 재제출하지 말고 다음 명령을 먼저 실행하세요.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
kaggle kernels status siyul1122/speech-enhancement-auto-v220261121052503-j0001
kaggle kernels list --mine
py tools/reconcile_ralph_v2.py
```

`reconcile_ralph_v2.py`는 기본적으로 **상태 확인만 수행**하고, Kaggle GPU 작업을 제출하지 않으며, 로컬 기록을 변경하지 않습니다.

- `RUNNING/QUEUED`: 새 실험을 시작하지 않습니다. 종료를 기다리세요.
- `COMPLETE`: 결과 파일을 가져와 안전하게 상태를 업데이트할 수 있습니다.
- `ERROR/CANCELLED`: GPU 학습 종료가 확인된 경우에만 안전하게 완료 상태로 기록합니다.
- `404/UNKNOWN/인증 실패`: 웹의 비공개 노트북과 계정 상태도 확인하기 전에는 절대로 중복 제출하지 않습니다.

실제 상태가 `COMPLETE`나 `ERROR/CANCELLED`로 확인된 **후에만**:

```powershell
py tools/reconcile_ralph_v2.py --resolve
```

복구 스크립트는 `COMPLETE`인 작업의 `result.json`, `best_model.pt`, `validation_per_file.csv`를 확인하고, 164개 전부의 점수가 있어야 완료로 기록합니다. 스모크 결과는 글로벌 우승 모델로 채택하지 않습니다. `ERROR/CANCELLED`인 작업은 종료 상태로 표시하되 성공 결과로 간주하지 않습니다.

이전 실행이 정상 완료된 2 Epoch 스모크 단계에서 끝났고 동일 코드에 대한 본 학습을 아직 제출하지 않았다면, 검증된 기존 후보를 자동 재사용해 **중복 스모크 없이 25 Epoch 학습**으로 진입합니다. 복구가 불완전하거나 이미 본 학습이 제출된 경우에는 안전상 재사용하지 않습니다.

## 컴퓨터 재시작 후 자동 복구 (2026-11-23)

Kaggle GPU 학습 중 컴퓨터 또는 PowerShell이 종료되면 원격 Kaggle 작업은 완료되더라도 로컬 `remote_state.json`이 `SUBMIT_PENDING`으로 남을 수 있습니다. **기존 `& .\\tools\\start_adaptive.ps1 -ConfirmedNoRemoteJob` 실행 명령은 그대로 사용**합니다.

업데이트된 시작 스크립트는 자동으로 `py tools/reconcile_ralph_v2.py --resolve`을 먼저 실행합니다. 실제 원격 작업이 `COMPLETE`인 경우 결과와 검증 164개 점수, 모델 파일을 내려받아 `COMPLETE`로 기록합니다. `ERROR/CANCELLED` 작업은 실제 Kaggle 상태가 확인된 경우에만 종료 처리합니다. `RUNNING/QUEUED/UNKNOWN`이면 제어기의 기존 중복 제출 보호가 계속 작동하여 새 실험을 시작하지 않습니다.

또한 Ralph 제어기는 이전 실행의 `round_*/full` 결과 중 **파일이 실제로 검증되었고 원래 제출된 모델 코드가 로컬 Kaggle 패키지와 일치하는 경우**에만 점수를 다시 읽어 계열별 최상위 모델과 전체 우승 모델 후보로 반영합니다. 예를 들어 `run_v220261121054951/round_0005/full`을 복구해도 새 실행은 그 점수를 무시하거나 같은 원격 작업을 다시 제출하지 않습니다.

실행 예시:

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
& .\\tools\\start_adaptive.ps1 -ConfirmedNoRemoteJob
```

**사용자가 확인한 상태(2026-11-23):** `job-10`은 `RESOLVED COMPLETE. Verified checkpoint, result, 164 PESQ rows.`로 복구되었으며, 그 직후 `No unresolved Ralph V2 remote jobs recorded.`가 출력되었습니다. `job-10`의 실제 PESQ는 로컬 `runs/ralph_v2/run_v220261121054951/round_0005/full/result.json` 파일에서 확인합니다.

## 완료된 V3 2 Epoch 스모크 재사용 (2026-11-21)

`run_v220261121052503/round_0001/smoke`의 Kaggle 테스트가 완료되었고 `V3_RESIDUAL_PHASE`의 2 Epoch PESQ-WB는 **1.8653953**입니다. 이는 정상 작동 검증이며 25 Epoch 성능으로 간주하지 않습니다. 이 파일들은 앞서 `reconcile_ralph_v2.py --resolve`로 164개 Validation 점수와 체크포인트까지 확인되었습니다.

Adaptive 제어기 최신 버전은 **완료된 스모크 기록이 있고 해당 후보의 전체 학습이 제출된 적이 없으면** `candidate.py`를 안전하게 복원하여 **다음 실행의 첫 라운드에서 바로 25 Epoch 본 학습**을 시작합니다. 같은 2 Epoch GPU 스모크를 다시 제출하지 않습니다.

복원 전 검사:
- `smoke/remote_state.json`이 `COMPLETE`
- `result.json`이 `V3_RESIDUAL_PHASE`, 정확히 2 Epoch, Validation 164개
- 164개 PESQ 파일, 스모크 체크포인트 존재
- 당시 생성된 `round_0001/candidate.py`가 보호된 데이터/평가/훈련 코드를 변경하지 않음
- 이전 `round_0001/full` 폴더가 존재하지 않음 (중복 제출 방지)
- 동일 계열에 이미 채택한 25 Epoch 모델이 없는 경우

로그에서 다음을 확인할 수 있습니다.

```text
RECOVERED VERIFIED SMOKE: ... family=residual_phase; skipping repeat 2-epoch GPU job
Using previously validated smoke candidate; no duplicate smoke GPU
ROUND 1: model family=residual_phase ...
```

**중요:** 스모크에서 학습된 2 Epoch *가중치*를 이어 학습하지는 않습니다. 검증을 통과한 *코드 및 실험 설정*을 그대로 사용해서 25 Epoch를 새로 학습합니다. 25 Epoch 결과가 검증되면 기존 2.4043과 비교 후 자동 개선 루프가 다음 모델/가설로 진행합니다.

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

## 4.5 목표 모드 (2026-11-25)

새 명령 예시: `& .\\tools\\start_adaptive.ps1 -ConfirmedNoRemoteJob -TargetPesq 4.5 -Epochs 25`

이 옵션을 쓸 때는 [PESQ 4.5 재시작 가이드](PESQ_45_RESTART.md)의 중복 GPU 작업/상태 복구 확인을 먼저 수행한다. V2는 유효한 전체 Validation 최고 점수가 목표에 도달한 경우에만 목표 달성으로 종료하고, 그 외에는 GPU 쿼터·오류·안전 제한을 지킨다. 목표 4.5는 달성을 보장하지 않는다.
