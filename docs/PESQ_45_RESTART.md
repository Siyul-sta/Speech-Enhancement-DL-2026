# Ralph Loop V2 재시작 — Validation PESQ-WB 4.5 목표

## 현재 확인 가능한 근거 (2026-11-25)
- GitHub에 기록된 최고 **전체 164개 Validation 평균 PESQ-WB 2.4043** (V1 U-Net).
- V2 복소수 마스크 **2.3125**, V3 위상 보정 **2.3290**.
- 로컬 `runs/` 및 비공개 Kaggle 작업 기록은 GitHub에 존재하지 않으므로 **현재 실제 최고 점수 / 원격 실행 상태는 로컬 파일과 Kaggle에서 재확인**한다.
- 4.5는 PESQ-WB의 사실상 최댓값이므로 **도전적인 장기 목표**이지 달성 보장이 아니다.

## 목표와 측정 원칙
- **목표:** 전체 Validation 164개에서 같은 평가 함수/clean 기준/샘플레이트로 계산한 PESQ-WB 산술평균 **4.5**.
- **중간 점검:** 3.0 / 3.5 / 4.0; 이 목표를 달성했다고 학습을 조기 종료하지 않는다.
- **종료:** 기존 글로벌 우승 모델 혹은 새로 완료한 전체 학습의 **검증된** 164개 점수가 `-TargetPesq` 이상일 때.
- 2 Epoch smoke, 일부 파일 평균, 손실함수 값, 성능 추정치, 로컬 상태 추정치로 우승/목표를 갱신하지 않는다.
- Validation clean은 평가와 실험 선택에만 사용. 실제 추론 입력은 **noisy WAV만** 사용. 숨겨진 Test 점수와 혼동하지 않는다.
- 기존 최고 체크포인트는 덮어쓰지 않고, 실험은 별도 `runs/ralph_v2/run_*/`에 누적된다.

## 실행 전에 확인 (PowerShell, Windows)
기존 반복이 아직 실행 중이면 **새로운 GPU 실험을 시작하지 않는다**. 대시보드만 연결한다.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git status --short
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match '(ralph_loop|ralph_v2)\.py'
} | Select-Object ProcessId, CommandLine
kaggle kernels list --mine
kaggle quota
py tools/reconcile_ralph_v2.py
```

위 명령에서 **로컬 Ralph가 종료되었고** 비공개 Kaggle 실험이 `RUNNING` / `QUEUED` 상태가 아니며 미확정 작업도 없는지 확인해야 한다. `UNKNOWN`, 404, 인증 실패는 작업 종료를 뜻하지 않는다. 이때는 새 학습을 제출하지 말 것. `remote_state.json`을 지우거나 임의로 `COMPLETE`로 수정하지 않는다.

Git 추적 파일에 수정이 남아 있다면 `git reset --hard`나 `git restore`로 지우지 말고 먼저 보존한다. 브랜치 전환이 막히면 강제로 전환하지 않는다.

## 목표 반영 브랜치로 이동
```powershell
git fetch origin feature/ralph-pesq45
git switch --track origin/feature/ralph-pesq45
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
py -m unittest discover -s tests -p "test_kaggle_v2.py" -v
py -m unittest discover -s tests -p "test_ralph_dashboard.py" -v
```

이미 로컬에 같은 이름의 브랜치가 있으면 `git switch feature/ralph-pesq45`로 이동하고 충돌이 없는 상태에서 최신 커밋을 확인한다.

## 안전한 재시작 (원격/로컬 기존 작업 종료를 확인한 경우에만)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob -TargetPesq 4.5 -Epochs 25
```

이 명령은 새로운 GPU 반복 실험을 **실제로 시작**한다. ChatGPT에서 위 명령을 대신 실행한 것은 아니다. PowerShell이 켜져 있어야 로컬 제어기가 계속 반복하며, Kaggle 원격 작업은 별도로 상태를 확인한다.

- Codex가 하나의 구조/손실 가설 제안 → 허용 변경 검사 → 필요 시 2 Epoch smoke → 25 Epoch Kaggle 비공개 학습 → 164개 전체 Validation 평가 → 최고 모델 비교 → 다음 반복.
- 글로벌 최고 점수가 +0.005 이상 좋아지지 않으면 정체로 판정. 두 번 정체 시 `residual_phase` → `magnitude` → `complex` 전환.
- 목표에 미달해도 **Kaggle GPU 잔여시간 2시간 보호**, 중복 원격 작업 보호, 타임아웃, 오류 한도 등 기존 안전 규칙이 우선한다.
- 무료 Kaggle 환경에 따라 코덱스/커널 수행이 제한될 수 있으며 4.5 달성은 보장되지 않는다.

## 관찰과 다음 실험의 우선순위
대시보드: http://127.0.0.1:8765 (포트 충돌 시 별도 포트 사용). 목표 숫자와 남은 점수를 4.5 기준으로 표시한다.

현재 25 Epoch·기존 3종 구조만으로 4.5에 도달하기는 쉽지 않다. 신규 실험은 먼저 **Validation 164개 파일별 이득/손실**, 특정 잡음에서의 퇴행, 과도한 음성 왜곡을 확인한다. 중간 목표를 지속적으로 못 넘으면 별도 후속 브랜치에서 학습 에폭/전략, 증강, 손실함수, 데이터 일반화 성능을 통제된 실험으로 검토한다. 단, 평가 코드·데이터 분할을 바꿔 점수를 올리는 방식은 금지한다.

## 실제 최고 점수 로컬 확인
```powershell
$run = Get-ChildItem .\runs\ralph_v2 -Directory -Filter "run_*" |
  Sort-Object Name -Descending | Select-Object -First 1
Get-Content (Join-Path $run.FullName "global_best.json")
Get-Content (Join-Path $run.FullName "experiments.jsonl") -Tail 5
```

이 파일들은 GitHub에 없어 **로컬에서 확인한 뒤에만** 최신 최고 성능을 확정할 수 있다.
