# 8시간 무감독(로컬 PC) Ralph Loop

> **주의:** GitHub Actions나 원격 자동화는 사용하지 않습니다. PC 및 인터넷 연결이 살아 있고 PowerShell이 실행 중이어야 합니다. 무료 GPU는 Kaggle 계정의 주간 한도에 묶여 있습니다. 한도 초과 시 우회하거나 유료 인스턴스를 생성하지 않습니다.

## 목표

- 사용자 지정 PESQ 목표값 3.0 달성에 의해 중단하지 않음 (PESQ-WB 지표 자체의 최대값은 4.5)
- 기존 E02/E03 또는 완료된 Ralph Loop 실험 중 **Validation 164개 PESQ-WB 최고 모델**을 기준으로 이어서 개선
- 성능 악화 시 모델/코드 채택하지 않고 이전 최고 모델 보존
- 성능 정체 2회 중단을 해제하되, 안전을 위해 GPU 여유 2시간 미만이면 중단
- **사용자가 지정한 8시간은 스크립트를 시작한 시점부터** 계산. 기존 3회 실험의 대기시간도 포함
- 기존 실행 중인 3회 작업을 중단하지 않고, Windows PowerShell에서 해당 Python 프로세스가 종료되는 것을 기다린 다음 새 반복 실행 시작
- 최대 40개 실험, 각각 최대 25 Epoch (Early Stopping 활성화), Codex 모델 `gpt-6.1-sol`
- Kaggle job이 정체·비정상 종료되고 작업 상태가 불명확하면 다른 GPU job을 무조건 추가하지 않고 중단

## 사용 방법 (현재 3회 실험이 실행 중인 상황)

**기존 학습 PowerShell은 그대로 두고**, 새로운 PowerShell 창을 열어 다음 두 명령을 실행하세요.

    cd C:\Users\LG\Speech-Enhancement-DL-2026
    git pull --ff-only

    powershell -ExecutionPolicy Bypass -File .\tools\start_overnight.ps1 -Hours 8

`git pull`은 이번에 수정한 `tools/ralph_loop.py` 및 신규 `tools/start_overnight.ps1`을 가져옵니다. 현재 Python 프로세스는 이전 버전 코드를 이미 메모리에 올려 실행 중이므로, **기존 실행 중인 PowerShell에 새 명령을 입력하지 마세요.** 동일 저장소에서 직접 수정 중인 파일과 원격 변경이 충돌하면 pull이 실패할 수 있으니 실패 시 강제 리셋하지 말고 원인부터 확인해야 합니다.

정상 상태로 실행되면 대기 로그:

    Waiting for existing 3-round ralph_loop.py process to FINISH...
    HH:mm:ss: existing Ralph Loop still running (PID ...); waiting...

현재 3회 실험 완료 후:

    Existing local Ralph process finished. Running offline tests...
    Ran 11 tests in ...
    OK
    Resuming best verified model. Remaining overnight budget: ... minutes.
    Starting reference: ... PESQ-WB=...
    ROUND 1: Codex editing a candidate source ...
    ...

동일한 8시간 안에서 다음 작업 제출에 필요한 시간이 부족하면 자동 종료하도록 설정했습니다. **이미 시작한 Kaggle 작업을 시각에 맞춰 강제 중지하는 것은 아닙니다.**

## 현재 3회 실험이 이미 종료된 경우

두 번째 PowerShell에서 위 두 명령을 실행하면 기다리지 않고 자동으로 기존 실험을 검증한 후 새 실험을 시작합니다.

## 로그와 결과 위치

- 실행 로그: `runs/ralph/overnight_YYYYMMDD_HHMMSS.log`
- 이번 새 반복 실험 폴더: `runs/ralph/run_YYYYMMDDHHMMSS/`
- 최고 기록: 새 실험 폴더의 `best.json`
- 최고 모델 소스: 새 실험 폴더의 `best_source.py`
- 최고 모델 체크포인트: `best.json`에 기록된 `checkpoint` 경로
- 라운드별 결과: `round_01/result.json`, `history.csv`, `error.txt`(오류 시)

반복 실험마다 `source_hash`, 164개 Validation 전체 평가 결과 및 로컬 체크포인트의 일관성을 검사해 이전 최고 모델을 이어받습니다. 이전 프로젝트의 CNN, E02, E03 파일은 덮어쓰지 않습니다.

## 안전 제한 및 유의사항

- Kaggle GPU 잔여량이 2시간 미만으로 보고되면 다음 작업 제출을 중단합니다.
- 최대 40회 반복 제한은 안전 제한이며, PESQ 목표 제한과 별개입니다.
- 모델이 2회 연속 개선되지 않아도 계속 시도합니다. 단, 후보 코드 생성이 연속 실패하거나 GPU 상태가 불명확해지면 더 많은 비용 소모나 중복 제출을 피하기 위해 중단할 수 있습니다.
- Codex의 모델 수정은 `ConvBlock`, `SpectrogramUNet`, `enhancement_loss` 정의로 제한되며, 정적 검사는 악성·잘못된 코드의 모든 가능성을 검출하지 않습니다.
- 모델 점수가 지속 상승하거나 PESQ 4.5점에 도달함을 보장하지 않습니다.
- 주의: `runs/`은 `.gitignore`로 제외되므로, 밤새 얻은 결과는 로컬 PC에만 존재합니다. 제출/병합은 별도로 진행합니다.
- Windows에서 **절전 모드가 해제**되어 있어야 하고, 네트워크 연결이 끊기지 않아야 합니다.
