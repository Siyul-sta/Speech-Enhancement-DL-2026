# 로컬 Ralph Loop: 시간 제한 없이 Kaggle 무료 GPU 한도까지

이 모드는 기존 8시간 `start_overnight.ps1` 자동화와 별개입니다. **기존 8시간 대기 스크립트를 중단한 뒤** 시작하세요. 현재 수행 중인 `ralph_loop.py --rounds 3`는 중단하지 않습니다.

## 실행 절차

1. **8시간 대기용 PowerShell 창에서만** `Ctrl+C`를 눌러 기존 `start_overnight.ps1`을 종료합니다. 현재 3회 학습 PowerShell은 그대로 유지합니다.
2. 별도 PowerShell 창에서 프로젝트로 이동하고 최신 버전을 받습니다.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_loop.py" -v
```

3. 기존 PowerShell 창에서 스크립트를 직접 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\tools\start_until_quota.ps1
```

출력이 `Waiting for existing 3-round ralph_loop.py process to FINISH...`이면 정상 대기입니다. 기존 3회 실험이 종료되면 자동으로 최고 PESQ 모델을 읽어 다음 실험을 진행합니다.

## 종료 기준

- **8시간 제한 없음** (`--max-total-minutes 0`)
- **반복 횟수 제한 없음** (`--rounds 0`)
- **PESQ 목표 중단 없음** (`--no-pesq-target`)
- **연속 개선 실패에 따른 조기 종료 없음** (`--max-stagnant 0`)
- GPU 잔여 시간이 **2시간 미만**인 것으로 확인되면 다음 GPU job을 제출하지 않고 종료합니다.
- 한 개의 GPU job은 완료를 최대 80분 기다립니다. 이후에도 Kaggle에서 실행 중일 수 있으므로 중복 제출을 막고 프로그램을 멈춥니다.
- Codex 오류, GPU 실패, 정보 불확실성 같은 안전 문제로도 중단될 수 있습니다.
- 데이터는 비공개 Kaggle 데이터셋에 유지하고, 기존 E02/E03 및 우승 체크포인트는 지킵니다.

로그: `runs/ralph/until_quota_YYYYMMDD_HHMMSS.log`, 우승 모델: 최신 `runs/ralph/run_*/best.json`.

**주의:** 무료 GPU 주간 사용량 한도 자체를 무시하지 않으며, 한도가 초기화될 때까지 자동 재개하는 기능은 아닙니다. GPU 잔여시간 부족으로 중단되면 다음 초기화 이후 사용자가 다시 실행할 수 있습니다. Windows PC 및 PowerShell/인터넷 연결을 계속 유지해야 합니다.