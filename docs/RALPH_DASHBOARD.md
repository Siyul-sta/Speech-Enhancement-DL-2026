# Ralph Speech Lab — 실시간 모니터링 대시보드

기존 Adaptive Ralph Loop V2의 Kaggle 작업을 PowerShell 로그 외에 브라우저에서도 한눈에 확인하는 로컬 UI입니다.

## 무엇을 볼 수 있나?

- 최고 PESQ-WB 및 목표 3.0까지 남은 점수
- 실행 중인 라운드, 모델 계열, Kaggle 노트북 식별자
- 실험 진행 단계: Codex 가설 생성 → 필요 시 스모크 테스트 → GPU 본 학습 → PESQ 검증
- 최근 점수 변화 그래프, 완료된 라운드 목록, 3개 모델 계열의 최고 점수
- 로컬 실행 로그 최신 12줄
- Kaggle 실행 중 로그에서 실제 Epoch과 Validation PESQ 값을 읽을 수 있을 때만 숫자로 표시

화면은 4초마다 갱신됩니다. 제어기에는 15초 간격의 하트비트가 추가되었습니다.

**보안·비용:** 기본 Python 라이브러리만 사용합니다. 브라우저 서버는 127.0.0.1(본인 컴퓨터)에만 바인딩하고, 원본 음성이나 비공개 데이터는 읽지 않으며, 절대로 Kaggle GPU 작업을 제출·종료하거나 체크포인트를 변경하지 않습니다.

## 지금 Ralph Loop가 이미 실행 중이라면

**실행 중인 컨트롤러를 끄지 말고 새 PowerShell 창**을 엽니다.

~~~powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\tools\start_dashboard.ps1
~~~

실행하면 기본 브라우저에서 다음 주소가 열립니다.

http://127.0.0.1:8765

대시보드 PowerShell 창은 켜 두세요. 닫거나 Ctrl+C를 누르면 **대시보드 서버만 중단**되고, 별도 창에서 돌고 있는 Ralph Loop와 Kaggle 작업에는 영향을 주지 않습니다.

**주의:** 오래된 버전의 Ralph 제어기는 UI용 진행 상태 파일을 쓰지 않았으므로, 과거 결과·PESQ 그래프는 표시되더라도 현재 실험의 단계·Epoch은 표시되지 않을 수 있습니다. 이때 기존 원격 GPU 작업을 강제로 중단하지 마세요.

## 새 Ralph Loop 실행 시 자동으로 함께 켜기

기존 실행 명령을 그대로 사용하면 대시보드 창과 브라우저를 함께 열도록 구성했습니다.

~~~powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_ralph_dashboard.py" -v
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
& .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob
~~~

마지막 줄은 **기존 Ralph 프로세스나 다른 Kaggle GPU 작업이 실행 중이지 않음을 확인한 후에만** 입력합니다. 이미 실행 중이면 위의 '대시보드만 열기' 명령만 사용하세요.

UI를 자동 실행하지 않으려면 다음과 같이 실행합니다.

~~~powershell
& .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob -NoDashboard
~~~

## 표시되는 값의 의미

- LIVE: 최근 155초 안에 로컬 제어기의 하트비트가 기록되었을 때. 서버가 꺼져 있거나 컴퓨터를 종료하면 이후 OFFLINE으로 표시됩니다.
- Training: 원격 Kaggle 상태를 주기적으로 확인하는 단계입니다. 실제 Epoch 로그를 확보한 경우에만 퍼센트를 계산합니다.
- PESQ 최고 기록: 로컬 우승 모델 정보입니다. 2 Epoch 스모크 점수로 갱신하지 않습니다.
- 실험 그래프: 완료된 라운드의 결과 및 검증된 복구 작업의 결과를 포함합니다.
- 기록은 로컬의 runs/ralph_v2 아래에서 읽습니다. 별도로 연결된 다른 Kaggle 실험에 대한 정보는 임의로 추정하지 않습니다.
- 오류 로그: 비밀번호나 인증키 등 민감한 키워드가 포함된 줄은 노출하지 않습니다. 로컬 UI는 개인 컴퓨터에서만 사용하는 것을 권장합니다.

## 실행 환경·테스트

대시보드 코드는 GitHub에 반영되었지만 사용자 Windows 브라우저에서의 실제 동작은 로컬 테스트가 필요합니다.

~~~powershell
py -m unittest discover -s tests -p "test_ralph_dashboard.py" -v
py -m unittest discover -s tests -p "test_ralph_v2.py" -v
~~~

실행 중 8765번 포트가 이미 사용 중이면 기존 브라우저의 http://127.0.0.1:8765 주소에서 먼저 확인하세요.
