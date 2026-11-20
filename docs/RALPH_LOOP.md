# Ralph Loop — 로컬 PC에서 Codex + Kaggle GPU 자동 개선

## 현황

- E02/E03 학습 큐 (`tools/kaggle_loop.py`)와 **별개**입니다. 기존 E02/E03 학습이 끝날 때까지 이 코드를 실행하지 마세요.
- 기존 E02 2 Epoch PESQ 1.8846, E02 11 Epoch에서 최고 2.2147까지 확인됨. **본 학습과 E03 최종 점수는 아직 여기서 확인하지 않았습니다.**
- 이 문서에 설명된 `tools/ralph_loop.py`는 Codex가 매번 학습 코드의 한 부분을 개선하도록 요청하고, 별도 Kaggle GPU 작업에 보내 PESQ-WB 164개 검증점수로 후보를 평가하며, 최고 후보를 보존하는 **유한 반복(최대 10회)** 자동화입니다.
- 소스 코드 구현을 완료했지만 **실제 사용자 Windows 환경에서 Codex→Kaggle의 전체 루프 테스트는 아직 하지 않았습니다.** 첫 실행 결과에 따라 수정이 필요할 수 있습니다.
- ChatGPT Work, GitHub Actions, 유료 Colab은 사용하지 않습니다. 실행하는 동안 Windows PC, PowerShell, 인터넷 연결이 유지되어야 합니다.

## 1. 이미 실행 중인 E02/E03 자동 실험 완료 확인

E02와 E03을 연속 학습하는 **원래 PowerShell 창**에서 아래 메시지까지 나와야 합니다.

    RESULT E02_UNET: ...
    RESULT E03_MRSTFT: ...
    Experiment queue finished; inspect runs/ before planning the next model.

E02/E03 중 하나가 `ERROR`라면 오류부터 해결하세요. 자동 루프를 동시에 실행하면 무료 GPU 쿼터를 쓸데없이 사용할 수 있고, 로컬 E03 실행 직전의 소스를 수정해 실행 내용이 달라질 수도 있습니다.

로그 실시간 확인:

    kaggle kernels logs siyul1122/speech-enhancement-e02-unet --follow --interval 10

## 2. 시작 전 점검

기존 코드 변경을 보관/커밋하고, 저장소의 추적 파일 변경사항이 없는 상태에서:

    cd C:\Users\LG\Speech-Enhancement-DL-2026
    git pull
    codex login status
    kaggle quota

`codex login status`는 ChatGPT로 로그인한 상태여야 합니다. API 키는 요구하지 않습니다.
Kaggle 데이터셋은 **기존 비공개 `siyul1122/speech-data-private`만 연결**합니다.
동일한 데이터를 재업로드하지 않습니다.

## 3. 드라이런 (무료: Codex/GPU 제출 없음)

    py tools/ralph_loop.py --username siyul1122 --dataset siyul1122/speech-data-private

출력에서 `DRY RUN ONLY`을 확인합니다.

## 4. AI 반복 3회 실행 (한도가 있는 자동 실험)

    py tools/ralph_loop.py --username siyul1122 --dataset siyul1122/speech-data-private --rounds 3 --epochs 25 --max-total-minutes 180 --execute

실제 작동 순서:
1. `runs/e02_unet/result.json`, `runs/e03_mrstft/result.json`에서 **현재 최고 PESQ(164건)** 탐색
2. `codex exec --sandbox workspace-write`가 `src/kaggle_train.py`에서 새 모델/손실 함수 가설 하나를 코드로 구현
3. 변경된 파일이 `src/kaggle_train.py`뿐인지 확인하고, Python 문법과 필수 엔트리포인트 검사
4. `speech-enhancement-ralph-r01`처럼 **독립적인 비공개 Kaggle T4 작업**으로 GPU 학습
5. 전체 Validation 164개 PESQ-WB 평균 확인 및 체크포인트 다운로드
6. 0.005점 이상 개선되면 새 최고 모델과 코드를 보존; 아니면 코드를 이전 최고 버전으로 되돌림
7. 정한 횟수/목표 점수/개선 정체/시간 제한/GPU 잔량 조건에서 중단

중단 안전장치:
- `--rounds 3`: 최대 3회 AI 수정 및 최대 3회 Kaggle 작업
- `--epochs 25`: 각각 최대 25 Epoch, Early Stopping 적용
- `--max-total-minutes 180`: 다음 작업 시작 전에 누적 시간 체크 (진행 중인 GPU 잡을 강제 종료하는 설정은 아님)
- `--max-job-minutes 80`: Kaggle 작업 완료 확인을 기다리는 최대 시간. 초과 시 로컬 루프는 멈추지만 Kaggle의 실제 작업은 계속될 수 있음
- `--min-gpu-quota-hours 2`: 매 실험 전 `kaggle quota` 확인, 잔여 GPU 시간이 2시간 미만이면 중단
- `--max-stagnant 2`: 2회 연속 PESQ 개선이 없으면 중단
- `--stop-at-pesq 3.0`: PESQ 3.0을 달성하면 중단 (3.0 달성은 보장하지 않음)

## 5. 진행 확인

자동 루프를 실행한 원래 PowerShell 화면에 각 실험 `PESQ-WB`, `IMPROVED`, `NO IMPROVEMENT`가 출력됩니다.

Kaggle 로그 보기:

    kaggle kernels logs siyul1122/speech-enhancement-ralph-r01 --follow --interval 10

파일은 `runs/ralph/`에 보관:
- `best.json`: 최고 기록, Kaggle 작업 주소, 체크포인트 경로
- `best_source.py`: 최고 모델의 재현 가능한 소스 코드
- `round_01/result.json`: 평균 PESQ, epoch, 성능 차이
- `round_01/best_model.pt`: Kaggle에서 다운로드한 최고 모델
- `round_01/candidate_source.py`, `hypothesis.txt`: 모델 변경 내용
- `round_01/codex_log.txt`: AI 작업 로그

`runs/`은 GitHub에 업로드되지 않도록 ignore 되어 있습니다.
원래 `checkpoints/best_mask_cnn.pt`는 절대 변경되지 않습니다.
Kaggle 외부에 음성파일을 추가 전송하지 않습니다.

## 6. 주의사항

- Codex가 변경한 코드에는 결함이나 성능 악화가 있을 수 있습니다. 문법/인터페이스 검사는 실행 성공이나 안전성을 보증하지 않습니다.
- **Kaggle 작업을 중복 제출하지 마세요.** 기존 E02/E03이 완료되기 전에는 AI 반복 루프를 실행하지 마세요.
- Codex 로컬 코드 편집을 허용하므로, 본인 PC에서는 주요 파일 백업을 유지하고 `AGENTS.md` 지침을 검토하세요.
- AI가 만든 코드는 Kaggle에서 실행되므로, 데이터 취급 제한에 주의하고 전면 무감독 실행 전에 코드 변경 내용과 네트워크 접근을 검토하는 것을 권장합니다.
- 일부 GPU 실행 중단이나 Codex 호출 제한, Kaggle 할당 제한으로 자동 실험이 실패할 수 있습니다.
- 최고 Validation PESQ는 숨겨진 Test PESQ를 보장하지 않습니다. 본 챌린지의 입력 제약은 **noisy only**로 유지해야 합니다.
- **이 컨트롤러는 실험 코드와 최고 가중치만 보존합니다.** 우수 후보가 나와도 실제 제출용 inference 구현/검증과 GitHub main 병합은 별도의 단계입니다.
