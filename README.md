# dist_train_rwkv

RWKV-7 LoRA SFT를 여러 GPU worker에 나눠 학습하는 **프로토타입**입니다. 한 대의 coordinator가 adapter와 학습 데이터 조각(unit)을 HTTP로 나눠 주고, worker들이 unit별 gradient를 계산해 돌려보내면 coordinator가 합산해서 AdamW를 한 번 갱신합니다. 현재 구현은 `RWKV/RWKV7-G1j-1.5B-20260831`, rank 8 / alpha 16, attention 4개 projection(`att.receptance/key/value/output`) LoRA에 고정돼 있습니다.

> **검증 범위.** 실제 GPU(RTX 3060)에서는 1.5B 모델로 **한 step**을 worker 프로세스 둘이 순차로 나눠 계산한 것까지 확인했습니다. 여러 step, 동시 worker, resume, Hugging Face push/resume은 CPU 테스트(합성 gradient, 가짜 Hub)로만 검증했습니다. 외부에 공개된 네트워크에 노출하지 마세요(아래 [보안](#보안과-한계) 참고).

## 어떻게 동작하나

```
          adapter snapshot(θ) + unit             gradient (FP32 safetensors)
coordinator ───────────────────▶ worker 1..N ───────────────────────────────▶ coordinator
 (AdamW, 체크포인트)             (frozen BF16 base + LoRA)        Σ gradient / Σ loss token → AdamW 1회
```

- **round = optimizer step.** round마다 고정된 adapter 버전(`theta_version`, `theta_hash`)과 unit 묶음이 정해집니다.
- worker는 unit 하나마다 lease를 받아 CE loss 합(`reduction="sum"`)의 gradient를 계산해 반환합니다. coordinator가 `Σ gradient / Σ loss token`으로 평균을 내고 AdamW를 한 번 적용합니다.
- 오래된 adapter 버전, 중복, 만료된 lease, 코드 해시 불일치, 형식이 틀린 gradient는 거부됩니다.
- worker는 coordinator에서 adapter만 받고, base 모델은 각자 로컬에 둡니다. 모델 샤딩이 아니라 **데이터 병렬**입니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `g1j_dist_coordinator.py` | coordinator 서버. 라운드 계획, 체크포인트/resume, Hub push |
| `g1j_dist_worker.py` | worker. unit gradient 계산 (CUDA 필요) |
| `rwkv_dist_protocol.py` | 프로토콜 코어: lease, 검증, 합산, AdamW |
| `rwkv_dist_client.py` | worker 쪽 HTTP 클라이언트, 재접속 대기 |
| `rwkv_dist_checkpoint.py` | coordinator 체크포인트(원자적 저장, 검증, 호환성 검사) |
| `rwkv_hub.py` | Hugging Face 모델 내려받기, adapter push, Hub 체크포인트 조회 |
| `prepare_g1j_protocol_units.py` | 샘플 unit manifest 생성 (4개 행짜리 프로토콜 검증용) |
| `validate_g1j_sft_protocol.py` | TRL로 2 step 학습해 초기 adapter를 만드는 검증 스크립트 |
| `validate_g1j_dist_checkpoint.py` | 결과 adapter가 PEFT로 로드되고 coordinator의 theta와 같은지 확인 |
| `test_*.py` | 단위/통합 테스트 (CPU) |
| `*Design*.md`, `G1j_1.5B_Training_Protocol_Validation_*.md`, `VRAM_Baseline_*.md` | 설계, 검증 기록, VRAM 측정 |
| `profile_*.py`, `*.json` | 단일 GPU VRAM 프로파일 스크립트와 결과 |

## 요구사항

- **Python.** 개발은 PyTorch 2.14.1(cu126), Transformers 5.20.1, PEFT 0.21.2 환경(`/venv/main`)에서 했고, 테스트는 Python 3.13 / CPU 전용 torch에서도 돌려 봤습니다. 더 낮은 Python 버전은 확인하지 않았습니다. 의존성 목록 파일(`requirements.txt`)은 아직 없습니다.
- **coordinator:** `torch`(CPU면 충분), `safetensors`, `numpy`, `huggingface_hub`(Hub 기능에만). GPU가 필요 없고, 모델 가중치도 필요 없습니다. `--model-dir`에 `release-manifest.json`이 있는 폴더만 있으면 해시 확인에 쓰입니다.
- **worker:** CUDA GPU, `torch`(CUDA), `transformers`, `peft`, 그리고 BF16 base 모델 스냅샷. 1.5B는 RTX 3060 12 GiB에서 검증했습니다.
- 데이터 준비/검증 스크립트는 추가로 `datasets`, `trl`이 필요합니다.
- 모든 요청은 공유 토큰(`Authorization: Bearer ...`)으로 인증합니다. 토큰은 환경 변수 `RWKV_PROTOCOL_TOKEN`으로 넘기는 편이 좋습니다(명령행 인자는 `ps`에 보입니다).

## 빠른 시작

아래는 저장소에 들어 있는 4개 unit짜리 샘플(`runs/g1j_local_protocol/units.json`)로 한 번 돌려 보는 흐름입니다. 모델은 `/workspace/RWKV7-G1j-1.5B-20260831`에 있다고 가정합니다(다른 위치면 `--model-dir` / `--model-path`).

### 0. 준비물 두 가지

1. **base 모델.** worker 머신에서 `--fetch-model-from-hf`로 고정 revision을 내려받을 수 있습니다(토큰 없이 공개 다운로드). 이미 받아 뒀다면 경로만 지정하세요.
2. **초기 adapter.** coordinator가 `--initial-adapter-dir`(기본값 `artifacts/g1j_1.5b_capybara_sft_protocol`)에서 `adapter_config.json`과 `adapter_model.safetensors`를 읽습니다. 이 폴더의 설정 파일은 저장소에 있지만 **가중치 파일은 `.gitignore`로 제외돼 있어 새로 클론하면 없습니다.** 검증된 방법은 한 가지입니다.

   ```bash
   python validate_g1j_sft_protocol.py   # CUDA 필요. 4개 행으로 2 step 학습한 adapter를 위 폴더에 저장
   ```

   이 adapter는 아주 조금 학습된 상태이고, 이 스크립트는 `g1j_1.5b_sft_protocol_validation.json`도 덮어씁니다. 학습이 안 된 깨끗한 초기값(`B=0`)을 만드는 스크립트는 아직 없습니다. 어느 쪽이든 rank 8 / alpha 16, 텐서 192개(파라미터 3,145,728개)여야 coordinator가 받아 줍니다. `--resume` / `--resume-from-hf`로 시작할 때는 체크포인트에서 가중치를 읽으므로 이 폴더에서는 `adapter_config.json`만 있으면 됩니다.

### 1. coordinator 실행

```bash
export RWKV_PROTOCOL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

python g1j_dist_coordinator.py \
  --units runs/g1j_local_protocol/units.json \
  --model-dir /workspace/RWKV7-G1j-1.5B-20260831 \
  --run-dir runs/my-first-run \
  --units-per-step 2
```

- `--run-dir`은 **새 폴더**를 쓰세요. 기본값(`runs/g1j_local_protocol/round-0001`)은 저장소에 커밋된 옛 실행 결과가 들어 있는 폴더라서 덮어쓰게 됩니다.
- 샘플 4개 unit은 `--units-per-step 2`면 2 step(unit 2개씩, loss token 48 / 142)이 되고, 옵션을 생략하면 190 loss token짜리 1 step이 됩니다.
- 시작하면 JSON 한 덩어리를 출력합니다(`run_id`, 계획된 step 수, 데이터/코드 해시 등). 기본은 `127.0.0.1:17821`에서 대기합니다.

### 2. worker 실행 (GPU 머신, 같은 토큰)

```bash
export RWKV_PROTOCOL_TOKEN=...   # coordinator와 같은 값
python g1j_dist_worker.py \
  --server http://127.0.0.1:17821 \
  --worker-id gpu-a \
  --model-path /workspace/RWKV7-G1j-1.5B-20260831
```

worker는 모델을 한 번 올린 뒤 join → lease → 계산 → 결과 반환을 반복하고, 학습이 끝났다는 신호를 받으면 종료합니다. 여러 대를 붙이려면 `--worker-id`만 다르게 하면 됩니다. (한 GPU에는 1.5B worker를 하나만 올릴 수 있었습니다.)

### 3. 진행 확인과 종료

```bash
curl -s -H "Authorization: Bearer $RWKV_PROTOCOL_TOKEN" http://127.0.0.1:17821/round
tail -f runs/my-first-run/round_log.jsonl
```

- 계획된 step이 모두 닫히면 coordinator가 자동으로 "학습 완료"를 표시하고 worker들이 종료합니다. coordinator 프로세스는 계속 떠 있으니 `Ctrl+C`로 끄세요. `--manual-training-complete`를 쓰면 마지막 round 이후에 `POST /training-complete`를 직접 보내야 완료됩니다.
- 결과 adapter 확인: `python validate_g1j_dist_checkpoint.py --run-dir runs/my-first-run`

## 학습 계획 정하기

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--epochs` | 1 | unit manifest를 몇 번 반복할지. 반복된 unit은 `epoch-0001/<id>`처럼 이름이 붙어 서로 다른 unit이 됩니다 |
| `--units-per-step` | 0 | optimizer step 하나에 쓰는 unit 수. 0이면 epoch 전체가 한 step |
| `--min-units-per-step` | 2 | step을 닫기 위해 필요한 최소 인정 unit 수(step의 unit 수가 더 적으면 그 수) |
| `--learning-rate` | 1e-4 | AdamW 학습률. weight decay 0, gradient clipping 없음 |
| `--lease-seconds` | 900 | lease 유효 시간. worker가 heartbeat로 갱신하고, 끊기면 만료 뒤 unit이 다시 발급됩니다 |

- 한 step은 그 step의 **모든 unit이 인정돼야** 닫힙니다(목표 loss token이 그 step unit들의 합이라서).
- `--units-per-step`은 학습 설정이기도 하고 손실 상한이기도 합니다. 값이 작으면 업데이트가 잦고 한 번에 평균 내는 토큰이 적어집니다. 동시에 coordinator가 죽었을 때 날아가는 작업도 그 step 하나로 줄어듭니다(0이면 epoch 하나만큼). 학습 쪽을 먼저 고려해서 정하세요.
- worker는 round 중간에 합류하면 다음 round가 열릴 때까지 대기합니다.
- 각 unit은 `input_ids`, `labels`(프롬프트는 -100), `loss_tokens`, `total_tokens`, 내용 해시를 가진 사전 토큰화 데이터입니다. manifest 형식(`rwkv-token-units-v1`)은 `prepare_g1j_protocol_units.py`를 참고하세요. 이 스크립트는 4개 행짜리 샘플만 만들며, 일반 데이터셋용 준비 도구는 아직 없습니다.

## 재시작과 resume

coordinator는 optimizer step이 끝날 때마다 `<run-dir>/ckpt/theta-NNNNNN/`에 어댑터, AdamW 상태, 해시 정보를 저장합니다(임시 폴더에서 완성한 뒤 한 번에 rename하므로 중간에 죽어도 깨진 버전이 보이지 않습니다).

```bash
# 같은 옵션으로 --resume만 추가해서 다시 실행
python g1j_dist_coordinator.py ...(원래 옵션 그대로)... --resume
```

- 가장 최근에 검증을 통과하는 체크포인트에서 이어 갑니다. 최신 것이 깨졌으면 이전 것으로 내려가며 경고를 출력합니다.
- **원래 실행과 데이터, round 계획(`--epochs`, `--units-per-step`), 학습률, 모델이 달라지면 거부**합니다. 이미 끝난 run도 거부합니다.
- `--resume` 없이 이미 체크포인트가 있는 `--run-dir`로 시작하면 에러입니다.
- **진행 중이던 round는 버리고 그 round를 처음부터 다시 시작합니다.** 이미 받은 gradient는 쓰지 않습니다.
- `run_id`가 유지되므로 돌고 있던 worker는 알아서 재접속합니다(최대 3시간 대기).
- `--keep-checkpoints N`(기본 3)으로 보관할 버전 수를 정합니다.

## Hugging Face push와 Hub에서 resume

```bash
export HF_TOKEN=hf_...   # 쓰기 토큰. coordinator에서만 사용합니다
python g1j_dist_coordinator.py ... \
  --push-adapter-to-hf --hf-adapter-repo USER/rwkv7-g1j-adapter
```

- 각 버전이 `checkpoints/<run_id>/theta-<version>/`에 올라갑니다(PEFT 설정/가중치, `adapter_metadata.json`, 조건에 따라 `optimizer.safetensors`). 기본적으로 10 step마다(`--hf-push-every-steps`)와 마지막 버전을 올립니다.
- 업로드는 백그라운드 큐에서 하고 실패하면 3번 재시도합니다. 상태는 `coordinator_state.json`의 `hub_push`에 기록되고, `POST /hub-push`로 실패한 버전을 다시 시도할 수 있습니다.
- repo는 기본이 **private**이며, 이미 public인 repo는 `--hf-adapter-public`을 줘야만 씁니다.
- **optimizer 상태 올리기 (`--hf-push-optimizer`).** Adam 모멘트가 있어야 Hub에서 이어받아도 연속 실행과 같은 결과가 나옵니다. 모멘트는 학습 gradient에서 나온 값이라 어댑터보다 민감하게 다룹니다.
  - `auto`(기본): Hub가 repo를 private이라고 확인했을 때만 올립니다. public이거나 알 수 없으면 올리지 않습니다.
  - `always`: public repo에도 올립니다(`--hf-adapter-public`과 같이 쓰면 시작할 때 경고).
  - `never`: 어댑터만 올립니다.
- **Hub에서 이어받기:** 새 머신에서 새 `--run-dir`로 시작합니다.

  ```bash
  python g1j_dist_coordinator.py ...(원래 옵션)... \
    --resume-from-hf --hf-adapter-repo USER/rwkv7-g1j-adapter --hf-run-id <run_id> [--hf-version N]
  ```

  `--hf-version`을 생략하면 검증을 통과하는 가장 최신 버전을 씁니다. optimizer 상태 없이 올라간 체크포인트는 어댑터만 복원하고 모멘트가 새로 시작된다는 경고를 출력하며, 이후 step은 연속 실행과 달라집니다. 이 기능 이전에 올린 체크포인트는 resume 메타데이터가 없어 거부됩니다. 해시는 손상된 업로드는 잡지만 repo에 쓸 수 있는 사람의 바꿔치기는 못 잡으니, 직접 관리하는 repo에서만 이어받으세요.
- 로컬 `--resume`으로 이어받을 때는 시작 시 Hub와 대조합니다. Hub에 이미 있는 버전은 다시 올리지 않고, 올라가지 못한 큐 항목은 다시 올립니다.

## coordinator 옵션 요약

| 옵션 | 설명 |
| --- | --- |
| `--units PATH` (필수) | unit manifest |
| `--model-id`, `--model-revision` | 코드에 고정된 G1j 1.5B 값과 같아야 합니다(다르면 시작을 거부) |
| `--token` | 프로토콜 토큰(기본 `$RWKV_PROTOCOL_TOKEN`) |
| `--bind`, `--port` | 기본 `127.0.0.1`, `17821`. 다른 주소에 바인딩하면 평문 HTTP 경고가 나옵니다 |
| `--model-dir`, `--fetch-model-from-hf`, `--hf-cache-dir` | base 모델 위치. coordinator는 `release-manifest.json`만 필요합니다 |
| `--initial-adapter-dir` | 초기 adapter 폴더(위 "준비물" 참고) |
| `--run-dir` | 산출물 폴더 |
| `--manual-training-complete` | 마지막 round 후 `POST /training-complete` 대기 |
| `--keep-checkpoints N` | 보관할 resume 체크포인트 수(기본 3) |
| `--keep-gradient-rounds N` | 닫힌 round 중 unit별 gradient 파일을 보관할 개수. -1(기본)은 전부 보관. 파일 하나가 약 12 MiB입니다 |
| `--resume`, `--resume-from-hf`, `--hf-run-id`, `--hf-version` | 위 resume 절 참고 |
| `--push-adapter-to-hf`, `--hf-adapter-repo`, `--hf-adapter-public`, `--hf-push-every-steps`, `--hf-push-optimizer`, `--hf-token` | 위 Hub 절 참고 |

## worker 옵션 요약

| 옵션 | 설명 |
| --- | --- |
| `--server` | coordinator 주소(기본 `http://127.0.0.1:17821`) |
| `--token` | 기본 `$RWKV_PROTOCOL_TOKEN` |
| `--worker-id` | 식별자(기본 `worker-<pid>`). 동시에 여러 worker를 쓸 때는 서로 다르게 |
| `--model-path`, `--fetch-model-from-hf`, `--hf-cache-dir` | base 모델 위치. worker는 HF 쓰기 토큰 옵션이 없고 다운로드도 비인증입니다 |
| `--max-units` | 처리할 unit 수 상한. 0은 학습이 끝날 때까지 |
| `--poll-seconds` | 대기 중 폴링 간격(기본 1초) |
| `--coordinator-down-timeout-seconds` | coordinator가 끊긴 상태로 기다릴 최대 시간(기본·최대 10800초) |

`Ctrl+C`/SIGTERM을 받으면 지금 잡고 있는 lease를 바로 반납하고 종료합니다. 갑자기 죽으면 lease가 만료될 때까지(기본 15분) unit이 묶여 있다가 다른 worker에게 넘어갑니다.

## 산출물 (`--run-dir`)

| 경로 | 내용 |
| --- | --- |
| `adapter_model.safetensors`, `adapter_config.json` | 현재 adapter (PEFT 형식). `PeftModel.from_pretrained(base, run_dir)`로 로드 |
| `theta.safetensors` | coordinator 내부 이름의 같은 텐서 |
| `ckpt/theta-NNNNNN/` | resume용 체크포인트 |
| `coordinator_state.json` | 현재 round, 해시, 인정된 결과 요약, Hub push 상태 |
| `round_log.jsonl` | 이벤트 로그(join, lease, 결과 인정/거부, round 종료, resume 등) |
| `gradients/round-NNNNNN/` | 인정된 unit별 gradient (감사용. resume과 Hub push는 읽지 않습니다) |
| `hub_push_queue/` | 아직 올라가지 않은 Hub push |
| `hub_resume/` | `--resume-from-hf`로 내려받은 파일 |
| `data_manifest.json` | 사용한 unit manifest 사본 |
| `optimizer.pt` | 이전 형식의 optimizer 덤프(resume은 쓰지 않음) |

## HTTP 엔드포인트

모두 `Authorization: Bearer <token>`이 필요합니다. 운영에서 직접 쓸 것은 `GET /round`(현재 round 상태)와 `POST /hub-push`(실패한 push 재시도, Hub 사용 시) 정도이고, 나머지(`/health`, `/join`, `/lease`, `/snapshot`, `/heartbeat`, `/release`, `/result`, `/training-complete`)는 worker 프로토콜입니다.

## 테스트

```bash
python -m unittest discover -p "test_*.py"
```

GPU 없이 약 30초에 130여 개가 돕니다. `transformers`/`peft`는 가짜로 대체해서 worker 제어 흐름을 검증하고, coordinator는 실제 HTTP 서버를 띄워 합성 gradient로 여러 round, 재시작, Hub 왕복(가짜 Hub)까지 검증합니다. 실제 모델 계산, 실제 Hugging Face 서버 호출, 여러 GPU 동시 실행은 이 테스트에 포함되지 않습니다.

## 보안과 한계

- **평문 HTTP + 공유 토큰 하나.** TLS, worker별 인증, 서명이 없습니다. 기본 바인딩은 루프백이고, 외부 네트워크에 노출하지 마세요. 원격 worker는 TLS를 앞에 두는 구성이 먼저 필요합니다.
- 결과 검증은 형식, 유한성, 토큰 수, 해시까지입니다. 그럴듯한 가짜 gradient를 보내는 악의적 worker는 막지 못합니다(canary/감사 없음).
- HF 쓰기 토큰은 coordinator에만 넘기세요. worker에는 토큰 옵션이 없습니다.
- 모델과 설정이 고정돼 있습니다(G1j 1.5B, rank 8, attention LoRA). **QLoRA는 분산 worker에 연결돼 있지 않습니다**(worker는 BF16 base만 로드). 추가 special-token 행 학습도 없습니다.
- 진행 중인 round는 복구하지 않습니다(위 resume 절).
- 알려진 남은 작업(timeout 기반 부분 step 종료, 중복 lease, 동시 worker 검증, 분산 동등성 허용오차 등)은 `G1j_1.5B_Training_Protocol_Validation_2026-10-18.md`에 정리돼 있습니다.

## 문제 해결

| 메시지 | 원인과 조치 |
| --- | --- |
| `worker bundle hash does not match coordinator's pinned code` | worker와 coordinator의 `g1j_dist_worker.py`, `rwkv_dist_protocol.py`, `rwkv_dist_client.py`, `rwkv_hub.py`가 다릅니다. 같은 커밋을 쓰고, 이 파일들을 바꾼 뒤에는 coordinator와 worker를 **같이** 재시작하세요 |
| `coordinator run changed after reconnect` | worker가 이전 `run_id`를 기억하고 있습니다. 새 run(`--resume` 없이 시작)을 띄웠다면 worker도 재시작하세요 |
| `<run-dir>/ckpt already holds checkpoints; pass --resume ...` | 이어갈 거면 `--resume`, 새로 시작할 거면 다른 `--run-dir` |
| `cannot resume: checkpoint belongs to a different run configuration: ...` | 메시지에 나온 항목(데이터, plan, 학습률 등)을 원래 실행과 같게 맞추세요 |
| HTTP 401 `unauthorized` | 토큰이 다릅니다 |
| `local base weights do not match the coordinator's pinned manifest` | worker의 모델 스냅샷이 고정 revision과 다릅니다 |
| worker가 `waiting_next_round`만 출력 | 정상입니다. round 도중에 합류해서 다음 round를 기다리는 중입니다 |
| Hub push가 `failed` | `coordinator_state.json`의 `hub_push.versions`에서 오류를 확인하고 `POST /hub-push`로 재시도하세요. 토큰은 오류 메시지에서 가려집니다 |
