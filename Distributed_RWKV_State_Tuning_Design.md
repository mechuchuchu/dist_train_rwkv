# Distributed RWKV-7 State Tuning — 설계 초안

2026-10-18

동결된 RWKV-7 모델의 시작 recurrent state만 학습한다. 7B, 13.3B 등 지원 모델은 크기 label이 아니라 실제 checkpoint tensor shape로 설정한다. 각 독립 SFT 예제는 같은 학습 state에서 시작하고, worker는 작은 data unit의 loss-sum gradient를 coordinator에 반환한다. 기존 분산 soft-prompt 설계의 lease, round, canary, audit, checkpoint 흐름을 재사용한다.

현재 장비와 13.3B/7.2B LoRA QLoRA 실측은 [`VRAM_Baseline_2026-10-18.md`](VRAM_Baseline_2026-10-18.md)에 기록했다. 사용자가 지정한 7.2B checkpoint는 `rwkv7-g1k-7.2b-20260930-ctx25600.pth`이며 content SHA-256은 `0853d4ce2e4e0629762797d05036a84cc69fb2fb9eff5590dd22aeb518a4c842`다. 그 QLoRA profile도 State Tuning의 initial-state gradient 경로를 대신하지 않는다. 모든 모델 크기의 State Tuning은 해당 autograd backend와 checkpoint를 따로 profile한 뒤 admission한다.

HF 13.3B 설계의 hardware tier는 worker당 단일 GPU 기준이다. BF16 State Tuning reference는 H100 80GB급에서 시작하고, 48GB는 profile 후보로 둔다. 현재 12GB에서 통과한 짧은 NF4 QLoRA는 State Tuning admission 근거가 아니다. 자세한 sequence sweep과 VRAM 기준은 baseline의 [tier 계획](VRAM_Baseline_2026-10-18.md)을 따른다.

1.5B HF 체크포인트에서 실제 chunked `Rwkv7Cache` initial-state gradient를 eager 경로와 비교한 짧은 canary가 통과했다. 단일 프로세스 worker-gradient 비교와 제한 사항은 [`G1j_1.5B_Training_Protocol_Validation_2026-10-18.md`](G1j_1.5B_Training_Protocol_Validation_2026-10-18.md)에 기록했다. 이는 긴 문맥 State-Tuning admission이나 분산 round 구현을 검증하지 않는다.

## 1. 목표와 범위

### 목표

- base model weight는 고정하고 task별 시작 state만 학습한다.
- worker가 사라지거나 느려도 token budget 기준으로 round를 닫는다.
- 한 unit의 state gradient를 다른 worker 또는 coordinator가 재계산할 수 있게 한다.
- 결과 state를 RWKV-7 inference runner의 초기 state로 바로 불러올 수 있게 한다.

### v1 범위

- base model: 호환되는 RWKV-7 checkpoint를 고를 수 있다. `model_id`, immutable revision, weights SHA-256과 shape manifest를 고정한다. Hugging Face snapshot `BlinkDL/rwkv7-g1`, revision `2a17e911155a5be23a4ed2792e978c77038f14e2`의 13.3B 파일은 현재 조사한 reference checkpoint다.
- 정확성 기준 worker는 bf16이다. 양자화 worker는 같은 gradient 경로를 검증한 뒤 별도 backend로 추가한다.
- 학습 대상: layer별 WKV matrix의 초기 state를 저랭크로 표현한 파라미터와 추가 special token의 embedding/lm-head 행을 학습한다. 기존 `V`개 token의 embedding/head 행과 나머지 base weight는 고정한다. `V`는 선택 checkpoint tokenizer 및 embedding/head shape로 확인한다.
- attention 및 FFN의 `x_prev` 두 벡터는 0으로 고정한다. 이 벡터까지 학습하는 것은 후속 버전 범위다.
- 각 SFT 행은 독립 예제다. 한 행의 끝 state를 다음 행이나 다른 unit으로 넘기지 않는다.

### 양자화 가능성

- 현재 조사한 13.3B reference checkpoint와 SHA-256이 일치한다고 기록한 GGUF 변환본이 있다. 해당 repo에는 Q4_K_M(7.88 GiB), Q5_K_M(9.31 GiB), Q6_K(10.83 GiB), Q8_0(13.72 GiB)이 있다. 이는 이 checkpoint의 GGUF 파일 크기이며 다른 크기 모델에 적용되지 않는다. [RWKV7-G1k-13.3B-GGUF](https://huggingface.co/shoumenchougou/RWKV7-G1k-13.3B-GGUF), 원본은 [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)이다.
- RWKV 프로젝트의 [rwkv.cpp](https://github.com/RWKV/rwkv.cpp)는 RWKV-7의 INT4/INT5/INT8 추론을 지원한다. 따라서 저장/추론용 원본 양자화는 가능하다.
- GGUF/rwkv.cpp는 현재 `train_temp`가 쓰는 PyTorch autograd 경로에 직접 끼울 수 없다. state tuning은 frozen weight를 통과해 초기 state까지 gradient를 역전파해야 한다.
- 별도 경로로 [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT)는 RWKV-7 state tuning과 INT8/NF4 quantized training을 함께 지원한다고 문서화한다. 공개 VRAM 수치는 RWKV-7 3B까지이므로 이번 13.3B와 지정한 7.2B checkpoint의 메모리 요구량과 품질은 이 프로젝트의 실측 전까지 확정하지 않는다.
- quantized backend를 쓰더라도 추가 embedding/head 행 `E_new`, `H_new`와 recurrent state adapter `B`는 fp32 trainable parameter로 둔다. 양자화 대상은 frozen base weight이며, quantized base를 통과하는 autograd가 `B`와 새 embedding 행까지 gradient를 전달하는지 확인해야 한다.
- 권장 순서: bf16으로 gradient/equivalence 기준을 먼저 만든 뒤, RWKV-PEFT 계열의 autograd 지원 quantized backend를 붙여 INT8부터 비교하고 NF4를 추가한다. 한 round 안에서 서로 다른 quantization recipe를 섞지 않는다. GGUF는 inference/export 비교용으로 취급한다.

이 objective에서 학습된 state는 task 전체에 공유되는 초기 memory다. 각 예제의 prompt와 completion은 이 state에서 시작해 정상적으로 순차 처리한다.

별도 adapter 경로인 LoRA SFT는 [`Distributed_RWKV_LoRA_SFT_Design.md`](Distributed_RWKV_LoRA_SFT_Design.md)에 정리한다.

## 2. 코드와 체크포인트에서 확인한 사실

- `RWKV-v7/train_temp/src/dataset.py`는 연속 token 구간으로 `x = tokens[:-1]`, `y = tokens[1:]`를 만든다. SFT에서는 이 방식을 유지하되, assistant completion에만 label을 주는 mask가 필요하다.
- `train_temp/src/model.py`의 WKV custom autograd 함수는 현재 `r,w,k,v,a,b`만 받고, CUDA forward에서 각 sequence의 시작 state를 0으로 초기화한다. backward도 initial-state gradient를 반환하지 않는다. 따라서 현재 training path를 그대로 사용해 state를 학습할 수 없다.
- `RWKV-v7/rwkv_v7_demo_rnn.py`는 inference state의 layout을 보여준다. layer마다 순서대로 `att_x_prev [D]`, `att_wkv [H,N,N]` fp32, `ffn_x_prev [D]`를 둔다.
- 현재 reference snapshot의 state dict를 mmap으로 읽어 확인한 구조는 `L=61`, `D=4096`, `H=64`, `N=64`, `V=65,536`이며 base weight는 bf16이다. 파일 크기는 26,540,868,485 bytes다. Worker는 이 값을 다른 checkpoint에 재사용하지 않고 각 선택 checkpoint에서 읽는다.
- 기존 soft-prompt 설계의 immutable token shards, loss-token budget, coordinator-only optimizer, canary/audit 및 round log는 그대로 적용할 수 있다.

## 3. 학습 파라미터 표현

### v1: 저랭크 WKV initial state

layer `l`, head `h`의 초기 WKV state를 다음과 같이 만든다.

```text
A[l,h] : [N, R]  — 고정된 deterministic orthogonal basis
B[l,h] : [R, N]  — 학습되는 fp32 parameter
S0[l,h] = A[l,h] @ B[l,h]
```

권장 시작 rank는 `R=8`이다. `A`는 base hash와 seed에서 재생성하며, coordinator와 worker가 같은 basis를 만든다. `B`는 0으로 초기화해 시작 state가 기존 zero-state 동작과 일치하게 한다. `A`는 고정하므로 worker가 보내는 파라미터와 gradient는 `B`뿐이다.

현재 13.3B reference의 `L=61,H=N=64,R=8`에서 `B` 크기는 `L×H×R×N = 1,998,848` fp32 값이다. state와 unit gradient는 각각 약 **7.63 MiB**다. 지정한 7.2B checkpoint는 `L=32,H=N=64`라 `B`가 1,048,576 fp32 값, 약 **4 MiB**다. 다른 checkpoint에서는 해당 shape의 `L,H,N`으로 다시 계산한다. 모든 WKV matrix를 dense fp32로 학습할 때 gradient 크기는 `4×L×H×N×N` bytes이고, `x_prev` 두 벡터까지 학습하면 실제 hidden width에 따라 추가된다. 저랭크 표현은 전송량을 줄이고 inference 시 dense `[L,H,N,N]` state로 복원한다.

**Dtype invariant:** 학습·추론에서 WKV recurrent matrix state `att_wkv/S0`와 저랭크 `A/B`, state-gradient 및 누적은 항상 FP32다. BF16 base weight나 NF4 frozen linear를 사용해도 이를 BF16/NF4로 cast하지 않는다. Upstream RNN demo에서도 `att_wkv`는 `torch.float`로 생성되고 WKV update가 FP32로 계산된다. v1에서 학습하지 않는 `att_x_prev`와 `ffn_x_prev`는 별도이며, 현재 demo는 이 두 activation-history vector에 model `DTYPE`을 사용한다.

일반식은 다음과 같고 고정된 RWKV 크기 상수를 사용하지 않는다.

```text
B parameter count = L × H × R × N
E_new/H_new count = 2 × M × D
fp32 trainable gradient bytes = 4 × (L×H×R×N + 2×M×D)
WKV backward scratch ≈ 4 × B_micro × H × T × (N²/16 + N) bytes
```

여기서 `L,D,H,N,V`와 embedding/head shape는 선택 checkpoint에서 추출한다. `T`는 현재 fused kernel의 16-token chunk 배수다. 식은 payload/scratch 추정이며 activation 및 loss workspace를 포함하지 않는다.

저랭크는 full-state objective와 다른 parameterization이다. rank를 높이거나 dense state를 학습하는 비교 실험으로 성능 한계를 확인한다.

### 추가 special token의 embedding 및 lm-head 행

학습 파라미터 전체는 `θ = {B, E_new, H_new}`로 둔다. `B`는 위의 저랭크 WKV state adapter다. `M`개의 추가 토큰에 대해 `E_new`와 `H_new`는 각각 `[M,D]` fp32 trainable matrix이며 `D`는 checkpoint에서 읽는다. 선택 checkpoint의 `emb.weight`와 `head.weight`가 별도 matrix인지 또는 tied인지 확인한다. 이 설계는 별도 행을 전제하므로 tied checkpoint는 별도 구현/profile이 필요하다. 기존 vocab 행과 나머지 base weight는 frozen 상태다.

- 기존 tokenizer의 token ID `[0,V)`는 유지하고 special token ID를 `[V,V+M)`에 append한다. tokenizer 파일/revision, token 문자열과 ID의 순서, 추가 토큰 목록 및 hash를 run manifest에 고정한다. 모든 worker는 같은 확장 tokenizer와 그 tokenizer로 미리 만든 immutable token shard를 사용한다.
- PyTorch 구현은 base embedding/head를 늘려 전체 행을 학습 대상으로 등록하지 않는다. 입력 ID가 `V`보다 작으면 frozen `emb.weight` lookup을 쓰고, 추가 ID면 해당 `E_new` 행을 lookup한다. 기존 head에 입력되는 최종 표현 `z`로 frozen base logits `[B,T,V]`와 `z @ H_new.T` logits `[B,T,M]`를 계산해 이어 붙이고, 확장 vocab 기준 CE를 계산한다. 새 행은 fp32 master parameter로 보관하고 model compute dtype으로 cast한다.
- 초기화 제안: 토큰 문자열을 **확장 전 tokenizer**로 인코딩한 기존 token ID들의 embedding 행 평균으로 `E_new`를, 같은 ID들의 lm-head 행 평균으로 `H_new`를 초기화한다. 빈 분해나 특수한 제어 문자열은 norm을 맞춘 deterministic random 초기값을 사용하고 manifest에 방법/seed를 기록한다.
- 토큰이 모델 출력으로 사용될 예정이면 해당 token ID가 assistant target label에 실제 포함되어야 `H_new`가 긍정적인 target 학습 신호를 받는다. 입력 전용 marker라면 `E_new`는 학습되지만 `H_new`는 softmax denominator에 의한 억제 신호만 받으므로 출력용으로 간주하지 않는다.
- 추가되는 파라미터와 gradient 전송량은 fp32 기준 `2 × M × D × 4` bytes다. 현재 `D=4096` reference에서는 토큰당 32 KiB, `M=16`이면 512 KiB다. 이는 현재 checkpoint의 예시이며 수치는 optimizer state를 제외한 payload다.

Coordinator는 `B`, `E_new`, `H_new`를 같은 token-weighted loss와 optimizer step으로 갱신한다. worker는 세 항목의 summed gradient를 반환하며 optimizer state는 보유하지 않는다.

### State 초기화 및 데이터 경계

- 모든 독립 행은 같은 `B`에서 시작한다. 같은 unit의 batch 안에서도 state는 행마다 독립 복사본으로 broadcast한다.
- token 0 직전의 `att_x_prev`와 `ffn_x_prev`는 0이다. v1에서는 학습하지 않는다.
- 긴 문서를 여러 행으로 나눌 때 각 행은 같은 learned initial state로 시작한다. 앞 chunk의 최종 state를 다음 chunk로 연결하지 않는다. 연결하면 unit 순서와 lease가 objective에 영향을 줘 병렬 gradient 합산이 성립하지 않는다.
- inference에서는 export된 `S0`를 각 layer의 WKV 초기 state로 넣고, x-prev 두 항목은 0으로 만든다.

## 4. Worker 계산

1. Worker는 고정된 base checkpoint, code hash, 확장 tokenizer 및 token-to-ID hash, token shard와 round의 `θ` hash를 확인한다.
2. frozen base model을 `eval()`로 실행한다. base weight는 gradient 대상에서 제외하지만, loss에서 `B`까지의 autograd 경로는 유지한다.
3. 각 예제의 token sequence를 `B`가 만든 초기 WKV state로 독립 실행한다. 추가 token ID는 `E_new`에서 lookup하고, 출력은 base logits와 새 token logits를 이어 붙인다. assistant label만 계산하고 나머지는 `-100`으로 mask한다.
4. `cross_entropy(..., reduction="sum", ignore_index=-100)`으로 `loss_sum`을 계산하고, `∂loss_sum/∂{B,E_new,H_new}`를 fp32로 구한다. unit 안의 micro-batch gradient는 나누지 않고 더한다.
5. `loss_tokens`와 함께 세 trainable tensor의 summed gradient를 반환한다. Worker는 optimizer state를 보유하지 않는다.

`train_temp`는 현재 `l2wrap_cross_entropy`를 사용한다. v1은 분산 SFT 설계와 일치하도록 plain summed CE를 기준으로 한다. L2Wrap을 쓰려면 worker, verifier, equivalence 기준에 동일한 loss와 custom backward를 적용해야 한다.

## 5. Kernel 및 모델 변경

Production worker는 inference demo의 token-by-token Python loop를 직접 학습 경로로 쓰지 않는다. 새 kernel을 쓰기 전에 upstream이 안내하는 [RWKV-CUDA state-tuning/state-passing kernels](https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused)와 [RWKV-PEFT state-tuning 경로](https://github.com/Joluck/RWKV-PEFT)를 먼저 비교한다. 기존 코드를 이 분산 protocol에 맞게 적용할 수 없을 때 `train_temp` fused WKV kernel을 확장한다.

- CUDA forward에 FP32 `initial_state [B,H,N,N]`를 입력해 첫 chunk의 state로 사용한다. Kernel에서 state를 다른 dtype으로 downcast하지 않는다.
- custom autograd backward가 `d_initial_state`를 반환하게 한다. 같은 `B`가 batch의 여러 행에 broadcast되므로 행별 gradient를 합산한 뒤 `dB = Aᵀ · dS0`를 계산한다. 입력 lookup과 확장 vocab logits는 별도 PyTorch 연산으로 구현해 `E_new`, `H_new`에 gradient를 전달한다.
- loss와 gradient는 base vocab `V`개와 추가 token을 합친 확장 vocab 기준이어야 한다. 새 token row를 출력 logits에만 추가하고 target ID remapping을 빠뜨리지 않는지 검사한다.
- kernel의 현재 zero initialization은 기본값으로 유지해 기존 pretraining 코드와 호환한다.
- v1은 WKV initial state만 바꾸므로 time-mix와 channel-mix의 첫-token shift는 기존 zero padding을 유지한다.
- 기준 backend는 BF16 base weight를 쓰고 WKV state `att_wkv/S0` 계산/누적, `A/B` master copy 및 gradient는 FP32로 둔다. QLoRA/다른 quantized backend에서도 이 state dtype invariant는 유지하고, base quantization recipe와 별도 필드인 `state_dtype="fp32"`를 worker manifest에 고정한다.
- `ctx_len`은 positional embedding의 고정 상한이 아니라 한 training sequence의 길이 설정이다. 현재 reference는 ctx 25,600으로 학습됐다. 각 checkpoint에서 확인한 training context로 길이 sweep을 구성한다. `train_temp`는 `args.ctx_len`을 바꿀 수 있지만 fused WKV kernel의 현재 길이는 16의 배수여야 하며, training context를 넘는 품질은 보장되지 않는다 ([RWKV-LM](https://github.com/BlinkDL/RWKV-LM), [context-length discussion](https://github.com/BlinkDL/RWKV-LM/discussions/179)).
- Context admission은 checkpoint별 짧은 길이부터 시작한다. 현재 13.3B LoRA QLoRA에서 3,072 OOM 및 7.2B에서 4,096 통과로 측정한 길이 경계는 initial-state gradient를 포함하지 않으므로 State Tuning에 적용하지 않는다. State backend가 준비되면 짧은 context에서 `d_initial_state` 경로를 profile하고, 통과한 micro-batch로 `1024/2048 → 3072/4096 → 8192 → 16384 → checkpoint training context`처럼 늘린다. 각 단계에서 forward/backward/optimizer step과 state-gradient finite 여부를 기록한다. 32k 이상은 더 큰 VRAM worker, 긴 의존성 데이터와 별도 품질 검증이 있을 때만 연다. 독립 row 사이 state는 계속 reset한다.
- WKV backward scratch는 `s[B,H,T/16,N,N]`와 `sa[B,T,H,N]` fp32 buffer 기준 `4×B_micro×H×T×(N²/16+N)` bytes다. 현재 reference의 `H=N=64`에서는 row당 약 80 KiB/token, `T=25,600`에서 약 1.95 GiB다(activation/workspace 제외). 각 모델은 실제 `H,N`으로 계산하고 gradient checkpointing과 낮은 동시 micro-batch를 적용한다. Effective token batch 증가는 admission된 micro-batch에 local accumulation을 더해 해결한다.
- 현재 13.3B BF16 weight만 약 24.7 GiB여서 이 12 GiB GPU에 맞지 않는다. 다른 크기 모델의 admission은 weight format estimate가 아닌 end-to-end peak로 결정한다. 긴 sequence에서는 initial-state gradient norm/finite 여부를 길이별로 기록한다.

필수 correctness gate는 짧은 sequence에서 dense PyTorch reference가 계산한 `dS0`와 CUDA custom backward의 `dS0`가 일치하는지 확인하는 것이다.

## 6. Data unit과 round

- Coordinator가 chat template, 확장 RWKV tokenizer, label mask를 한 번 적용하고 token-id shard를 commit hash로 고정한다. 기존 vocab의 ID가 그대로인지, 각 special token 문자열이 단일 추가 ID로 인코딩되는지 확인한다.
- 생성 대상 special token은 assistant target에 포함한다. tokenizer/label-mask 전처리 후에도 의도한 target 위치가 유효한 loss label로 남는지 확인한다.
- 한 row는 독립 대화/예제다. unit manifest에는 row 범위, 전체 token 수, loss token 수, shard hash가 들어간다.
- unit 총량은 우선 **16k–64k total tokens**를 목표로 두되 model/backend의 처리시간에 따라 조정한다. 각 unit은 여러 짧은 row 또는 하나 이상의 긴 row로 구성할 수 있고, `max_len`은 unit budget과 별도로 해당 checkpoint profile을 통과한 context까지 허용한다. Single-GPU에서 context bucket별 처리시간을 측정한 뒤 volunteer worker admission 한계를 정한다.
- round는 `B`의 token budget을 채우거나 hard timeout 조건을 만족하면 닫는다. 기존 lease, heartbeat, duplicate suppression, speculative re-issue는 유지한다.
- round마다 coordinator가 `B`에 Adam update를 적용한다. worker는 해당 round의 `B`만 읽고 gradient만 반환한다.

unit `i`가 loss-sum gradient `G_i`와 loss token 수 `n_i`를 반환하면, clipping 전 batch-equivalent gradient는 다음과 같다.

```text
G = (Σ_i G_i) / (Σ_i n_i)
```

이 등식이 single large batch와 같은 결과를 주는 것은 clipping/trimmed mean 전까지다. Byzantine 대응으로 per-unit clipping 또는 trimmed mean을 적용하면 raw batch gradient와 달라지는 것은 의도된 robust aggregation으로 기록한다. State gradient norm과 기존 soft-prompt의 `1.0` global clip은 스케일이 다르므로 임계값을 그대로 복사하지 않고 single-machine calibration으로 정한다.

## 7. 메시지와 저장 형식

Round descriptor는 기존 `theta_*` 항목을 state adapter 정보로 확장한다.

```json
{
  "round_id": 1,
  "state_version": 0,
  "state_hash": "sha256:...",
  "state_url": ".../state_0.safetensors",
  "state_schema": "rwkv7_state_and_token_rows_v1",
  "state_dtype": "fp32",
  "rank": 8,
  "basis_seed": 12345,
  "trainable_tensors": ["B", "E_new", "H_new"],
  "base_vocab_size": "V from checkpoint",
  "model_shape": {"n_layer": "L", "n_embd": "D", "n_head": "H", "head_size": "N"},
  "added_tokens": [
    {"text": "<|example|>", "id": 65536}
  ],
  "tokenizer_hash": "sha256:...",
  "base": {
    "repo": "BlinkDL/rwkv7-g1",
    "revision": "2a17e911155a5be23a4ed2792e978c77038f14e2",
    "weights_sha256": "...",
    "dtype": "bf16"
  },
  "max_len": "admitted context for this checkpoint/backend",
  "target_loss_tokens": 300000,
  "min_units": 8,
  "hard_timeout_s": 600
}
```

Result metadata는 `unit_id`, `lease_id`, `state_version/hash`, `loss_tokens`, `loss_sum`, `grad_dtype=fp32`, `grad_sha256`, `worker_id`, signature를 포함한다. Binary payload는 이름 있는 tensor `B`, `E_new`, `H_new`의 summed fp32 gradient를 담는다.

Coordinator checkpoint에는 `B`, `E_new`, `H_new`, Adam state, basis seed, 추가 token 문자열/ID, tokenizer hash, base revision/hash, data commit, worker code hash, round log, validation 결과를 저장한다. Inference export에는 optimizer state 없이 dense initial WKV state와 추가 embedding/head 행을 저장한다. Export loader는 추가 token ID를 사용하는 확장 tokenizer와 row tensor를 함께 로드하고, 원본 체크포인트를 수정하지 않는다.

## 8. 검증과 악성 gradient 대응

- **Admission:** worker가 고정 unit을 계산한다. state gradient shape, finite 값, base hash, custom-kernel parity와 loss를 검사한다.
- **Token rows:** 확장 tokenizer에서 기존 ID 보존, 새 토큰의 단일 ID encoding, input occurrence에 대한 `E_new` gradient, 출력 target occurrence에 대한 `H_new` gradient가 정상인지 확인한다. 모든 trainable gradient가 finite인지 함께 검사한다.
- **Canary:** 일부 unit은 coordinator 내부에서 reference gradient를 계산한다. Canary의 정체는 worker에 노출하지 않는다.
- **Audit:** 낮은 trust worker는 다른 worker 또는 trusted verifier에서 재계산한다. BF16/CUDA 차이에 대한 cosine 허용치는 정상 worker 표본으로 보정한다.
- **Validation:** 고정 held-out SFT set의 loss와 task metric은 trusted evaluator가 계산한다. worker 보고 loss로 rollback하지 않는다.
- **Rollback/log:** best `B` 및 optimizer checkpoint를 보관하고, state version별로 사용 unit, gradient hash, optimizer step, validation을 append-only JSONL에 남긴다.
- Coordinator가 base model을 실행할 GPU를 갖지 않으면 coordinator 단독 canary/validation이 불가능하다. 이 경우 trusted verifier worker를 별도 구성하고 해당 역할을 run manifest에 기록한다.

## 9. 단계별 진행

1. **Reference:** 작은 RWKV-7 모델과 짧은 sequence에서 PyTorch recurrence로 initial WKV gradient를 확인한다.
2. **Kernel parity:** fused CUDA kernel에 initial state 입력 및 gradient 반환을 추가하고 forward/loss/gradient를 reference와 비교한다.
3. **7.2B checkpoint profile:** 지정한 `rwkv7-g1k-7.2b-20260930-ctx25600.pth` (`L=32,D=4096,H=N=64,V=65,536`, BF16; SHA-256은 위 manifest 값)을 기준으로 State Tuning autograd backend의 짧은 context forward/backward/optimizer step과 `d_initial_state`를 측정한다. 통과한 micro-batch로 시작해 ctx 25,600까지 단계별로 profile한다.
4. **13.3B reference profile:** 7B에서 parity를 확보한 동일 state backend를 13.3B에 적용한다. LoRA profile 수치를 재사용하지 말고 `B`, `E_new`, `H_new`의 peak VRAM, tokens/sec, rank, batch/accumulation, context, zero-state 비교와 held-out 품질을 새로 측정한다.
5. **Unit equivalence:** 같은 batch를 단일 실행과 여러 unit으로 나눠 계산한다. clipping off 상태에서 token-weighted summed gradient가 tolerance 내에서 일치해야 한다.
6. **Local fault simulation:** lease 만료, duplicate, stale state version, forged gradient, delayed worker와 rollback을 확인한다.
7. **Volunteer rollout:** 소수의 admission-passed worker에서 canary/audit pass rate와 round latency를 측정한 뒤 참가자를 늘린다.

## 10. 미결정 사항

- 하나의 task-wide state를 학습할지, task/persona별 state를 여러 개 둘지.
- rank 4, 8, 16 간 품질과 bandwidth 비교.
- WKV state만으로 성능이 충분한지, 후속 버전에서 attention/FFN `x_prev`를 학습할지.
- tokenizer와 SFT dataset/mix, validation set 및 evaluation metric.
- 추가할 special token의 문자열/개수와 입력 전용인지 생성 대상인지.
- 7.2B 및 13.3B 각 model/backend profile의 worker 최소 VRAM, unit token 수와 round budget.
- 신뢰할 수 있는 coordinator/verifier GPU를 어디에 둘지.

## 참고 파일

- `RWKV-LM/RWKV-v7/train_temp/src/model.py`
- `RWKV-LM/RWKV-v7/train_temp/src/dataset.py`
- `RWKV-LM/RWKV-v7/rwkv_v7_demo_rnn.py`
- `dist_train_rwkv/Distributed_Soft_Prompt_SFT_Design_Doc.md`
