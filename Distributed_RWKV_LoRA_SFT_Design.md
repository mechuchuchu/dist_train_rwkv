# Distributed RWKV-7 LoRA / QLoRA SFT — 설계 초안

2026-10-18

이 문서는 7B, 13.3B 등 RWKV-7 checkpoint 크기를 선택해 frozen base에 LoRA/QLoRA와 추가 special-token embedding/lm-head 행을 학습하는 분산 SFT 설계다. Coordinator protocol은 [`Distributed_Soft_Prompt_SFT_Design_Doc.md`](Distributed_Soft_Prompt_SFT_Design_Doc.md)를 재사용하고, 각 worker 설정은 선택한 checkpoint의 실제 tensor shape에서 생성한다. State tuning 설계는 [`Distributed_RWKV_State_Tuning_Design.md`](Distributed_RWKV_State_Tuning_Design.md)를 참고한다.

현재 RTX 3060에서 지정 13.3B와 7.2B checkpoint의 full-model QLoRA 결과는 [`VRAM_Baseline_2026-10-18.md`](VRAM_Baseline_2026-10-18.md)에 기록했다. FP32 master LoRA 기준 13.3B all-linear rank-8은 2,048 tokens/step까지 통과하고 3,072에서 OOM, 7.2B는 4,096 tokens/step 조합까지 통과했다. 13.3B의 2,048은 peak reserved 여유가 약 0.64 GiB뿐이므로 실제 SFT admission 전에는 더 보수적으로 시작한다. 이 결과는 attention-only v1을 대신 측정한 것이 아니다. 선택한 model hash, target module, quant recipe 및 batch/context마다 별도 admission profile을 요구한다.

현재 HF Transformers attention-only 13.3B profile과 GPU VRAM tier 결정은 [`Distributed_RWKV_HF_Transformers_Design.md`](Distributed_RWKV_HF_Transformers_Design.md) 및 baseline의 [VRAM tier 계획](VRAM_Baseline_2026-10-18.md)에 정리했다. 기준 설계는 12GB에서 짧은 NF4 QLoRA, 13.3B BF16 LoRA의 H100 80GB reference worker이며, 실행 경로별 수치를 서로 대체하지 않는다.

## 0. 다중 checkpoint 지원 원칙

- 7B/13.3B는 모델 선택 label이며 코드는 고정된 layer 수나 폭을 쓰지 않는다. 실제 체크포인트에서 `L=n_layer`, `D=n_embd`, WKV `H=n_head`, `N=head_size`, 각 linear의 `in/out_features`, base vocab `V`, embedding/head shape 및 tied 여부, 학습 context metadata를 읽는다. 모델별 config, tokenizer와 training code가 호환되는 RWKV-7 계열인지도 확인한다.
- 한 run은 `model_id`, immutable revision, checkpoint SHA-256을 고정한다. worker admission 및 canary/audit은 label이 아닌 이 식별자와 shape manifest를 비교한다.
- LoRA와 token-row 크기는 tensor shape에서 산출한다. `4D` FFN, vocab 65,536, `L=61`, `D=4096`을 일반 규칙으로 두지 않는다.
- 7B급 checkpoint는 13.3B와 별도 프로파일 대상으로 다룬다. Parameter count가 작으면 weight와 activation 부담이 낮아질 수 있지만, 각 파일의 1-step profile이 끝나기 전 VRAM admission을 단정하지 않는다.
- 현재 profile 대상은 사용자가 지정한 `rwkv7-g1k-7.2b-20260930-ctx25600.pth`다. 파일 SHA-256 `0853d4ce2e4e0629762797d05036a84cc69fb2fb9eff5590dd22aeb518a4c842`, metadata shape `L=32,D=4096,H=64,N=64,V=65,536`, FFN width 16,384, BF16이다. 이 수치도 해당 파일 식별자를 위한 manifest 예시다.
- Context bucket은 checkpoint에서 확인된 training context 이내를 기본 sweep으로 잡는다. RTX 3060의 13.3B profile 결과를 7B나 다른 architecture의 길이 상한으로 복사하지 않는다.

## 1. 목표와 권장 v1 범위

- base checkpoint는 고정하고 LoRA adapter 및 추가 special token 행만 학습한다.
- v1 LoRA 대상은 각 layer의 time-mix `receptance`, `key`, `value`, `output` projection이다. Channel-mix LoRA는 custom CUDA kernel 경로를 수정해야 하므로 2단계 확장으로 둔다.
- 추가 special token은 새 ID로 append하고 그 token의 embedding 행과 lm-head 행을 모두 학습한다. 기존 vocab 행과 원본 checkpoint는 바꾸지 않는다.
- RWKV recurrent state는 원본 zero-state 동작을 유지한다. LoRA와 learned initial state를 동시에 넣는 hybrid는 별도 실험으로 둔다.
- LoRA 기준 worker는 bf16 base weight, fp32 LoRA master parameter/gradient를 쓴다. QLoRA worker는 NF4 frozen base와 bf16 compute를 쓰며, bf16 reference와 parity를 통과한 뒤 별도 backend로 허용한다.

## 2. 코드에서 확인한 모델 구조

구조 관찰과 수치 예시는 현재 지정한 13.3B checkpoint(61 layers, `D=4096`, vocab 65,536)에 대한 것이다. Worker는 각 checkpoint의 config/state dict에서 dimensions를 검증해 사용한다. `train_temp/src/model.py`의 time-mix `receptance`, `key`, `value`, `output`은 해당 모델의 실제 `D→D` projection이며, `RWKV7_CLAMPW_CUDA` 앞뒤 activation을 사용한다. 이 projection LoRA의 입력 gradient는 기존 fused WKV custom autograd 경로로 전달 가능하므로 WKV initial-state gradient 변경은 필요 없다.

Channel-mix는 `key: D→4D`, `value: 4D→D`지만 fast path가 두 `weight`를 `_CmixLayerV2Fn` custom CUDA op에 직접 넘긴다. 일반 `nn.Linear` wrapper만 씌워서는 LoRA branch가 호출되지 않는다. 1차 구현은 이를 건드리지 않고, 향후 eager PyTorch reference 또는 adapter 입력을 받는 fused kernel을 추가한다.

이 profile의 `emb.weight`와 `head.weight`는 vocab 65,536행이며 서로 별도 matrix다. 각 선택 checkpoint에서는 이 shape와 tying 여부를 검사한다. 새 token의 embedding/head 행은 기존 행과 별도 parameter다. model 안의 native `w1/w2`, `a1/a2`, `v1/v2`, `g1/g2`는 RWKV-7의 기존 저랭크 구조이며 이 설계의 LoRA 대상 projection과 혼동하지 않는다.

## 3. LoRA parameterization

frozen linear weight를 `W ∈ R[out,in]`이라 할 때 다음을 계산한다.

```text
 A ∈ R[r,in]          — trainable
 B ∈ R[out,r]         — trainable
 W_eff = W + (α/r) · B @ A
 linear(x) = x @ Wᵀ + (α/r) · (x @ Aᵀ) @ Bᵀ
```

기본값은 `rank r=8`, `alpha=16`, dropout 0이다. `A`는 Kaiming 계열 초기화, `B`는 0으로 초기화한다. BF16 LoRA는 첫 forward가 원본 checkpoint와 같고, QLoRA는 NF4로 양자화한 frozen base의 첫 forward를 기준으로 삼아 BF16 결과와 parity를 측정한다. LoRA factor는 fp32 master parameter로 저장하고 bf16 matmul에 맞춰 cast한다. QLoRA에서도 adapter와 special-token 행은 동일하게 fp32 master로 둔다. `A`와 `B`를 worker 사이에서 평균하지 않는다. 모든 worker가 같은 adapter snapshot에서 unit gradient를 계산하고 coordinator가 gradient만 모아 한 번 update한다.

### 대상 모듈과 통신량

| checkpoint | target | rank 8 trainable 값 | fp32 gradient payload |
| --- | --- | ---: | ---: |
| 7.2B (`L=32`) | attention-only v1 | 8,388,608 | 32 MiB |
| 13.3B (`L=61`) | attention-only v1 | 15,990,784 | 61 MiB |
| 7.2B (`L=32`) | attention + FFN all-linear | 18,874,368 | 72 MiB |
| 13.3B (`L=61`) | attention + FFN all-linear | 35,979,264 | 137.25 MiB |

표는 두 지정 checkpoint의 실제 shape에서 계산한 rank-8 예시다. 일반 계산은 각 target `nn.Linear`마다 `r×(in_features+out_features)`를 합산한다. 즉 `P_lora=Σ_m r(in_m+out_m)`이고 fp32 gradient payload는 `4×P_lora` bytes다. rank를 바꾸면 선형 비례한다. 이 payload는 unit의 summed gradient 크기이며 adapter snapshot 다운로드도 대략 같다. special token 행은 별도 추가한다.

각 special token `M`개에 대해 별도 fp32 행 `E_new,H_new ∈ R[M,D]`를 둔다. 기존 token ID `[0,V)`는 유지하고 신규 ID는 `[V,V+M)`에 append한다. 추가 크기는 parameter 또는 fp32 gradient 기준 `2×M×D×4` bytes다. 이번 `D=4096` checkpoint에서만 토큰당 32 KiB이며, 다른 폭이면 공식에 따라 달라진다.

- 입력 token ID `<V`는 frozen `emb.weight`에서 조회하고 추가 ID는 `E_new`에서 조회한다.
- 기존 `ln_out` 이후의 표현 `z`에 대해 base head logits와 `z @ H_new.T`를 이어 전체 확장 vocab의 CE를 계산한다.
- 생성 대상 special token은 assistant target label에 있어야 `H_new`가 긍정 target 신호를 받는다. 입력 전용 marker는 embedding 학습 대상이지만 head는 softmax denominator 억제 신호만 받는다.
- 각 행은 확장 전 tokenizer로 literal 문자열을 인코딩한 기존 ID들의 해당 embedding/head 행 평균으로 초기화한다. 빈 분해에는 norm-matched deterministic random 초기값을 쓰고 seed/방법을 기록한다.
- tokenizer revision, token 문자열/ID 목록과 순서, tokenizer hash, 전처리된 shard hash를 고정한다. 기존 vocab ID가 그대로인지, 새 문자열이 각 하나의 ID로 encode되는지 확인한다.

## 4. Worker 구현

### Attention projection

1. Checkpoint를 불러온 뒤 base parameter 전체를 `requires_grad=False`로 고정한다.
2. `att.receptance`, `att.key`, `att.value`, `att.output`을 `LinearWithLoRA` 형태로 감싼다. frozen `W`는 원본과 같은 연산을 하고, 저랭크 branch를 더한다.
3. 모델 전체 forward를 `torch.no_grad()`로 감싸지 않는다. base weight는 gradient 대상이 아니어도 autograd graph는 adapter와 `E_new/H_new`까지 이어져야 한다.
4. 현재 fused WKV op가 반환하는 `dr,dw,dk,dv,da,db`가 projection의 adapter branch까지 역전파되는지 짧은 reference에서 확인한다. WKV 초기 state는 zero로 두므로 state-gradient kernel 변경은 없다.

### Special-token 행과 loss

입력 embedding lookup은 base row와 `E_new`를 ID 범위에 따라 라우팅한다. `head.weight`는 고정하고, head 입력 `z`에서 확장 token logits를 별도 계산한다. 기존 logits를 잘라내거나 새 token을 output softmax에 누락하면 안 된다. Assistant completion label만 유효하게 두고 prompt 및 loss 제외 영역은 `-100`으로 mask한다.

### Channel-mix 확장

v1은 `ffn.key/value` LoRA를 비활성화한다. 이를 켜려면 `relu(key(x))²` 앞의 key LoRA와 value projection 뒤의 LoRA branch가 모두 custom op에서 계산되고, backward가 네 factor(`A_key/B_key/A_value/B_value`)의 gradient를 반환해야 한다. 우선 slow PyTorch 경로로 수치 기준을 만들거나 fused op의 forward/backward signature를 확장한다. Channel-mix branch가 실제 호출되지 않는 단순 module replacement는 허용하지 않는다.

## 5. 분산 학습 protocol

각 worker는 round의 고정 adapter `θ={A_{l,m},B_{l,m},E_new,H_new}`와 고정 unit을 사용한다. `eval()` 상태로 dropout 없이 실행하며, unit 내 여러 micro-batch의 summed CE gradient를 나누지 않고 합산한다.

```text
G_i = ∂loss_sum_i / ∂θ
n_i = labeled loss token count
round gradient = (Σ_i G_i) / (Σ_i n_i)
```

Coordinator만 fp32 Adam/AdamW와 optimizer state를 보유한다. factor별 gradient는 이름과 shape가 고정된 safetensors payload로 반환한다. 기존 protocol의 `θ_hash`, lease 중복 제거, stale 결과 거부, canary/audit, clipping 및 held-out rollback을 유지한다. Coordinator는 adapter factor를 평균하거나 서로 다른 round/version gradient를 섞지 않는다.

LoRA v1의 통신량은 soft prompt나 WKV low-rank state보다 훨씬 크다. rank-8 attention-only의 61 MiB gradient는 rank-8 WKV state adapter의 7.63 MiB보다 약 8배 크고, unit마다 업로드된다. 기존 설계의 8k–32k total-token unit보다 조금 키운 **16k–64k total tokens/unit**을 출발점으로 삼아 WAN 전송시간과 worker 처리시간을 함께 측정한다. 이는 unit 전체의 effective batch를 키운다는 뜻이다. `ctx_len`은 이 unit token budget과 별개다. GPU에 동시에 쌓는 micro-batch는 VRAM이 허용하는 크기로 유지하고, 나머지는 local gradient accumulation으로 합산한다.

### Context 길이

RWKV에는 transformer식 positional embedding의 고정 context limit이 없다. 공식 RWKV 자료도 이론상 긴 context 처리가 가능하다고 설명하지만, 실제 품질은 학습 context와 수치 정밀도의 영향을 받는다 ([RWKV-LM](https://github.com/BlinkDL/RWKV-LM), [context-length discussion](https://github.com/BlinkDL/RWKV-LM/discussions/179)). 지정 checkpoint의 `ctx25600`은 학습된 context 길이다. `train_temp`는 `args.ctx_len`을 늘리면 forward 상한을 올릴 수 있고, 현재 fused WKV kernel의 고정 chunk 길이는 16 token이다. 따라서 모든 sequence 길이는 16의 배수로 맞춘다.

- 목표 context 상한은 각 checkpoint의 config/model card에서 확인한 training context로 정한다. 이번 checkpoint는 25,600이며 bucket은 짧은 길이에서 시작해 `1024/2048` 등 실측에서 가능성이 확인된 길이를 포함하고 이후 `3072 → 4096 → 8192 → 16384 → 25600` 순으로 단계별 admission한다. 현재 RTX 3060의 all-linear 13.3B profile은 FP32 master adapter 기준 2,560까지 실행됐지만 여유가 거의 없고 3,072에서 OOM이다. 7.2B는 4,096까지 실행됐고 6,144는 여유가 거의 없다. 다른 모델은 별도 sweep을 한다. 모든 row 길이는 fused WKV 16-token 경계에 맞추고 독립 예제를 packing하지 않는다.
- 25,600 초과는 코드상 시도할 수 있지만 checkpoint의 학습 범위를 넘어선다. long-context SFT 데이터와 별도 held-out 검증이 있을 때 `32768`, 이후 `51200`을 exploratory run으로 추가한다. 길이만 늘리고 긴 의존성을 담은 학습 예제가 없으면 성능 개선을 가정하지 않는다.
- `train_temp`의 custom WKV backward는 fp32 `s[B,H,T/16,N,N]`와 `sa[B,T,H,N]` buffer를 만든다. 일반 scratch 산식은 약 `4×B_micro×H×T×(N²/16+N)` bytes다. 이번 checkpoint의 `H=N=64`에서는 row당 약 80 KiB/token, `T=25,600`이면 약 1.95 GiB이며 다른 모델은 실제 `H,N`으로 계산한다. 길이/동시 batch를 키우면 이 항도 비례해 커진다. Gradient checkpointing과 local gradient accumulation을 사용하되, accumulation은 한 번에 동시 실행하는 micro-batch의 peak를 낮게 유지한다.

현재 13.3B all-linear QLoRA profile은 FP32 master adapter로 같은 총 2,048 tokens에서 `B=1,T=2048`과 `B=2,T=1024`를 실행했다. 둘 다 통과했지만 peak reserved는 11,254 MiB (10.99 GiB)로 여유가 약 0.64 GiB다. 7.2B에서는 `B=1,T=4096`, `B=2,T=2048`, `B=4,T=1024`가 각각 통과했고 peak reserved는 약 9.11 GiB였다. 이들은 synthetic one-step 조건의 측정이며 추가 token rows/head loss를 포함하지 않는다. 더 큰 effective batch는 통과한 micro-batch에 gradient accumulation을 더해 늘리고, 더 큰 simultaneous batch는 checkpoint별 admission sweep으로 정한다.

1. rank 4와 attention-only 대상으로 gradient 품질과 전송시간을 비교한다.
2. 고정 unit 여러 개의 gradient를 worker에서 합쳐 한 결과로 전송하는 macro-unit을 검토한다. 이 방식은 per-unit clipping/audit 해상도가 낮아지므로 audit은 macro-unit 전체를 재계산한다.
3. 마지막으로 bf16 또는 blockwise int8 gradient encoding을 검토하고, fp32 기준에 대한 loss/gradient parity와 canary 허용치를 별도 설정한다.

무압축 fp32 gradient를 기준선으로 삼는다. 전송 압축, local gradient accumulation, macro-unit은 manifest에 protocol version을 기록하고 한 round 안에서 혼합하지 않는다.

## 6. Optimizer와 checkpoint

- 초기 단일 GPU sweep: attention-only rank 4/8, `alpha/r` 1 또는 2, adapter LR `1e-4`–`3e-4` 범위, special token 행은 adapter와 같은 LR 및 별도 LR 모두 비교한다. 이 값들은 시작 탐색 범위이지 검증된 설정은 아니다.
- Weight decay는 LoRA factor와 token rows에 기본 0으로 시작한다. 필요하면 validation 기준으로 별도 실험한다.
- Worker에는 optimizer state가 없다. Coordinator checkpoint에는 모든 LoRA factor, `E_new/H_new`, optimizer state, base revision/hash, tokenizer/token-map hash, shard commit, code hash, round log를 저장한다.
- Inference export는 base model을 수정하지 않는 adapter safetensors와 token row/tokenizer extension을 묶는다. runtime loader는 frozen base 위에 LoRA branch를 계산한다. GGUF/rwkv.cpp로 배포하려면 먼저 adapter merge 및 변환 경로를 확인해야 하며, 원본 GGUF가 LoRA를 직접 로드한다고 가정하지 않는다.

## 7. QLoRA worker

QLoRA 기준 recipe는 frozen dense base weight를 4-bit NF4로 저장하고, quantization scale도 double-quantize하며, matmul compute는 bf16으로 한다. LoRA factor와 새 token 행은 fp32 master parameter다. NF4, double quantization, paged optimizer의 메모리 목적은 QLoRA 논문에 정의되어 있고, bitsandbytes의 4-bit linear는 NF4/FP4 및 compute dtype 설정을 제공한다 ([QLoRA paper](https://arxiv.org/abs/2305.14314), [bitsandbytes quantization docs](https://huggingface.co/docs/bitsandbytes/main/quantization)). 여기서는 optimizer state가 coordinator에만 있고 adapter가 작으므로 paged optimizer는 v1 필수 요소가 아니다.

진짜 4-bit 메모리 절감을 얻으려면 LoRA를 붙인 네 attention projection만 양자화해서는 부족하다. 전체 weight 대부분을 차지하는 `ffn.key/value`도 frozen NF4 base로 실행해야 한다. `emb.weight`와 원래 `head.weight`는 우선 bf16 유지(합계 약 1 GiB)로 둔다. `E_new/H_new`는 별도 fp32 parameter다. Base head logits가 메모리 병목이면 chunked logits/CE와 quantized head를 따로 비교한다.

현재 코드에 bitsandbytes `Linear4bit`만 끼우는 것으로 끝나지 않는다.

- Time-mix 네 projection은 `nn.Linear` 호출이므로 quantized linear + LoRA wrapper로 교체할 수 있다. wrapper는 frozen quantized base matmul과 low-rank branch를 더하고 입력 activation gradient를 반환해야 한다.
- Channel-mix fast path는 raw `key.weight/value.weight`를 `_CmixLayerV2Fn`에 넘긴다. 먼저 reference QLoRA 구현에서 fast path를 우회하고, `Linear4bit` 두 연산과 `relu(key(x))²`를 명시적으로 실행한다. 정확성/메모리 기준이 확보되면 NF4 dequantize+matmul을 지원하도록 fused op를 바꾼다. 단순히 weight tensor를 4-bit로 바꿔 기존 bf16 op에 넘기는 구현은 성립하지 않는다.
- Embedding은 frozen bf16 lookup으로 시작한다. Head는 기존 L2Wrap raw-weight CUDA 경로를 끄거나 수정하고, 확장 vocab logits의 output/gradient를 chunked CE로 계산한다. QLoRA의 1차 correctness path는 plain summed CE로 고정한다.
- 모든 quantized base 연산은 base weight gradient를 만들지 않으면서 입력 activation gradient를 LoRA 및 `E_new`까지 전달해야 한다. 이 점을 admission parity에서 검증한다.

GGUF/rwkv.cpp inference 경로는 이 학습 경로의 대체재가 아니다. QLoRA worker의 quantization recipe(예: NF4 + double quant + bf16 compute)는 round manifest에 고정하고 서로 다른 recipe의 gradient를 섞지 않는다. 13.3B all-linear FP32-master profile은 2,048-token step에서 peak allocated 10,566 MiB / reserved 11,254 MiB였고, 지정 7.2B all-linear FP32-master profile은 4,096-token step에서 reserved 9,324 MiB였다. LoRA target 수가 더 적은 attention-only v1과 added-token rows의 peak는 별도 측정한다. Activation memory와 head/loss 경로는 checkpoint parameter count만으로 유도하지 않는다.

## 8. 검증 단계와 위험

1. **Reference:** 작은 모델/짧은 입력에서 일반 `nn.Linear` LoRA가 RWKV attention projection별 output/gradient reference와 맞는지 비교한다.
2. **7.2B checkpoint profile:** 지정된 `rwkv7-g1k-7.2b-20260930-ctx25600.pth`의 metadata/hash를 고정하고 all-linear QLoRA rank-8 reference sweep을 수행했다. 이어 attention-only QLoRA rank-4/8 및 special-token rows를 실제 v1 경로로 측정하고, 통과한 설정에서 품질 baseline을 만든다.
3. **13.3B reference profile:** 현재 full-model 측정은 all-linear rank-8 QLoRA와 synthetic token rows 없는 설정이다. 설계 v1의 attention-only rank-4/8 및 추가 token rows를 별도로 측정한다. BF16 LoRA는 가중치 자체가 현재 RTX 3060에 맞지 않으므로 더 큰 VRAM worker에서 기준을 만든다.
4. **Custom-op parity:** fused WKV를 통과하는 `A/B` gradient와 LoRA 없는 원본 forward가 B=0에서 일치하는지 확인한다.
5. **Special token gate:** 기존 ID `[0,V)` 보존, 신규 단일-ID encoding, `E_new` 입력 gradient, target에 있는 `H_new` gradient 및 finite 여부를 검사한다.
6. **Distributed equivalence:** 동일 unit 집합의 single-batch gradient와 여러 unit에서 합산한 gradient가 clipping 전 token-weighted 기준으로 tolerance 내에서 맞아야 한다.
7. **Network/fault simulation:** checkpoint별 실제 gradient payload, unit token budget, 재발급, 중복/오래된 adapter 결과, canary/audit 및 rollback을 측정한다.

주요 제약은 LoRA가 WKV state tuning보다 통신량과 trainable parameter 수가 크다는 점, QLoRA는 특히 FFN custom-op 경로를 고쳐야 한다는 점이다. 더 큰 effective batch는 gradient 변동과 토큰당 통신비용을 낮출 수 있지만 unit 완료시간을 늘린다. Peak VRAM은 동시 micro-batch와 context에 좌우되므로 accumulation과 분리해 설정한다. v1 attention-only LoRA는 initial-state backward 변경 없이 적용 가능하고 QLoRA는 frozen base 저장량을 줄인다.

## 참고 코드

- `RWKV-LM/RWKV-v7/train_temp/src/model.py` — `RWKV_Tmix_x070`, `RWKV_CMix_x070`, embedding/head, custom op 경로
- `RWKV-LM/RWKV-v7/train_temp/src/dataset.py` — token shift; SFT에서는 assistant label mask 추가 필요
- `dist_train_rwkv/Distributed_Soft_Prompt_SFT_Design_Doc.md` — 공통 unit/round/coordinator protocol
