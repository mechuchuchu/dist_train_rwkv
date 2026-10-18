# RWKV-7 VRAM / Runtime Baseline and Model-Scale Profiles

측정일: 2026-10-18 (UTC)  
측정 환경: 현재 Vast PyTorch 컨테이너, 단일 RTX 3060

## 장비와 환경 실측

| 항목 | 실측값 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060, compute capability 8.6 |
| VRAM (nvidia-smi) | 총 12,288 MiB, idle used 1 MiB, free 11,911 MiB |
| CUDA driver | 595.84 |
| PyTorch 접근 가능 GPU memory | 12,489,916,416 bytes = 11,911.3 MiB |
| Idle GPU 상태 | 45 °C, 13.15 W / 170 W, GPU utilization 0%, memory utilization 1% |
| CPU/RAM | 3 vCPU, RAM 31 GiB; 측정 시 available 약 29 GiB |
| Python stack | Python `/venv/main`, torch 2.11.0+cu128, torch CUDA 12.8 |
| 4-bit backend | bitsandbytes 0.50.2 (기설치, 이번 측정을 위한 설치 없음) |

`nvidia-smi`의 12,288 MiB와 PyTorch의 11,911.3 MiB는 서로 다른 가용 메모리 표기다. 실제 worker admission은 PyTorch가 접근 가능한 크기와 최대 peak 기준으로 판단한다.

## 체크포인트 크기

- 파일: `/workspace/.hf_home/hub/models--BlinkDL--rwkv7-g1/snapshots/2a17e911155a5be23a4ed2792e978c77038f14e2/rwkv7-g1k-13.3b-20260930-ctx25600.pth`
- snapshot은 HF blob `84a633fa02abcbdfcf91258b3b5bc000c66edfee2dbc6743f95a5efe1002fe4a`를 가리킨다.
- 파일 크기: **26,540,868,485 bytes = 24.72 GiB**. BF16 전체 모델 weight만으로도 현재 GPU의 접근 가능 VRAM보다 약 2.1배 크므로, 이 GPU에서는 BF16 전체 모델 로딩을 시도하지 않았다.
- NF4 weight의 단순 4-bit 하한은 BF16 payload의 약 1/4이다. `emb.weight`와 `head.weight`를 BF16으로 남기는 설계에서는 둘의 합계가 약 1 GiB다. 실제 QLoRA 점유량에는 quantization metadata, 임시 dequant/matmul buffers, activation 및 CUDA workspace가 추가된다. 이 값은 산술 추정이며 전체 모델 측정값이 아니다.

## 초기 NF4 단일 FFN 블록 실측 (이후 전체 모델 profile로 대체)

초기 baseline으로 bitsandbytes `Linear4bit`를 사용해 FFN 한 블록만 측정했다. 이는 아래 전체 모델 profile을 하기 전의 기록이다. 구성은 NF4, `compress_statistics=True`, BF16 compute, batch 1, fp32 rank-8 LoRA factor 2쌍, projection shape `4096→16384→4096`이었다. 이 단일 블록 결과는 전체 모델의 context admission에 사용하지 않는다.

| Sequence T | fwd/bwd 직전 allocated | peak allocated | peak reserved | 한 번 실행 시간 |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 83.3 MiB | 772.5 MiB | 858.0 MiB | 0.35 s |
| 8,192 | 147.5 MiB | 2,004.6 MiB | 2,822.0 MiB | 0.20 s |
| 16,384 | 211.5 MiB | 3,924.7 MiB | 5,254.0 MiB | 0.39 s |
| 25,600 | 283.5 MiB | 6,084.9 MiB | 8,720.0 MiB | 0.61 s |

시간은 길이마다 단일 실행한 rough timing이며 warm-up/반복이 없어 throughput 비교에 쓰지 않는다. Memory 값은 이 synthetic one-block probe에서만 재현된 수치다. 25,600 토큰에서도 단일 블록은 끝났지만, peak reserved가 PyTorch 접근 가능 VRAM의 약 73%를 차지했다. 여기에 전체 quantized base와 recurrent kernel scratch가 더해지므로 full-model 25,600 운용 가능성을 뜻하지 않는다.

## WKV scratch 산술 추정

`train_temp/src/model.py`의 custom WKV autograd는 각 호출에서 다음 fp32 tensor를 할당한다.

```text
s  : [B,H,T/16,N,N]
sa : [B,T,H,N]
```

`B=1`, checkpoint `H=N=64`일 때 이 둘의 합은 활성 sequence row당 약 **80 KiB × T**다.

| T | 단일 WKV 호출의 `s+sa` bytes 환산 |
| ---: | ---: |
| 2,048 | 약 160 MiB |
| 4,096 | 약 320 MiB |
| 8,192 | 약 640 MiB (0.625 GiB) |
| 16,384 | 약 1,280 MiB (1.25 GiB) |
| 25,600 | 약 2,000 MiB (1.95 GiB) |
| 51,200 | 약 4,000 MiB (3.91 GiB) |

이는 코드 shape에서 계산한 workspace 크기이며 GPU allocator로 직접 측정한 WKV 호출은 아니다. Activation, gradient, CUDA context 및 다른 kernel buffers도 별도로 든다. Gradient checkpointing은 여러 layer의 저장 activation을 줄이지만 활성 WKV 호출의 길이 의존 scratch 자체를 없애지는 않는다.

## 전체 모델 QLoRA 실측 — 13.3B checkpoint

2026-10-18에 전체 checkpoint를 실제로 로드하고 optimizer step까지 실행했다. 권위 결과는 FP32 master LoRA factor/AdamW state와 BF16 matmul로 다시 측정했다. Recipe는 rank-8 NF4 + double quantization, 366개 attention/FFN linear 전체 LoRA, 모든 RWKV block의 gradient checkpointing (`grad_cp=1`), plain CE와 AdamW one-step이다. Gradient clipping은 끄고 측정했다. Embedding과 vocabulary head는 frozen BF16이며 입력은 synthetic token이라 추가 special-token row는 포함하지 않았다. JSON은 [`full_model_profile_aggregate_2026-10-18.json`](full_model_profile_aggregate_2026-10-18.json), 실행 코드는 [`profile_full_rwkv7_qlora.py`](profile_full_rwkv7_qlora.py)다. 이전 BF16 adapter-factor 실행값은 이 FP32-master 결과로 대체한다.

| Micro-batch × context | Tokens/step | Peak allocated | Peak reserved | Step 시간 | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 × 1,024 | 1,024 | 9.05 GiB | 9.70 GiB | 4.54 s | 통과, 약 226 tok/s |
| 2 × 1,024 | 2,048 | 10.32 GiB | 10.99 GiB | 7.88 s | 통과, 약 260 tok/s |
| 1 × 2,048 | 2,048 | 10.32 GiB | 10.99 GiB | 8.00 s | 통과, 약 256 tok/s |
| 1 × 2,560 | 2,560 | 10.84 GiB | 11.48 GiB | 9.85 s | 통과, 여유 거의 없음 |
| 1 × 3,072 | 3,072 | — | — | — | OOM |

총 2,048 tokens/step은 실행됐지만 peak reserved가 10.99 GiB로, PyTorch 접근 가능 VRAM 11.63 GiB 중 약 0.64 GiB만 남겼다. 실제 SFT와 추가 token rows/head loss 전에는 `B=1,T=1024`를 보수적인 시작점으로 잡고, `B=2,T=1024` 또는 `B=1,T=2048`은 그 조건으로 다시 admission할 후보로 둔다. 2,560은 155 MiB 정도만 남긴 synthetic 통과값이라 기본값으로 쓰지 않는다. 다른 checkpoint, token rows, allocator 상태, LoRA target/rank, CUDA 버전에 따라 OOM 경계는 달라진다.

이 결과는 현재 13.3B 파일의 실제 성공/실패 측정이다. 별도 7.2B checkpoint의 결과는 다음 절에 기록하며, 어떤 측정도 다른 checkpoint에 그대로 적용하지 않는다. 동일한 parameter count label이라도 architecture와 BF16 embedding/head 크기, layer 수, head shape에 따라 memory가 달라진다.

profile 범위/한계:

- 모든 366개 linear를 QLoRA 대상으로 삼았고 설계서 LoRA v1의 attention-only와 target 범위가 다르다. attention-only v1의 admission 수치로 사용하지 않는다.
- Cmix는 quantized module을 경유하는 eager PyTorch reference로 우회했다. checkpoint의 low-rank 내부 dimension에 맞게 임시 source copy를 조정했고 CUDA 12.8 호환을 위해 `atomicAdd(float2*)`를 scalar atomic add 두 번으로 바꿨다. upstream source는 수정하지 않았다.
- 한 번의 synthetic step이므로 throughput은 rough timing이고 품질/수렴 지표가 아니다. LoRA factor와 AdamW state는 FP32 master이며 matmul만 BF16이다. Gradient checkpointing은 모든 block에 적용했고 gradient clipping은 끈 상태다. host peak RSS는 약 25.64 GiB였다. 메모리 여유가 작은 설정은 시스템 RAM도 고려한다.
- 이 profile은 LoRA backward와 optimizer step을 확인했지만 trainable initial WKV state의 `d_initial_state` 경로는 테스트하지 않았다. State Tuning의 admission 결과로 재사용하면 안 된다.

## 전체 모델 QLoRA 실측 — 지정 7.2B checkpoint

사용자가 지정한 [`rwkv7-g1k-7.2b-20260930-ctx25600.pth`](../.hf_home/hub/models--BlinkDL--rwkv7-g1/snapshots/2a17e911155a5be23a4ed2792e978c77038f14e2/rwkv7-g1k-7.2b-20260930-ctx25600.pth)는 SHA-256 `0853d4ce2e4e0629762797d05036a84cc69fb2fb9eff5590dd22aeb518a4c842`, 14,400,007,869 bytes다. mmap으로 읽은 모델은 7,199,932,416 parameters, `L=32,D=4096,H=N=64,V=65,536`, FFN width 16,384, BF16이다. Embedding과 head는 별도 `[65536,4096]` matrix다. 저장 구조, 내부 time-mix ranks(`w=128,a=128,v=96,g=480`)도 profile manifest에 기록했다.

이 profile은 NF4 + double quantization, BF16 compute, FP32 master rank-8 LoRA를 192개 attention/FFN linear 전체에 적용했다(18,874,368 adapter parameters). 모든 block gradient checkpointing (`grad_cp=1`), gradient clipping off, plain CE, AdamW optimizer step을 사용했으며 입력은 synthetic token이고 추가 special-token rows는 제외했다. 상세값은 [`full_model_profile_7_2b_aggregate_2026-10-18.json`](full_model_profile_7_2b_aggregate_2026-10-18.json)에 있다.

| Micro-batch × context | Tokens/step | Peak allocated | Peak reserved | Step 시간 | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 × 1,024 | 1,024 | 5.51 GiB | 6.01 GiB | 2.50 s | 통과, 약 409 tok/s |
| 2 × 1,024 | 2,048 | 6.42 GiB | 6.87 GiB | 4.15 s | 통과, 약 493 tok/s |
| 1 × 2,048 | 2,048 | 6.42 GiB | 6.87 GiB | 4.21 s | 통과, 약 487 tok/s |
| 1 × 3,072 | 3,072 | 7.25 GiB | 8.11 GiB | 6.09 s | 통과, 약 505 tok/s |
| 1 × 4,096 | 4,096 | 8.09 GiB | 9.14 GiB | 8.10 s | 통과, 약 505 tok/s |
| 4 × 1,024 | 4,096 | 7.95 GiB | 9.11 GiB | 8.16 s | 통과, 약 502 tok/s |
| 2 × 2,048 | 4,096 | 8.09 GiB | 9.11 GiB | 8.00 s | 통과, 약 512 tok/s |
| 1 × 6,144 | 6,144 | 9.77 GiB | 11.12 GiB | 12.00 s | 끝났으나 메모리 여유가 작음 |
| 1 × 8,192 | 8,192 | — | — | — | OOM |
| 4 × 2,048 | 8,192 | — | — | — | OOM |

profile adapter는 설계에 맞춰 FP32 master factor/optimizer state, BF16 matmul로 두었다. Gradient clipping은 끈 상태다. 이 worker의 이 recipe에서는 `micro-batch=2, context=2048` 또는 `micro-batch=4, context=1024`가 4,096 tokens/step을 확보하면서 peak reserved를 약 9.11 GiB에 두는 실측 후보이다. 이는 1-step synthetic profile이므로 실제 SFT row, 추가 token rows/head loss, 여러 optimizer step을 넣은 설정으로 다시 admission한다. `1×6144`는 약 0.51 GiB 여유만 남겨 안전 기본값으로 권하지 않으며, `8192` total tokens는 두 형태 모두 OOM이었다. 앞선 13.3B 결과와 달리 같은 RTX 3060에서도 모델 크기별 허용 batch/context가 다름을 보여준다.

13.3B profile과 동일하게 이 측정은 all-linear QLoRA이며 attention-only v1의 실제 memory 수치는 아니다. Cmix eager reference, CUDA 12.8 compatibility patch, synthetic one-step의 제한이 같다. host peak RSS는 약 14.33 GiB였다. State Tuning의 initial-state gradient를 거치지 않았으므로 State Tuning VRAM admission과 혼용하지 않는다.

## 크기별 공통 프로파일 설계

7B/13.3B 같은 이름은 편의 label로만 사용한다. 실제 tensor shape에서 모델 설정을 구성하고 `model_id`, revision, weight SHA-256마다 별도 profile을 남긴다. 새 checkpoint는 최소한 `n_layer (L)`, `n_embd (D)`, WKV `n_head (H)` 및 head size `N`, FFN projection shape, base vocab `V`, embedding/head shape와 tying 여부, 알려진 training context limit, dtype를 추출한다. 코드는 `D=4096`, `L=61`, `V=65536`, `FFN=4D`를 상수로 가정하지 않는다.

규모에 따른 trainable payload와 일부 작업 공간은 실제 shape로 계산한다.

```text
LoRA parameter count = Σ over targeted linear modules r × (in_features + out_features)
LoRA/token-row fp32 gradient bytes = 4 × (LoRA parameter count + 2 × M × D)
State-Tuning B count = L × H × R × N
WKV scratch bytes ≈ 4 × B_micro × H × T × (N²/16 + N), with T divisible by 16
NF4 base payload lower bound ≈ 0.5 × quantized_weight_count bytes + metadata
```

Embedding/head 등 BF16 유지 tensor는 실제 개별 shape로 더한다. NF4의 theoretical payload만으로 학습 가능성을 판정하지 않는다. Activation, head logits/loss, dequant buffer, optimizer update, WKV scratch까지 포함한 1-step peak가 admission 기준이다.

체크포인트마다 단계적으로 `batch × context`를 sweep한다. synthetic input으로 load 및 forward/backward/optimizer step을 먼저 확인한 후 실제 SFT row와 special-token rows를 포함한 최종 설정을 재측정한다. 총 tokens/step을 키우고 싶으면 안전한 micro-batch 내에서 local gradient accumulation을 적용해 peak를 억제한다. Context 자체는 16-token WKV chunk 경계에 맞추고 checkpoint의 알려진 training context까지 먼저 비교한다. 더 긴 context는 checkpoint별 별도 품질 검증과 VRAM 측정을 거친다.

각 결과에는 model identity/hash, shape metadata, trainable module 목록/rank, quantization recipe, micro-batch, accumulation, context, checkpointing, loss/head 방식, peak allocated/reserved, free VRAM, step time/tokens-per-second, host RSS, OOM, 코드/환경 hash를 함께 기록한다. LoRA, QLoRA, State Tuning은 gradient path가 다르므로 각 recipe에 별도 admission 기록을 둔다.
