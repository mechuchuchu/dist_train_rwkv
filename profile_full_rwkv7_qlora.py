#!/usr/bin/env python3
"""Profile one full RWKV-7 checkpoint with NF4 QLoRA on a single GPU.

This is a diagnostic, single-process training-step probe. It uses the upstream
RWKV-v7 train_temp CUDA kernels, replaces linear projections with bitsandbytes
NF4 + rank-r LoRA, and patches the Cmix call to use those quantized modules.
The embedding and vocabulary head stay frozen in BF16, matching the intended
special-token-row design. Input tokens are synthetic; no optimizer checkpoint
or model weights are written.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import re
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


CHECKPOINT = Path(
    "/workspace/.hf_home/hub/models--BlinkDL--rwkv7-g1/snapshots/"
    "2a17e911155a5be23a4ed2792e978c77038f14e2/"
    "rwkv7-g1k-13.3b-20260930-ctx25600.pth"
)
TRAIN_TEMP = Path("/workspace/RWKV-LM/RWKV-v7/train_temp")
DEFAULT_OUTPUT = Path(
    "/workspace/dist_train_rwkv/full_model_profile_results_2026-10-18.json"
)
PROFILE_SOURCE_ROOT = Path("/tmp/rwkv7-profile-vhhp6wd1")


def parse_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        batch, context = item.lower().split("x", 1)
        cases.append((int(batch), int(context)))
    return cases


def infer_checkpoint_config(checkpoint_path: Path, requested_head_size: int | None = None):
    """Read only checkpoint metadata/tensor shapes; do not materialize weights."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    block_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if not block_ids or block_ids != set(range(max(block_ids) + 1)):
        raise ValueError("checkpoint block indices are missing or non-contiguous")
    emb = state["emb.weight"]
    if emb.ndim != 2:
        raise ValueError(f"unexpected embedding shape: {tuple(emb.shape)}")
    n_layer = max(block_ids) + 1
    vocab_size, n_embd = map(int, emb.shape)
    att_key = state["blocks.0.att.key.weight"]
    ffn_key = state["blocks.0.ffn.key.weight"]
    r_k = state["blocks.0.att.r_k"]
    head_size = int(requested_head_size or r_k.shape[-1])
    if r_k.ndim != 2 or int(r_k.shape[-1]) != head_size:
        raise ValueError(
            f"r_k shape {tuple(r_k.shape)} conflicts with head_size={head_size}"
        )
    n_head = int(r_k.shape[0])
    if n_head * head_size != int(att_key.shape[0]):
        raise ValueError(
            "inferred head count/size do not match attention projection width: "
            f"{n_head}*{head_size} != {att_key.shape[0]}"
        )
    head = state["head.weight"]
    native_ranks = {
        name: int(state[f"blocks.0.att.{name}"].shape[1])
        for name in ("w1", "a1", "v1", "g1")
    }
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "dim_att": int(att_key.shape[0]),
        "dim_ffn": int(ffn_key.shape[0]),
        "head_size": head_size,
        "n_head": n_head,
        "embedding_dtype": str(emb.dtype),
        "embedding_shape": list(emb.shape),
        "head_shape": list(head.shape),
        "checkpoint_tensor_elements": sum(t.numel() for t in state.values()),
        "native_time_mix_ranks": native_ranks,
    }


class QLoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device="cuda", dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device="cuda", dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep FP32 master factors/optimizer state, with BF16 matmuls matching
        # the frozen base compute dtype. Autograd casts gradients to FP32.
        adapter = F.linear(
            F.linear(x, self.lora_A.to(dtype=x.dtype)),
            self.lora_B.to(dtype=x.dtype),
        )
        return self.base(x) + adapter * self.scale


def replace_linears(parent: nn.Module, bnb, rank: int, alpha: float) -> int:
    """Recursively swap nn.Linear for NF4 base + FP32-master LoRA."""
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear):
            if parent is model_ref[0] and name == "head":
                continue
            quant = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
                device="meta",
            )
            quant.weight = bnb.nn.Params4bit(
                child.weight.detach(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
                module=quant,
            )
            if child.bias is not None:
                quant.bias = nn.Parameter(child.bias.detach(), requires_grad=False)
            quant = quant.to("cuda")
            setattr(parent, name, QLoRALinear(quant, rank, alpha))
            replaced += 1
            del child, quant
        else:
            replaced += replace_linears(child, bnb, rank, alpha)
    return replaced


def install_cmix_eager(model: nn.Module) -> None:
    """Route Cmix through its modules; the fused op accepts raw, not NF4 weights."""
    def cmix_forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = F.pad(x, (0, 0, 1, -1)) - x
        key = x + xx * self.x_k
        key = torch.relu(self.key(key)) ** 2
        return self.value(key)

    for block in model.blocks:
        block.ffn.forward = cmix_forward.__get__(block.ffn, type(block.ffn))


def load_full_model(args, checkpoint_path: Path):
    import bitsandbytes as bnb

    # Keep upstream sources untouched. CUDA 12.8 rejects its atomicAdd(float2)
    # helper, so patch the helper in a temporary source copy to two scalar adds.
    source_root = PROFILE_SOURCE_ROOT
    shutil.rmtree(source_root, ignore_errors=True)
    shutil.copytree(TRAIN_TEMP / "cuda", source_root / "cuda")
    (source_root / "src").mkdir()
    model_source = (TRAIN_TEMP / "src" / "model.py").read_text()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    for key in list(state.keys()):
        if key.startswith("_forward_module."):
            state[key.replace("_forward_module.", "", 1)] = state.pop(key)

    # Checkpoints can use different native time-mix adapter dimensions. Read
    # those ranks from the checkpoint and apply them only to this temp copy.
    internal_ranks = {
        "D_DECAY_LORA": int(state["blocks.0.att.w1"].shape[1]),
        "D_AAA_LORA": int(state["blocks.0.att.a1"].shape[1]),
        "D_MV_LORA": int(state["blocks.0.att.v1"].shape[1]),
        "D_GATE_LORA": int(state["blocks.0.att.g1"].shape[1]),
    }
    for variable, rank in internal_ranks.items():
        model_source, count = re.subn(
            rf"(?m)^(\s*){variable} = .*?$",
            rf"\g<1>{variable} = {rank} # inferred from checkpoint",
            model_source,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"could not patch {variable} in temporary model source")

    # train_temp reuses the same Python helper name for several CUDA ops;
    # later definitions shadow earlier ones at runtime. Give each helper a
    # unique name in this isolated copy while keeping the registered ops intact.
    helper_renames = (
        (
            "_forward_op(x, x_r, x_w, x_k, x_v, x_a, x_g)",
            "_mix6_forward_op(x, x_r, x_w, x_k, x_v, x_a, x_g)",
        ),
        (
            "_forward_op(k, k_k, a, k_a)",
            "_kk_forward_op(k, k_k, a, k_a)",
        ),
        (
            "_forward_op(x, r, k, v, r_k, weight, bias, g)",
            "_lnx_forward_op(x, r, k, v, r_k, weight, bias, g)",
        ),
        ("_forward_op(a0, a12)", "_agate_forward_op(a0, a12)"),
        (
            "_forward_op(v, v_first, v0, v12)",
            "_vres_forward_op(v, v_first, v0, v12)",
        ),
    )
    for old, new in helper_renames:
        if model_source.count(old) != 2:
            raise RuntimeError(f"expected definition and caller for helper {old}")
        model_source = model_source.replace(old, new)
    (source_root / "src" / "model.py").write_text(model_source)
    (source_root / "src" / "__init__.py").write_text("")

    atomic_float2 = "atomicAdd(reinterpret_cast<float2*>(ptr), make_float2(x0, x1));"
    scalar_atomic_adds = "atomicAdd(ptr + 0, x0);\n    atomicAdd(ptr + 1, x1);"
    for source_name in (
        "rwkv7_tmix_mix6_bf16_v5.cu",
        "rwkv7_tmix_kk_pre_bf16_v5.cu",
        "rwkv7_cmix_bf16_v5.cu",
    ):
        source_path = source_root / "cuda" / source_name
        source_text = source_path.read_text()
        if atomic_float2 not in source_text:
            raise RuntimeError(f"expected CUDA 12.8 compatibility helper in {source_path}")
        source_path.write_text(source_text.replace(atomic_float2, scalar_atomic_adds))

    # The source model imports these extensions eagerly even though this probe
    # routes Cmix through its eager implementation and uses PyTorch CE.
    import torch.utils.cpp_extension as cpp_extension
    original_load = cpp_extension.load

    def load_needed_extension(*, name, **kwargs):
        if name in {"rwkv7_cmix_bf16_v5", "rwkv7_l2wrap_ce_bf16_v2"}:
            return None
        cached = glob.glob(
            str(Path(cpp_extension.get_default_build_root()) / "*" / name / f"{name}.so")
        )
        if cached:
            torch.ops.load_library(max(cached, key=os.path.getmtime))
            return None
        return original_load(name=name, **kwargs)

    cpp_extension.load = load_needed_extension
    sys.path.insert(0, str(source_root))
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    os.chdir(source_root)
    try:
        from src.model import RWKV

        # Meta initialization avoids touching 24.7 GiB of throwaway parameters.
        with torch.device("meta"):
            model = RWKV(args)
    finally:
        torch.set_default_dtype(original_default_dtype)
        cpp_extension.load = original_load

    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del state
    gc.collect()

    model.requires_grad_(False)
    # Bind the model root for the recursive converter so only its vocabulary
    # head is kept in BF16.
    model_ref[:] = [model]
    n_linears = replace_linears(model, bnb, args.lora_rank, args.lora_alpha)
    model.to("cuda")
    install_cmix_eager(model)
    model.train()
    torch.cuda.synchronize()
    gc.collect()
    return model, n_linears


model_ref = [None]


def run_case(model, optimizer, batch: int, context: int, vocab: int, n_layer: int):
    torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    ids = torch.randint(vocab, (batch, context), device="cuda", dtype=torch.long)
    targets = torch.randint(vocab, (batch, context), device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    x = model.emb(ids)
    v_first = torch.zeros_like(x)
    for block in model.blocks:
        x, v_first = checkpoint(block, x, v_first, use_reentrant=False)
    hidden = model.ln_out(x)
    logits = model.head(hidden)
    loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        "batch": batch,
        "context": context,
        "tokens_per_step": batch * context,
        "loss": float(loss.detach().float().cpu()),
        "step_seconds": elapsed,
        "tokens_per_second": batch * context / elapsed,
        "allocated_before_mib": allocated_before / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "free_before_mib": free_before / 2**20,
        "total_gpu_mib": total / 2**20,
        "cuda_allocated_after_mib": torch.cuda.memory_allocated() / 2**20,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--cases", default="1x2048,1x4096,1x8192,2x2048")
    parser.add_argument("--max-context", type=int, default=25600)
    parser.add_argument("--head-size", type=int)
    parser.add_argument("--rank", dest="lora_rank", type=int, default=8)
    parser.add_argument("--alpha", dest="lora_alpha", type=float, default=16.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli = parser.parse_args()

    cases = parse_cases(cli.cases)
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(cli.checkpoint)
    if any(t > cli.max_context or t % 16 for _, t in cases):
        raise ValueError("each context must be <= --max-context and divisible by 16")

    model_config = infer_checkpoint_config(cli.checkpoint, cli.head_size)

    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_KERNEL"] = ""
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_HEAD_SIZE"] = str(model_config["head_size"])
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = "0"
    os.environ["RWKV_CTXLEN"] = str(cli.max_context)
    sys.path.insert(0, str(TRAIN_TEMP))

    import argparse as argparse_module

    model_args = argparse_module.Namespace(
        n_layer=model_config["n_layer"],
        n_embd=model_config["n_embd"],
        vocab_size=model_config["vocab_size"],
        dim_att=model_config["dim_att"],
        dim_ffn=model_config["dim_ffn"],
        head_size=model_config["head_size"],
        ctx_len=cli.max_context,
        grad_cp=1,
        my_testing="x070",
        lora_rank=cli.lora_rank,
        lora_alpha=cli.lora_alpha,
    )

    print(
        f"Loading {cli.checkpoint} | cases={cases} | rank={cli.lora_rank} "
        f"alpha={cli.lora_alpha}",
        flush=True,
    )
    model, n_linears = load_full_model(model_args, cli.checkpoint)
    adapters = [p for p in model.parameters() if p.requires_grad]
    adapter_params = sum(p.numel() for p in adapters)
    optimizer = torch.optim.AdamW(adapters, lr=1e-4)
    torch.cuda.synchronize()
    load_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(cli.checkpoint),
        "checkpoint_size_bytes": cli.checkpoint.stat().st_size,
        "model_shape": model_config,
        "n_layer": model_args.n_layer,
        "n_embd": model_args.n_embd,
        "vocab_size": model_args.vocab_size,
        "base_quantization": "bitsandbytes NF4, double quantization, BF16 compute",
        "lora_rank": cli.lora_rank,
        "lora_alpha": cli.lora_alpha,
        "adapter_dtype": "FP32 master factors/optimizer state, BF16 matmul",
        "gradient_clipping": False,
        "quantized_linear_count": n_linears,
        "trainable_adapter_parameters": adapter_params,
        "embedding_and_head": "frozen BF16; no extra token rows in synthetic probe",
        "activation_checkpointing": "torch.utils.checkpoint, each RWKV block",
        "cmix_path": "upstream eager Cmix reference routed through quantized modules",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cases": [],
        "oom_case": None,
        "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    print(
        f"Loaded full model | NF4 projections={n_linears} | "
        f"LoRA params={adapter_params:,} | "
        f"GPU allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB | "
        f"host peak RSS={load_record['host_peak_rss_gib']:.2f} GiB",
        flush=True,
    )

    for batch, context in cases:
        try:
            result = run_case(
                model,
                optimizer,
                batch,
                context,
                model_args.vocab_size,
                model_args.n_layer,
            )
            load_record["cases"].append(result)
        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            load_record["oom_case"] = {
                "batch": batch,
                "context": context,
                "error": str(exc).splitlines()[0],
            }
            print(f"CUDA OOM at batch={batch}, context={context}: {exc}", flush=True)
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    load_record["host_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(load_record, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {cli.output}", flush=True)


if __name__ == "__main__":
    main()
