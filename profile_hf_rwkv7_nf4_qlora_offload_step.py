#!/usr/bin/env python3
"""Run a real QLoRA forward/backward/optimizer step with RWKV blocks on CPU."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import bitsandbytes as bnb
import psutil
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_MODEL = Path("/workspace/RWKV7-G1k-13.3B-20260930")
ATTENTION_TARGETS = ["att.receptance", "att.key", "att.value", "att.output"]


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def cgroup_memory() -> tuple[int | None, int | None]:
    candidates = (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    )
    for current_path, limit_path in candidates:
        current = read_int(current_path)
        limit = read_int(limit_path)
        if current is not None:
            return current, limit
    return None, None


def memory_snapshot() -> dict[str, float | None]:
    current, limit = cgroup_memory()
    return {
        "process_rss_gib": round(psutil.Process().memory_info().rss / 2**30, 3),
        "host_available_gib": round(psutil.virtual_memory().available / 2**30, 3),
        "cgroup_current_gib": round(current / 2**30, 3) if current is not None else None,
        "cgroup_limit_gib": round(limit / 2**30, 3) if limit is not None else None,
    }


def device_name(device: torch.device) -> str:
    return "cpu" if device.type == "cpu" else f"cuda:{device.index or 0}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cuda-layers", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("hf_rwkv7_nf4_qlora_offload_step_20261018.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this mixed CPU/GPU training probe")
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least 2")
    if not (args.model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"No local sharded safetensors index in {args.model}")

    torch.manual_seed(20261018)
    torch.cuda.reset_peak_memory_stats(0)
    start_load = time.perf_counter()
    config_data = json.loads((args.model / "config.json").read_text())
    layer_count = int(config_data["num_hidden_layers"])
    if not 0 <= args.cuda_layers <= layer_count:
        raise ValueError(f"cuda-layers must be in [0, {layer_count}]")
    has_cpu_layers = args.cuda_layers < layer_count
    # Transformers' 4-bit loader keeps CPU-dispatched modules as ordinary FP32
    # Linear layers. Match their input dtype; CUDA NF4 modules also compute FP32.
    compute_dtype = torch.float32 if has_cpu_layers else torch.bfloat16
    device_map: dict[str, int | str] = {"rwkv7.emb": 0, "head": 0}
    for layer in range(layer_count):
        device_map[f"rwkv7.blocks.{layer}"] = 0 if layer < args.cuda_layers else "cpu"
    device_map["rwkv7.ln_out"] = 0 if not has_cpu_layers else "cpu"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_enable_fp32_cpu_offload=has_cpu_layers,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        local_files_only=True,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=ATTENTION_TARGETS,
        ),
    )
    model.train()
    load_seconds = time.perf_counter() - start_load

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, args.sequence_length),
        device="cuda:0",
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were created")
    trainable_devices = Counter(device_name(parameter.device) for _, parameter in trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=1e-4, foreach=False
    )
    optimizer.zero_grad(set_to_none=True)
    before_step = memory_snapshot()

    torch.cuda.synchronize()
    start_step = time.perf_counter()
    status = "passed"
    error = None
    try:
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = output.loss
        loss.backward()
        gradients = [
            parameter.grad
            for _, parameter in trainable
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("No LoRA gradients were produced")
        if not torch.isfinite(loss).item() or not all(
            torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise RuntimeError("Non-finite loss or LoRA gradient")
        grad_norm_sq = sum(float(gradient.detach().float().norm().item()) ** 2 for gradient in gradients)
        grad_norm = grad_norm_sq**0.5
        if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm == 0:
            raise RuntimeError(f"LoRA gradients are not finite and nonzero: {grad_norm}")
        optimizer.step()
        torch.cuda.synchronize()
        loss_value = float(loss.detach().float().item())
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        loss_value = None
        grad_norm = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start_step
    after_step = memory_snapshot()

    quantized_modules = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    lora_modules = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    report = {
        "status": status,
        "error": error,
        "model": str(args.model.resolve()),
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": bnb.__version__,
            "quantization": "NF4 + double quantization",
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
            "gradient_checkpointing": True,
            "cpu_layers_fp32": has_cpu_layers,
        },
        "placement": {
            "cuda_layers": args.cuda_layers,
            "cpu_layers": layer_count - args.cuda_layers,
            "trainable_lora_tensor_counts_by_device": dict(trainable_devices),
            "device_map": {
                "rwkv7.emb": "cuda:0",
                f"rwkv7.blocks.0-{args.cuda_layers - 1}": "cuda:0" if args.cuda_layers else None,
                f"rwkv7.blocks.{args.cuda_layers}-{layer_count - 1}": "cpu" if has_cpu_layers else None,
                "rwkv7.ln_out": "cpu" if has_cpu_layers else "cuda:0",
                "head": "cuda:0",
            },
        },
        "lora": {
            "target_suffixes": ATTENTION_TARGETS,
            "target_module_count": len(lora_modules),
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "trainable_tensor_count": len(trainable),
            "non_null_gradient_tensor_count": sum(
                parameter.grad is not None for _, parameter in trainable
            ),
            "gradient_global_norm": grad_norm,
            "quantized_linear_modules": quantized_modules,
        },
        "probe": {
            "sequence_length": args.sequence_length,
            "loss": loss_value,
            "optimizer_step_completed": status == "passed",
            "load_seconds": round(load_seconds, 3),
            "forward_backward_optimizer_seconds": round(step_seconds, 3),
            "memory_before_step": before_step,
            "memory_after_step": after_step,
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
            "cuda_free_gib_after_step": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
