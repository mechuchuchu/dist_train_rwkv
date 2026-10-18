#!/usr/bin/env python3
"""Run one tiny PEFT QLoRA training step on the local HF RWKV-7 checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_MODEL = Path("/workspace/RWKV7-G1k-13.3B-20260930")
ATTENTION_TARGETS = ["att.receptance", "att.key", "att.value", "att.output"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("hf_rwkv7_nf4_qlora_step_20261018.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA probe")
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least 2")
    if not (args.model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"No local sharded safetensors index in {args.model}")

    torch.manual_seed(20261018)
    torch.cuda.reset_peak_memory_stats(0)
    start_load = time.perf_counter()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
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
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start_step = time.perf_counter()
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = output.loss
    loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = [parameter.grad for _, parameter in trainable if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No LoRA gradients were produced")
    if not torch.isfinite(loss) or not all(torch.isfinite(grad).all().item() for grad in gradients):
        raise RuntimeError("Non-finite loss or LoRA gradient")
    grad_norm = torch.linalg.vector_norm(
        torch.stack([grad.detach().float().norm() for grad in gradients])
    )
    if not torch.isfinite(grad_norm) or grad_norm.item() == 0:
        raise RuntimeError(f"LoRA gradients are not finite and nonzero: {grad_norm.item()}")
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start_step

    target_modules = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    quantized_modules = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    report = {
        "model": str(args.model.resolve()),
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": bnb.__version__,
            "quantization": "NF4 + double quantization",
            "compute_dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "lora": {
            "target_suffixes": ATTENTION_TARGETS,
            "target_module_count": len(target_modules),
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "trainable_tensors": len(trainable),
            "non_null_gradient_tensors": len(gradients),
            "gradient_global_norm": grad_norm.item(),
            "quantized_linear_modules": quantized_modules,
            "example_targets": target_modules[:8],
        },
        "probe": {
            "sequence_length": args.sequence_length,
            "loss": loss.detach().float().item(),
            "loss_finite": bool(torch.isfinite(loss)),
            "gradients_finite": True,
            "optimizer_step_completed": True,
            "load_seconds": round(load_seconds, 3),
            "forward_backward_optimizer_seconds": round(step_seconds, 3),
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
            "cuda_free_gib_after_step": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
