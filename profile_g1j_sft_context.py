#!/usr/bin/env python3
"""Profile a real BF16 LoRA optimizer step at one RWKV-7 sequence length."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
TARGETS = ["att.receptance", "att.key", "att.value", "att.output"]
PROMPT_UNIT = (
    "Context note: keep the facts in this passage available for the final response. "
    "Each independent training example begins from the same recurrent state. "
    "The worker computes a loss sum over labeled assistant tokens and returns its "
    "gradient with the exact loss-token count. "
)
ANSWER = (
    "The protocol uses an independent initial recurrent state for every row, "
    "and combines summed gradients by the number of assistant loss tokens. "
)


def tokenize_at_length(tokenizer, target_length: int):
    def encode(repetitions: int):
        messages = [
            {"role": "user", "content": PROMPT_UNIT * repetitions},
            {"role": "assistant", "content": ANSWER},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

    lo, hi = 0, max(target_length, 1)
    best = encode(lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        encoded = encode(mid)
        if len(encoded["input_ids"]) <= target_length:
            best = encoded
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def host_memory():
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cgroup_current = None
    cgroup_limit = None
    try:
        cgroup_current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        cgroup_limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
    except (OSError, ValueError):
        pass
    return {
        "process_peak_rss_gib": round(rss_bytes / 2**30, 3),
        "cgroup_current_gib": None if cgroup_current is None else round(cgroup_current / 2**30, 3),
        "cgroup_limit_gib": None if cgroup_limit is None else round(cgroup_limit / 2**30, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training profile")
    if args.sequence_length < 16 or args.sequence_length > 16384:
        raise ValueError("sequence length must be between 16 and the model's 16,384 context")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20261018)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    encoded = tokenize_at_length(tokenizer, args.sequence_length)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
    assistant_mask = torch.tensor(encoded["assistant_masks"], dtype=torch.bool, device="cuda:0")
    target_positions = torch.nonzero(assistant_mask, as_tuple=False).flatten()
    target_positions = target_positions[target_positions > 0]
    if target_positions.numel() == 0:
        raise RuntimeError("The current template produced no assistant target labels")
    predictor_positions = target_positions - 1
    target_ids = input_ids[0, target_positions]

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGETS,
        ),
    )
    model.train()
    targets = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    if len(targets) != 96 or any(".ffn." in name for name in targets):
        raise RuntimeError(f"Expected exactly 96 attention LoRA modules, got {len(targets)}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0)
    torch.cuda.synchronize()
    memory_before = torch.cuda.mem_get_info(0)[0] / 2**30
    torch.cuda.reset_peak_memory_stats(0)

    result = {"status": "error"}
    try:
        step_started = time.perf_counter()
        output = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=predictor_positions,
        )
        logits = output.logits[0].float()
        loss_sum = torch.nn.functional.cross_entropy(
            logits,
            target_ids,
            reduction="sum",
        )
        loss_tokens = int(target_ids.numel())
        loss_sum.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
            raise RuntimeError("Missing or non-finite LoRA gradients")
        grad_norm = torch.linalg.vector_norm(
            torch.stack([grad.detach().float().norm() for grad in gradients])
        )
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0:
            raise RuntimeError("LoRA gradient norm is not finite and nonzero")
        with torch.no_grad():
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(loss_tokens)
        optimizer.step()
        torch.cuda.synchronize()
        result = {
            "status": "passed",
            "loss_sum": float(loss_sum.detach().float().item()),
            "loss_tokens": loss_tokens,
            "mean_loss": float(loss_sum.detach().float().item() / loss_tokens),
            "gradient_global_norm_of_loss_sum": float(grad_norm.item()),
            "finite_gradient_tensors": len(gradients),
            "optimizer_step_completed": True,
            "forward_backward_optimizer_seconds": round(time.perf_counter() - step_started, 3),
        }
        del output, logits, loss_sum, gradients
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        result = {"status": "oom", "error": str(exc).splitlines()[0]}
    finally:
        torch.cuda.synchronize()

    report = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_path": str(MODEL_PATH),
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "BF16 frozen base; FP32 attention LoRA factors",
        "sequence_length_target": args.sequence_length,
        "sequence_length_actual": int(input_ids.shape[1]),
        "target_modules": TARGETS,
        "target_module_count": len(targets),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "gradient_checkpointing": True,
        "use_cache": False,
        "logits_to_keep": int(predictor_positions.numel()),
        "assistant_only_loss": True,
        "template_end_of_turn_warning": "Current model chat template omits EOS from assistant loss mask; user plans a later template revision.",
        "cuda_free_gib_before_step": round(memory_before, 3),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
        "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
        "cuda_free_gib_after_step": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
        "host_memory": host_memory(),
        "step": result,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path = args.output or Path(__file__).with_name(
        f"g1j_1.5b_sft_context_{args.sequence_length}.json"
    )
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    del model, optimizer, input_ids, target_ids
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
