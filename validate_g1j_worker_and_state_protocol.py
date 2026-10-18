#!/usr/bin/env python3
"""Check RWKV7-G1j worker-gradient aggregation and learned initial-state gradients."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
DATASET_ID = "trl-lib/Capybara"
DATASET_REVISION = "e235e846458bff3398a88aed812347f7f0756520"
DATASET_ROW_IDS = [1, 16]
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
REPORT_PATH = Path(__file__).with_name("g1j_1.5b_worker_state_protocol_validation.json")
STATE_ADAPTER_DIR = Path(__file__).parent / "artifacts/g1j_1.5b_state_tuning_protocol"
ATTENTION_TARGETS = ["att.receptance", "att.key", "att.value", "att.output"]
RANK = 8


def load_base():
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
    return model


def tokenize_messages(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    labels = input_ids.clone()
    assistant_mask = torch.tensor(encoded["assistant_masks"], dtype=torch.bool)
    labels[~assistant_mask] = -100
    return input_ids, labels


def causal_loss_sum(logits, labels):
    shifted_logits = logits[:, :-1, :].contiguous().float()
    shifted_labels = labels[:, 1:].contiguous()
    count = int(shifted_labels.ne(-100).sum().item())
    if count == 0:
        raise RuntimeError("A unit has no assistant loss tokens after causal shift")
    loss_sum = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="sum",
        ignore_index=-100,
    )
    return loss_sum, count


def grads_as_cpu(named_parameters):
    return {
        name: parameter.grad.detach().float().cpu().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        for name, parameter in named_parameters
    }


def flattened(grads):
    return torch.cat([grads[name].reshape(-1) for name in sorted(grads)])


def cosine_and_relative_error(reference, candidate):
    reference = flattened(reference)
    candidate = flattened(candidate)
    cosine = F.cosine_similarity(reference, candidate, dim=0).item()
    relative = (candidate - reference).norm().item() / reference.norm().clamp_min(1e-12).item()
    return float(cosine), float(relative)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this protocol check")
    started = time.perf_counter()
    torch.manual_seed(20261018)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID,
        split="train[:32]",
        revision=DATASET_REVISION,
    ).select(DATASET_ROW_IDS)
    examples = [tokenize_messages(tokenizer, row["messages"]) for row in dataset]

    # Simulate independent workers returning loss-sum gradients, then compare
    # their token-weighted aggregate against the same examples as one padded batch.
    model = load_base()
    lora = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=RANK,
            lora_alpha=2 * RANK,
            lora_dropout=0.0,
            bias="none",
            target_modules=ATTENTION_TARGETS,
        ),
    )
    lora.train()
    trainable = [(name, p) for name, p in lora.named_parameters() if p.requires_grad]
    if sum(p.numel() for _, p in trainable) != 3_145_728:
        raise RuntimeError("Unexpected attention-only LoRA parameter count")
    initial_params = {name: p.detach().cpu().clone() for name, p in trainable}
    worker_grads = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu") for name, p in trainable}
    loss_sum_total, loss_tokens_total = 0.0, 0

    for input_ids, labels in examples:
        lora.zero_grad(set_to_none=True)
        ids = input_ids.unsqueeze(0).to("cuda:0")
        target = labels.unsqueeze(0).to("cuda:0")
        output = lora(input_ids=ids, use_cache=False)
        loss_sum, loss_tokens = causal_loss_sum(output.logits, target)
        loss_sum.backward()
        for name, grad in grads_as_cpu(trainable).items():
            worker_grads[name].add_(grad)
        loss_sum_total += float(loss_sum.detach().float().item())
        loss_tokens_total += loss_tokens
        del output, loss_sum, ids, target

    lora.zero_grad(set_to_none=True)
    pad_id = tokenizer.pad_token_id or 0
    max_length = max(ids.numel() for ids, _ in examples)
    batch_ids = torch.full((len(examples), max_length), pad_id, dtype=torch.long)
    batch_labels = torch.full((len(examples), max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    for row, (ids, labels) in enumerate(examples):
        length = ids.numel()
        batch_ids[row, :length] = ids
        batch_labels[row, :length] = labels
        attention_mask[row, :length] = 1
    output = lora(
        input_ids=batch_ids.to("cuda:0"),
        attention_mask=attention_mask.to("cuda:0"),
        use_cache=False,
    )
    central_loss_sum, central_loss_tokens = causal_loss_sum(
        output.logits, batch_labels.to("cuda:0")
    )
    central_loss_sum.backward()
    central_grads = grads_as_cpu(trainable)
    grad_cosine, grad_relative_error = cosine_and_relative_error(worker_grads, central_grads)
    central_loss_value = float(central_loss_sum.detach().float().item())
    loss_relative_error = abs(loss_sum_total - central_loss_value) / max(
        abs(central_loss_value), 1e-12
    )
    if loss_tokens_total != central_loss_tokens:
        raise RuntimeError(
            f"Worker/central token counts differ: {loss_tokens_total} != {central_loss_tokens}"
        )
    if grad_cosine < 0.99 or grad_relative_error > 0.1 or loss_relative_error > 0.01:
        raise RuntimeError(
            "Worker gradient aggregation failed: "
            f"loss_relative={loss_relative_error}, cosine={grad_cosine}, "
            f"gradient_relative={grad_relative_error}"
        )
    worker_mean = {name: grad / loss_tokens_total for name, grad in worker_grads.items()}
    central_mean = {name: grad / central_loss_tokens for name, grad in central_grads.items()}

    def coordinator_step(grads):
        with torch.no_grad():
            for name, parameter in trainable:
                parameter.copy_(initial_params[name].to(parameter.device, parameter.dtype))
                parameter.grad = grads[name].to(parameter.device, dtype=parameter.dtype)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable], lr=1e-4, weight_decay=0.0
        )
        optimizer.step()
        result = {name: p.detach().cpu().clone() for name, p in trainable}
        del optimizer
        return result

    worker_update = coordinator_step(worker_mean)
    central_update = coordinator_step(central_mean)
    update_diffs = [
        (worker_update[name] - central_update[name]).abs().max().item()
        for name in worker_update
    ]
    max_update_diff = max(update_diffs)
    mean_update_diff = sum(update_diffs) / len(update_diffs)
    del coordinator_step, trainable, output, lora, model, batch_ids, batch_labels, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # A low-rank FP32 initial WKV state must receive finite gradients through the
    # frozen real model. Compare the chunk-parallel path with the tokenwise eager
    # reference at the same zero-initialized state.
    state_model = load_base()
    state_model.eval()
    for parameter in state_model.parameters():
        parameter.requires_grad_(False)
    state_input, state_labels = tokenize_messages(
        tokenizer,
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4."},
        ],
    )
    state_input = state_input.unsqueeze(0).to("cuda:0")
    state_labels = state_labels.unsqueeze(0).to("cuda:0")
    cfg = state_model.config
    layers, heads, head_dim = cfg.num_hidden_layers, cfg.num_heads, cfg.head_dim
    rank = RANK
    generator = torch.Generator(device="cpu").manual_seed(20261018)
    raw_basis = torch.randn(layers, heads, head_dim, rank, generator=generator)
    basis = torch.linalg.qr(raw_basis, mode="reduced").Q.to("cuda:0", dtype=torch.float32)

    def state_gradient(implementation):
        state_model.config.wkv_implementation = implementation
        state_b = torch.nn.Parameter(
            torch.zeros(layers, heads, rank, head_dim, device="cuda:0", dtype=torch.float32)
        )
        cache = state_model.rwkv7.allocate_state(
            batch=1, device="cuda:0", dtype=torch.bfloat16
        )
        initial = torch.einsum("lhdr,lhrn->lhdn", basis, state_b)
        for layer_index, layer in enumerate(cache.layers):
            layer.recurrent_states[cache.WKV] = initial[layer_index].unsqueeze(0)
        output = state_model(
            input_ids=state_input,
            state=cache,
            use_cache=False,
        )
        state_loss, state_loss_tokens = causal_loss_sum(output.logits, state_labels)
        state_loss.backward()
        if state_b.grad is None:
            raise RuntimeError(f"No initial-state adapter gradient for {implementation}")
        grad = state_b.grad.detach().float().cpu().clone()
        loss_value = float(state_loss.detach().float().item())
        if not torch.isfinite(grad).all() or grad.norm().item() == 0:
            raise RuntimeError(f"Invalid state gradient for {implementation}")
        return state_b, grad, loss_value, state_loss_tokens

    state_model.zero_grad(set_to_none=True)
    chunked_b, chunked_grad, chunked_loss, state_loss_tokens = state_gradient("chunked")
    state_optimizer = torch.optim.AdamW([chunked_b], lr=1e-4, weight_decay=0.0)
    state_optimizer.step()
    state_parameter_changed = bool(chunked_b.detach().abs().max().item() > 0)
    STATE_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    save_file({"B": chunked_b.detach().cpu().contiguous()}, str(STATE_ADAPTER_DIR / "state_adapter.safetensors"))
    (STATE_ADAPTER_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "model_repo": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "rank": rank,
                "basis_seed": 20261018,
                "tensor": "B[layer,head,rank,head_dim]",
                "dtype": "float32",
                "optimizer": "AdamW(lr=1e-4,weight_decay=0)",
                "steps": 1,
                "validation_artifact_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    del state_optimizer
    state_model.zero_grad(set_to_none=True)
    del chunked_b
    torch.cuda.empty_cache()

    eager_b, eager_grad, eager_loss, eager_loss_tokens = state_gradient("eager")
    state_grad_cosine = float(
        F.cosine_similarity(chunked_grad.flatten(), eager_grad.flatten(), dim=0).item()
    )
    state_grad_relative_error = float(
        ((chunked_grad - eager_grad).norm() / eager_grad.norm().clamp_min(1e-12)).item()
    )
    state_loss_relative_error = abs(chunked_loss - eager_loss) / max(abs(eager_loss), 1e-12)
    if eager_loss_tokens != state_loss_tokens:
        raise RuntimeError("Chunked/eager state loss-token counts differ")
    if state_grad_cosine < 0.98 or state_grad_relative_error > 0.1:
        raise RuntimeError(
            "State gradient reference parity failed: "
            f"cosine={state_grad_cosine}, relative={state_grad_relative_error}"
        )
    if not state_parameter_changed:
        raise RuntimeError("The learned WKV initial-state parameter did not update")

    report = {
        "model": {
            "repo": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(MODEL_PATH),
            "dtype": "BF16 frozen base; FP32 trainable parameters/state",
            "layers": layers,
            "hidden_size": cfg.hidden_size,
            "heads": heads,
            "head_dim": head_dim,
            "training_context": 16384,
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": __import__("peft").__version__,
        },
        "worker_gradient_protocol": {
            "dataset_repo": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "row_indices": DATASET_ROW_IDS,
            "unit_count": len(examples),
            "loss_tokens": loss_tokens_total,
            "worker_vs_padded_batch_loss_sum_relative_error": loss_relative_error,
            "attention_only_lora_target_count": 24 * 4,
            "trainable_parameters": 3_145_728,
            "fp32_gradient_payload_mib": 3_145_728 * 4 / 2**20,
            "per_unit_loss_sum_reported": True,
            "worker_vs_padded_batch_gradient_cosine": grad_cosine,
            "worker_vs_padded_batch_gradient_relative_l2_error": grad_relative_error,
            "gradient_cosine_canary_threshold": 0.99,
            "gradient_relative_l2_threshold": 0.1,
            "gradient_canary_passed": True,
            "worker_vs_padded_batch_coordinator_adamw_max_parameter_diff": max_update_diff,
            "worker_vs_padded_batch_coordinator_adamw_mean_tensor_max_diff": mean_update_diff,
            "strict_optimizer_update_parity_claimed": False,
            "optimizer_update_difference_requires_followup": True,
            "aggregation_passed": True,
        },
        "state_tuning_protocol": {
            "rank": rank,
            "trainable_B_parameter_count": layers * heads * rank * head_dim,
            "fp32_B_payload_mib": layers * heads * rank * head_dim * 4 / 2**20,
            "sequence_tokens": int(state_input.shape[1]),
            "loss_tokens": state_loss_tokens,
            "initial_state_dtype": "float32",
            "base_parameters_frozen": True,
            "gradient_checkpointing": False,
            "chunked_loss_sum": chunked_loss,
            "eager_loss_sum": eager_loss,
            "chunked_eager_loss_relative_error": state_loss_relative_error,
            "chunked_eager_B_gradient_cosine": state_grad_cosine,
            "chunked_eager_B_gradient_relative_l2_error": state_grad_relative_error,
            "finite_nonzero_B_gradient": True,
            "optimizer_step_changed_B": state_parameter_changed,
            "saved_one_step_state_adapter": str(STATE_ADAPTER_DIR / "state_adapter.safetensors"),
            "parity_passed": True,
        },
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
        "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
        "cuda_free_gib_after_test": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
