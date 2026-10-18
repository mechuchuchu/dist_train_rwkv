#!/usr/bin/env python3
"""Run a pinned, short HF TRL LoRA-SFT protocol check on RWKV7-G1j-1.5B."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
import trl
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
DATASET_ID = "trl-lib/Capybara"
DATASET_REVISION = "e235e846458bff3398a88aed812347f7f0756520"
DATASET_ROW_IDS = [1, 16, 21, 25]
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
OUTPUT_DIR = Path(__file__).parent / "artifacts/g1j_1.5b_capybara_sft_protocol"
REPORT_PATH = Path(__file__).with_name("g1j_1.5b_sft_protocol_validation.json")
ATTENTION_TARGETS = ["att.receptance", "att.key", "att.value", "att.output"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this protocol check")
    if not (MODEL_PATH / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(MODEL_PATH)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(20261018)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID,
        split="train[:32]",
        revision=DATASET_REVISION,
    ).select(DATASET_ROW_IDS)
    model = load_base()
    model.config.use_cache = False

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=ATTENTION_TARGETS,
    )
    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=192,
        packing=False,
        assistant_only_loss=True,
        use_cache=False,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=20261018,
        data_seed=20261018,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # Check the trainer's actual collated batch before fitting: assistant-only
    # masking must leave target labels while masking prompt tokens.
    first_batch = next(iter(trainer.get_train_dataloader()))
    labels = first_batch["labels"]
    input_ids = first_batch["input_ids"]
    loss_tokens = int(labels[:, 1:].ne(-100).sum().item())
    input_tokens = int(first_batch.get("attention_mask", torch.ones_like(input_ids)).sum().item())
    if loss_tokens <= 0 or loss_tokens >= input_tokens:
        raise RuntimeError(
            f"Unexpected assistant-only label mask: {loss_tokens=} {input_tokens=}"
        )
    expected_target_count = sum(
        int(tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )["assistant_masks"].count(1))
        for row in dataset
    )

    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    target_modules = [
        name
        for name, module in trainer.model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    wrong_targets = [name for name in target_modules if ".ffn." in name]
    if len(target_modules) != 24 * 4 or wrong_targets:
        raise RuntimeError(
            f"Expected 96 attention-only LoRA modules, got {len(target_modules)}; "
            f"FFN targets={wrong_targets[:4]}"
        )

    train_start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_start
    if trainer.state.global_step != 2:
        raise RuntimeError(f"Expected two optimizer steps, got {trainer.state.global_step}")
    changed_tensors = sum(
        not torch.equal(trainable_before[name], parameter.detach().cpu())
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    if changed_tensors == 0:
        raise RuntimeError("The optimizer completed but no LoRA parameter changed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Verify that a fresh base plus the saved adapter reproduces the trained
    # model's next-token logits, rather than only checking that files exist.
    probe_ids = input_ids[:1].to("cuda:0")
    trainer.model.eval()
    with torch.no_grad():
        reference_logits = trainer.model(
            input_ids=probe_ids, use_cache=False
        ).logits[:, -1, :].float().cpu()
    del first_batch, trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    fresh_base = load_base()
    reloaded = PeftModel.from_pretrained(
        fresh_base, str(OUTPUT_DIR), is_trainable=False
    )
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(
            input_ids=probe_ids, use_cache=False
        ).logits[:, -1, :].float().cpu()
    logit_delta = reference_logits - reloaded_logits
    max_abs_logit_diff = float(logit_delta.abs().max().item())
    relative_rms_logit_diff = float(
        (logit_delta.norm() / reference_logits.norm().clamp_min(1e-12)).item()
    )
    logit_cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_logits.flatten(), reloaded_logits.flatten(), dim=0
        ).item()
    )
    top_token_match = bool(
        reference_logits.argmax(dim=-1).eq(reloaded_logits.argmax(dim=-1)).all().item()
    )
    reload_passed = logit_cosine >= 0.9999 and relative_rms_logit_diff <= 1e-2 and top_token_match
    if not reload_passed:
        raise RuntimeError(
            "Adapter reload logits failed parity: "
            f"max_abs={max_abs_logit_diff}, relative_rms={relative_rms_logit_diff}, "
            f"cosine={logit_cosine}, top_token_match={top_token_match}"
        )

    report = {
        "model": {
            "repo": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(MODEL_PATH),
            "release_manifest_sha256": sha256(MODEL_PATH / "release-manifest.json"),
            "serialized_base_parameter_count": int(
                json.loads((MODEL_PATH / "release-manifest.json").read_text())
                ["conversion"]["serialized_parameter_count"]
            ),
            "dtype": "bfloat16 base; fp32 LoRA master factors",
            "layers": 24,
            "hidden_size": 2048,
            "num_heads": 32,
            "head_dim": 64,
            "training_context": 16384,
        },
        "dataset": {
            "repo": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train[:32]",
            "row_indices": DATASET_ROW_IDS,
            "format": "conversational messages",
            "assistant_only_loss": True,
            "first_collated_batch_input_tokens": input_tokens,
            "first_collated_batch_loss_tokens_after_causal_shift": loss_tokens,
            "selected_examples_template_assistant_tokens": expected_target_count,
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": __import__("peft").__version__,
            "datasets": __import__("datasets").__version__,
        },
        "lora": {
            "targets": ATTENTION_TARGETS,
            "target_module_count": len(target_modules),
            "trainable_parameter_count": sum(value.numel() for value in trainable_before.values()),
            "changed_trainable_tensors": changed_tensors,
            "rank": 8,
            "alpha": 16,
            "dropout": 0,
        },
        "training": {
            "max_steps": 2,
            "global_step": 2,
            "last_logged_loss": float(train_result.training_loss),
            "elapsed_seconds": round(train_seconds, 3),
            "optimizer_step_completed": True,
            "adapter_saved": True,
            "adapter_reload_logits_max_abs_diff": max_abs_logit_diff,
            "adapter_reload_logits_relative_rms_diff": relative_rms_logit_diff,
            "adapter_reload_logits_cosine": logit_cosine,
            "adapter_reload_top_token_match": top_token_match,
            "adapter_reload_passed": reload_passed,
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
            "cuda_free_gib_after_reload": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
        },
        "wall_seconds_including_dataset_load_and_reload": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
