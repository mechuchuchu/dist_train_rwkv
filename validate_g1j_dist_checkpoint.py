#!/usr/bin/env python3
"""Verify a coordinator-produced adapter is PEFT-loadable and matches its theta."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/g1j_local_protocol/round-0001"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter reload validation")
    theta = load_file(str(args.run_dir / "theta.safetensors"), device="cpu")
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
    model = PeftModel.from_pretrained(
        model, str(args.run_dir), is_trainable=False, local_files_only=True
    )
    model.eval()
    actual = {name: p.detach().float().cpu() for name, p in model.named_parameters() if ".lora_" in name}
    if set(actual) != set(theta):
        raise RuntimeError(f"loaded adapter schema differs from theta: {len(actual)} != {len(theta)}")
    max_diff = max((actual[name] - theta[name]).abs().max().item() for name in theta)
    if max_diff != 0:
        raise RuntimeError(f"PEFT-loaded adapter differs from coordinator theta by {max_diff}")

    data_manifest = json.loads((args.run_dir / "data_manifest.json").read_text())
    unit = data_manifest["units"][0]
    input_ids = torch.tensor(unit["input_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
    labels = torch.tensor(unit["labels"], dtype=torch.long, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
    if not torch.isfinite(loss):
        raise RuntimeError("reloaded coordinator adapter produced non-finite loss")
    digest = hashlib.sha256(logits[:, -1, :].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    result = {
        "adapter_reload": "passed",
        "adapter_tensor_count": len(theta),
        "adapter_max_abs_difference_vs_coordinator_theta": max_diff,
        "probe_unit_id": unit["unit_id"],
        "probe_loss_tokens": unit["loss_tokens"],
        "probe_loss_sum": float(loss.item()),
        "probe_mean_loss": float(loss.item() / unit["loss_tokens"]),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "last_position_logits_sha256": digest,
    }
    (args.run_dir / "checkpoint_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
