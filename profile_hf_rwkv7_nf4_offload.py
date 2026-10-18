#!/usr/bin/env python3
"""Short Transformers/bitsandbytes NF4 inference probe with RWKV layer offload.

The HF RWKV-7 recurrent cache is a custom Python object, so Accelerate's ordinary
tensor-moving hooks cannot split its state across a mixed CPU/GPU block map. This
probe keeps the input embedding and first N blocks on CUDA, the remaining recurrent
blocks on CPU, and allocates each block's recurrent cache on that block's device.
CPU-offloaded modules stay FP32; CUDA Linear modules use NF4 with double
quantization. The script only performs a short forward and one recurrent decode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType

import bitsandbytes as bnb
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = Path("/workspace/RWKV7-G1k-13.3B-20260930")


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def memory_snapshot() -> dict[str, float | int | None]:
    cgroup_current = read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    return {
        "process_rss_gib": round(psutil.Process().memory_info().rss / 2**30, 3),
        "host_available_gib": round(psutil.virtual_memory().available / 2**30, 3),
        "cgroup_current_gib": round(cgroup_current / 2**30, 3) if cgroup_current is not None else None,
    }


def cuda_device(value) -> torch.device:
    if isinstance(value, int):
        return torch.device("cuda", value)
    if isinstance(value, torch.device):
        return value
    return torch.device(value)


def make_device_map(num_layers: int, cuda_layers: int) -> dict[str, int | str]:
    if not 0 <= cuda_layers <= num_layers:
        raise ValueError(f"cuda_layers must be between 0 and {num_layers}")
    placement: dict[str, int | str] = {"rwkv7.emb": 0, "head": 0}
    for layer in range(num_layers):
        placement[f"rwkv7.blocks.{layer}"] = 0 if layer < cuda_layers else "cpu"
    placement["rwkv7.ln_out"] = 0 if cuda_layers == num_layers else "cpu"
    return placement


def install_layer_local_cache(model, layer_devices: list[torch.device]) -> None:
    """Allocate the custom cache state on the same device as each RWKV block."""
    rwkv = model.rwkv7
    original_empty_state = rwkv._empty_state

    def empty_state_for_layer_devices(this, batch: int, device, dtype):
        cache = original_empty_state(batch, device, dtype)
        for layer_idx, layer in enumerate(cache.layers):
            target = layer_devices[layer_idx]
            for slot, tensor in list(layer.recurrent_states.items()):
                if tensor is not None and tensor.device != target:
                    layer.recurrent_states[slot] = tensor.to(target)
            layer.device = target
        return cache

    rwkv._empty_state = MethodType(empty_state_for_layer_devices, rwkv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cuda-layers", type=int, default=40)
    parser.add_argument(
        "--prompt",
        default="User: Reply with one short word.\n\nAssistant:",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Use this many deterministic synthetic token IDs instead of tokenizing --prompt.",
    )
    parser.add_argument(
        "--wkv-chunk-size",
        type=int,
        default=64,
        help="Chunk width for the HF chunked WKV reference (16 may reduce long-prefill memory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("hf_rwkv7_nf4_offload_20261018.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this mixed CPU/GPU probe")
    if not (args.model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"No local sharded safetensors index in {args.model}")

    torch.cuda.reset_peak_memory_stats(0)
    start_load = time.perf_counter()
    config_data = json.loads((args.model / "config.json").read_text())
    device_map = make_device_map(config_data["num_hidden_layers"], args.cuda_layers)
    has_cpu_layers = args.cuda_layers < config_data["num_hidden_layers"]
    activation_dtype = torch.float32 if has_cpu_layers else torch.bfloat16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=activation_dtype,
        # Transformers uses this switch to permit CPU/disk entries in a 4-bit
        # device map. Those CPU modules remain ordinary FP32 Linear layers.
        llm_int8_enable_fp32_cpu_offload=has_cpu_layers,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        local_files_only=True,
        dtype=activation_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    if args.wkv_chunk_size < 1:
        raise ValueError("wkv-chunk-size must be positive")
    if args.wkv_chunk_size != 64:
        model_module = sys.modules[model.__class__.__module__]
        original_chunked_wkv = model_module.rwkv7_chunked

        def rwkv7_chunked_with_selected_size(*args, **kwargs):
            return original_chunked_wkv(*args, chunk_size=args_cli_chunk_size, **kwargs)

        args_cli_chunk_size = args.wkv_chunk_size
        model_module.rwkv7_chunked = rwkv7_chunked_with_selected_size
    model.eval()
    load_seconds = time.perf_counter() - start_load

    layer_devices = [
        cuda_device(device_map[f"rwkv7.blocks.{layer}"])
        for layer in range(config_data["num_hidden_layers"])
    ]
    install_layer_local_cache(model, layer_devices)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True
    )
    embed_device = model.rwkv7.emb.weight.device
    if args.prompt_tokens is not None:
        if args.prompt_tokens < 1:
            raise ValueError("prompt-tokens must be positive")
        torch.manual_seed(20261018)
        input_ids = torch.randint(
            0,
            config_data["vocab_size"],
            (1, args.prompt_tokens),
            device=embed_device,
        )
        attention_mask = torch.ones_like(input_ids)
    else:
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(embed_device)
        attention_mask = encoded.attention_mask.to(embed_device)

    bnb_linear_count = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    all_linear_count = sum(isinstance(module, torch.nn.Linear) for module in model.modules())
    fp32_linear_count = sum(type(module) is torch.nn.Linear for module in model.modules())
    before_forward = memory_snapshot()
    torch.cuda.synchronize()
    start_forward = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded = model(input_ids=next_id, state=output.state, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start_forward

    if not torch.isfinite(output.logits).all() or not torch.isfinite(decoded.logits).all():
        raise RuntimeError("Non-finite logits in NF4/offload probe")
    observed_state_devices = [
        str(layer.recurrent_states[0].device)
        for layer in output.state.layers
    ]
    observed_state_device_counts = dict(Counter(observed_state_devices))
    layer_map_summary = {}
    if args.cuda_layers:
        layer_map_summary[f"rwkv7.blocks.0-{args.cuda_layers - 1}"] = "cuda:0"
    if args.cuda_layers < config_data["num_hidden_layers"]:
        layer_map_summary[
            f"rwkv7.blocks.{args.cuda_layers}-{config_data['num_hidden_layers'] - 1}"
        ] = "cpu"

    report = {
        "model": str(args.model.resolve()),
        "model_config": {
            "architecture": config_data["architectures"][0],
            "layers": config_data["num_hidden_layers"],
            "hidden_size": config_data["hidden_size"],
            "heads": config_data["num_heads"],
            "head_dim": config_data["head_dim"],
            "vocab_size": config_data["vocab_size"],
            "source_dtype": config_data["dtype"],
            "training_context": 25600,
            "wkv_chunk_size": args.wkv_chunk_size,
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "accelerate": __import__("accelerate").__version__,
            "bitsandbytes": bnb.__version__,
            "quantization": "NF4 + double quantization",
            "quantized_linear_modules": bnb_linear_count,
            "fp32_torch_linear_modules": fp32_linear_count,
            "all_linear_modules": all_linear_count,
            "compute_dtype": str(activation_dtype).removeprefix("torch."),
            "embedding_dtype": str(model.rwkv7.emb.weight.dtype).removeprefix("torch."),
            "head_module_type": type(model.head).__name__,
            "head_weight_dtype": str(model.head.weight.dtype).removeprefix("torch."),
            "cpu_offloaded_modules_remain_fp32": has_cpu_layers,
        },
        "placement": {
            "cuda_layers": args.cuda_layers,
            "cpu_layers": config_data["num_hidden_layers"] - args.cuda_layers,
            "device_map": {
                "rwkv7.emb": "cuda:0",
                **layer_map_summary,
                "rwkv7.ln_out": "cuda:0" if not has_cpu_layers else "cpu",
                "head": "cuda:0",
            },
            "cache_wkv_layer_device_counts": observed_state_device_counts,
        },
        "probe": {
            "prompt_tokens": int(input_ids.shape[1]),
            "synthetic_prompt": args.prompt_tokens is not None,
            "decode_tokens": 1,
            "prefill_logits_finite": bool(torch.isfinite(output.logits).all()),
            "decode_logits_finite": bool(torch.isfinite(decoded.logits).all()),
            "prefill_top_token_id": int(next_id.item()),
            "prefill_top_token_text": tokenizer.decode([int(next_id.item())]),
            "prefill_and_decode_seconds": round(forward_seconds, 3),
            "load_seconds": round(load_seconds, 3),
            "cpu_memory_before_forward": before_forward,
            "cpu_memory_after_forward": memory_snapshot(),
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
            "cuda_free_gib_after_forward": round(torch.cuda.mem_get_info(0)[0] / 2**30, 3),
            "model_memory_footprint_gib": round(model.get_memory_footprint() / 2**30, 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
