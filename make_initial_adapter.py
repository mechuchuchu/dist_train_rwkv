#!/usr/bin/env python3
"""Create the clean rank-8 LoRA adapter a fresh distributed run starts from.

The coordinator needs an `adapter_config.json` and `adapter_model.safetensors`
for theta version 0. This builds them with the worker's own model loader, so
the tensor names, shapes and dtypes are by construction the ones every worker
will ask for, then writes a PEFT adapter directory the coordinator accepts.

The starting point is the standard LoRA one: every lora_A is drawn from
Kaiming-uniform(a=sqrt(5)) and every lora_B is zero, so the adapted model is
exactly the base model before the first optimizer step. Values depend only on
--seed and the tensor shapes, not on how the base model happened to load.

Works for any RWKV7 size (--model-id/--model-revision, --base-model): the layer
count and width come from the model itself. The base model is only used for its
module structure, but the worker's loader puts it on cuda:0, so run this where a
worker runs and copy the directory to the coordinator host if that is a
different machine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import peft
import torch
import transformers

from g1j_dist_coordinator import default_initial_adapter_dir, load_initial_theta, read_model_shape
from g1j_dist_worker import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_worker_model,
    trainable_parameters,
)
from rwkv_dist_protocol import default_model_dir, sha256_bytes, tensor_map_sha256


DEFAULT_SEED = 20261018
RECORD_NAME = "adapter_init.json"
WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"


def initialise_lora(params: dict[str, torch.nn.Parameter], seed: int) -> None:
    """Kaiming-uniform lora_A, zero lora_B, deterministic in (seed, shapes)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name in sorted(params):
            parameter = params[name]
            if ".lora_A." in name:
                values = torch.empty(tuple(parameter.shape), dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(values, a=math.sqrt(5), generator=generator)
                parameter.copy_(values)
            elif ".lora_B." in name:
                parameter.zero_()
            else:
                raise RuntimeError(f"trainable parameter outside the LoRA schema: {name}")


def check_matches_worker(theta: dict[str, torch.Tensor],
                         params: dict[str, torch.nn.Parameter]) -> None:
    """The saved adapter must name exactly the tensors a worker will train."""
    if set(theta) != set(params):
        missing = sorted(set(params) - set(theta))[:4]
        extra = sorted(set(theta) - set(params))[:4]
        raise RuntimeError(f"adapter does not match the worker's trainable parameters; "
                           f"missing={missing}, extra={extra}")
    for name, parameter in params.items():
        if tuple(theta[name].shape) != tuple(parameter.shape):
            raise RuntimeError(f"shape mismatch for {name}: "
                               f"{tuple(theta[name].shape)} vs {tuple(parameter.shape)}")


def check_clean_init(theta: dict[str, torch.Tensor]) -> None:
    """lora_B all zero (adapted model == base model) and lora_A inside its init bound."""
    for name, tensor in theta.items():
        if ".lora_B." in name:
            if int(tensor.count_nonzero()) != 0:
                raise RuntimeError(f"{name} is not zero; the adapter would change the base model")
        else:
            bound = 1.0 / math.sqrt(tensor.shape[1])
            if int(tensor.count_nonzero()) == 0 or float(tensor.abs().max()) > bound * (1 + 1e-6):
                raise RuntimeError(f"{name} is not a Kaiming-uniform draw (bound {bound:.6g})")


def release_manifest_sha256(model_path: Path) -> str | None:
    manifest = model_path / "release-manifest.json"
    return sha256_bytes(manifest.read_bytes()) if manifest.is_file() else None


def refuse_to_overwrite(output_dir: Path, force: bool) -> None:
    existing = output_dir / WEIGHTS_NAME
    if existing.exists() and not force:
        raise FileExistsError(
            f"{existing} already exists; a run may be using it. "
            f"Pass --force to replace it or choose another --output-dir"
        )


def publish(stage: Path, output_dir: Path) -> None:
    """Move the staged files into place; weights first, so a config never outruns them."""
    if not output_dir.exists():
        os.replace(stage, output_dir)
        return
    for name in (WEIGHTS_NAME, CONFIG_NAME, RECORD_NAME):
        os.replace(stage / name, output_dir / name)
    shutil.rmtree(stage)


def build_initial_adapter(model_path: Path, output_dir: Path, *, seed: int, force: bool = False,
                          model_id: str = DEFAULT_MODEL_ID,
                          model_revision: str = DEFAULT_MODEL_REVISION) -> dict:
    refuse_to_overwrite(output_dir, force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{int(time.time())}"
    stage.mkdir()
    try:
        model = load_worker_model(model_path)
        params = trainable_parameters(model)
        wrong_dtype = sorted(name for name, p in params.items() if p.dtype != torch.float32)
        if wrong_dtype:
            raise RuntimeError(
                f"{len(wrong_dtype)} adapter tensors are not float32 (first: {wrong_dtype[0]}); "
                f"the protocol keeps theta in float32"
            )
        initialise_lora(params, seed)
        model.save_pretrained(str(stage), safe_serialization=True)

        # Check the files that were written, with the loader the coordinator will use.
        theta = load_initial_theta(stage, model_shape=read_model_shape(model_path))
        check_matches_worker(theta, params)
        check_clean_init(theta)
        theta_hash = tensor_map_sha256(theta)
        record = {
            "seed": seed,
            "theta_hash": theta_hash,
            "tensors": len(theta),
            "parameters": sum(t.numel() for t in theta.values()),
            "init": "lora_A kaiming_uniform(a=sqrt(5)), lora_B zeros",
            "base_model": {
                "repo": model_id, "revision": model_revision, "path": str(model_path),
                "release_manifest_sha256": release_manifest_sha256(model_path),
            },
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__},
        }
        (stage / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
        publish(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="base model repository, recorded with the adapter (default: %(default)s)")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="immutable commit hash of --model-id, recorded with the adapter")
    parser.add_argument("--base-model", type=Path, default=None,
                        help="local directory of the base model (default: /workspace/<model name>)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write the adapter; the coordinator's --initial-adapter-dir "
                             "default (default: artifacts/<model name>_lora_init)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for the lora_A draw (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help=f"replace an existing {WEIGHTS_NAME}")
    args = parser.parse_args(argv)

    if args.base_model is None:
        args.base_model = default_model_dir(args.model_id)
    if args.output_dir is None:
        args.output_dir = default_initial_adapter_dir(args.model_id)
    if not torch.cuda.is_available():
        parser.error("CUDA is required: the worker's model loader places the base on cuda:0. "
                     "Run this on a worker machine and copy the directory to the coordinator host")
    if not args.base_model.is_dir():
        parser.error(f"base model directory not found: {args.base_model}")
    try:
        refuse_to_overwrite(args.output_dir, args.force)
    except FileExistsError as exc:
        parser.error(str(exc))

    record = build_initial_adapter(args.base_model, args.output_dir, seed=args.seed,
                                   force=args.force, model_id=args.model_id,
                                   model_revision=args.model_revision)
    print(f"wrote {args.output_dir}: {record['tensors']} tensors, "
          f"{record['parameters']:,} parameters, seed {record['seed']}")
    print(f"theta_hash {record['theta_hash']}")
    print(f"start the coordinator with --initial-adapter-dir {args.output_dir} "
          f"(same --model-id/--model-revision as above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
