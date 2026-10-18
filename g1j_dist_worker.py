#!/usr/bin/env python3
"""Compute pinned G1j SFT unit gradients and return them to the coordinator."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from rwkv_dist_client import (
    Client,
    CoordinatorHTTPError,
    CoordinatorUnavailable,
    CoordinatorWaitExpired,
    CoordinatorWaiter,
    StaleLease,
    WORKER_BUNDLE_FILES,
    check_lease_current,
    fetch_leased_snapshot,
    release_stale_lease,
)
from rwkv_dist_protocol import (
    decode_tensors,
    encode_tensors,
    sha256_bytes,
    tensor_map_sha256,
    unit_digest,
)
from rwkv_hub import ensure_hf_checkpoint


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
TARGET_MODULES = ["att.receptance", "att.key", "att.value", "att.output"]


def bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in WORKER_BUNDLE_FILES:
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def load_worker_model(model_path: Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
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
            target_modules=TARGET_MODULES,
        ),
    )
    model.train()
    return model


def trainable_parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def apply_snapshot(model, payload: bytes, expected_hash: str) -> dict[str, torch.nn.Parameter]:
    snapshot = decode_tensors(payload)
    actual_hash = tensor_map_sha256(snapshot)
    if actual_hash != expected_hash:
        raise RuntimeError(f"snapshot hash mismatch: {actual_hash} != {expected_hash}")
    params = trainable_parameters(model)
    if set(params) != set(snapshot):
        missing = sorted(set(params) - set(snapshot))[:4]
        extra = sorted(set(snapshot) - set(params))[:4]
        raise RuntimeError(f"snapshot tensor schema mismatch; missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in params.items():
            value = snapshot[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"snapshot shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return params


def compute_unit(model, params, unit: dict):
    if unit_digest(unit) != unit.get("unit_sha256"):
        raise RuntimeError(f"unit content hash mismatch for {unit.get('unit_id')}")
    input_ids = torch.tensor(unit["input_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
    labels = torch.tensor(unit["labels"], dtype=torch.long, device="cuda:0").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    loss_tokens = int(labels[:, 1:].ne(-100).sum().item())
    if loss_tokens != int(unit["loss_tokens"]):
        raise RuntimeError("unit manifest loss-token count does not match labels")
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shifted_logits = output.logits[:, :-1, :].contiguous().float()
    shifted_labels = labels[:, 1:].contiguous()
    loss_sum = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="sum",
        ignore_index=-100,
    )
    loss_value = float(loss_sum.detach().item())
    loss_sum.backward()
    gradients = {}
    for name, parameter in params.items():
        if parameter.grad is None:
            gradient = torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        else:
            gradient = parameter.grad.detach().float().cpu().contiguous()
        if not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"non-finite worker gradient: {name}")
        gradients[name] = gradient
    grad_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients.values()])).item()
    if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm == 0:
        raise RuntimeError("worker gradient norm must be finite and nonzero")
    payload = encode_tensors(gradients)
    del output, shifted_logits, shifted_labels, loss_sum, input_ids, labels, attention_mask
    return payload, loss_value, loss_tokens, grad_norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:17821")
    parser.add_argument("--token", default=os.environ.get("RWKV_PROTOCOL_TOKEN"))
    parser.add_argument("--worker-id", default=f"worker-{os.getpid()}")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--fetch-model-from-hf", action="store_true",
                        help="download the pinned model revision to --model-path before loading")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--max-units", type=int, default=0, help="0 means drain the open round")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--coordinator-down-timeout-seconds", type=float, default=10800.0,
                        help="stop after this much coordinator downtime; maximum is 10800 (3 hours)")
    args = parser.parse_args()
    if not args.token:
        parser.error("set --token or RWKV_PROTOCOL_TOKEN")
    if not torch.cuda.is_available():
        raise RuntimeError("worker requires a CUDA GPU")

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    client = Client(args.server, args.token)
    waiter = CoordinatorWaiter(
        client,
        max_wait_seconds=args.coordinator_down_timeout_seconds,
        on_status=lambda status, waited, error: print(json.dumps({
            "status": status,
            "waited_seconds": round(waited, 1),
            "detail": error,
        }), flush=True),
    )
    bundle_hash = bundle_sha256(Path(__file__).parent)
    descriptor = waiter.get_json("/round")
    run_id = descriptor["run_id"]
    if descriptor["base"]["repo"] != MODEL_ID or descriptor["base"]["revision"] != MODEL_REVISION:
        raise RuntimeError("coordinator base model identity does not match this worker")
    if descriptor["worker_code_sha256"] != bundle_hash:
        raise RuntimeError("worker bundle hash does not match coordinator's pinned code")
    query = urllib.parse.urlencode({"worker_id": args.worker_id})
    waiter.get_json(f"/join?{query}")
    model_path = ensure_hf_checkpoint(
        MODEL_ID, MODEL_REVISION, args.model_path,
        fetch_from_hf=args.fetch_model_from_hf,
        token=False,
        cache_dir=args.hf_cache_dir,
    )
    manifest_path = model_path / "release-manifest.json"
    actual_model_hash = sha256_bytes(manifest_path.read_bytes())
    if descriptor["base"]["release_manifest_sha256"] != actual_model_hash:
        raise RuntimeError("local base weights do not match the coordinator's pinned manifest")
    if descriptor["training_complete"]:
        print(json.dumps({"status": "training_complete", "run_id": run_id}), flush=True)
        return

    print(f"loading {MODEL_ID} on {torch.cuda.get_device_name(0)}", flush=True)
    model = load_worker_model(model_path)
    params = trainable_parameters(model)
    loaded_theta_hash = None
    processed = 0
    last_waiting_round = None
    while args.max_units == 0 or processed < args.max_units:
        join_status = waiter.get_json(f"/join?{query}")
        if join_status["status"] == "training_complete":
            print(json.dumps({"status": "training_complete", "run_id": run_id,
                              "processed_units": processed}), flush=True)
            break
        if join_status["status"] == "waiting_next_round":
            eligible_round = join_status["round_id"]
            if last_waiting_round != eligible_round:
                print(json.dumps({"status": "waiting_next_round",
                                  "eligible_round_id": eligible_round}), flush=True)
                last_waiting_round = eligible_round
            time.sleep(args.poll_seconds)
            continue
        last_waiting_round = None
        descriptor = waiter.get_json("/round")
        if descriptor["run_id"] != run_id:
            raise RuntimeError(
                f"coordinator run changed after reconnect: {run_id} -> {descriptor['run_id']}"
            )
        if descriptor["training_complete"]:
            print(json.dumps({"status": "training_complete", "run_id": run_id,
                              "processed_units": processed}), flush=True)
            break
        if descriptor["status"] == "closed":
            time.sleep(args.poll_seconds)
            continue
        lease_response = waiter.get_json(f"/lease?{query}")
        if lease_response["status"] == "training_complete":
            print(json.dumps({"status": "training_complete", "run_id": run_id,
                              "processed_units": processed}), flush=True)
            break
        if lease_response["status"] == "round_closed":
            if lease_response.get("round", {}).get("training_complete"):
                print(json.dumps({"status": "training_complete", "run_id": run_id,
                                  "processed_units": processed}), flush=True)
                break
            time.sleep(args.poll_seconds)
            continue
        if lease_response["status"] == "waiting":
            time.sleep(args.poll_seconds)
            continue

        lease = lease_response
        try:
            check_lease_current(lease, descriptor)
            if loaded_theta_hash != lease["theta_hash"]:
                snapshot_payload = fetch_leased_snapshot(waiter.request, lease)
                # A payload that fails its hash/schema check here is real
                # corruption, not a race, so apply_snapshot's RuntimeError
                # is deliberately left to propagate.
                params = apply_snapshot(model, snapshot_payload, lease["theta_hash"])
                loaded_theta_hash = lease["theta_hash"]
        except StaleLease as exc:
            # The round advanced between /round, /lease and /snapshot. Drop
            # this lease and re-sync instead of exiting the worker.
            outcome = release_stale_lease(client, lease, args.worker_id, str(exc))
            print(json.dumps({"status": "stale_lease_dropped",
                              "unit_id": lease["unit"]["unit_id"],
                              "lease_id": lease["lease_id"],
                              "detail": str(exc),
                              "release": outcome}), flush=True)
            time.sleep(args.poll_seconds)
            continue
        unit = lease["unit"]
        heartbeat_stop = threading.Event()
        heartbeat_errors = []

        def heartbeat_loop():
            interval = max(1.0, min(10.0, float(lease["lease_seconds"]) / 3.0))
            while not heartbeat_stop.wait(interval):
                body = json.dumps({
                    "lease_id": lease["lease_id"],
                    "worker_id": args.worker_id,
                }).encode()
                try:
                    client.request("/heartbeat", method="POST", body=body,
                                   headers={"Content-Type": "application/json"},
                                   timeout=min(10.0, interval))
                    waiter.note_recovered()
                except CoordinatorUnavailable as exc:
                    waiter.note_unavailable(exc)
                    heartbeat_errors.append(str(exc))
                    return
                except CoordinatorHTTPError as exc:
                    heartbeat_errors.append(str(exc))
                    return

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        started = time.perf_counter()
        interrupted = False
        try:
            gradient_payload, loss_sum, loss_tokens, grad_norm = compute_unit(model, params, unit)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join()
        if interrupted:
            body = json.dumps({
                "lease_id": lease["lease_id"],
                "worker_id": args.worker_id,
                "reason": "worker_shutdown_during_unit",
            }).encode()
            try:
                _, _, release_body = client.request(
                    "/release", method="POST", body=body,
                    headers={"Content-Type": "application/json"}, timeout=10.0,
                )
                print(json.dumps({"status": "released", **json.loads(release_body)}), flush=True)
            except Exception as exc:
                print(f"Could not release lease explicitly; it will expire: {exc}", flush=True)
            return
        if heartbeat_errors:
            print(json.dumps({"status": "heartbeat_interrupted",
                              "detail": heartbeat_errors[0],
                              "action": "submit_result_or_reacquire_after_reconnect"}), flush=True)
        metadata = {
            "unit_id": unit["unit_id"],
            "unit_sha256": unit["unit_sha256"],
            "lease_id": lease["lease_id"],
            "worker_id": args.worker_id,
            "round_id": lease["round_id"],
            "theta_version": lease["theta_version"],
            "theta_hash": lease["theta_hash"],
            "worker_code_sha256": bundle_hash,
            "loss_tokens": loss_tokens,
            "loss_sum": loss_sum,
            "gradient_norm": grad_norm,
            "gradient_sha256": sha256_bytes(gradient_payload),
            "compute_seconds": round(time.perf_counter() - started, 4),
        }
        encoded_metadata = base64.urlsafe_b64encode(json.dumps(metadata, separators=(",", ":")).encode()).decode()
        try:
            _, _, result_body = waiter.request(
                "/result",
                method="POST",
                body=gradient_payload,
                headers={
                    "Content-Type": "application/vnd.safetensors",
                    "X-Result-Metadata": encoded_metadata,
                },
            )
        except CoordinatorHTTPError as exc:
            error_code = exc.detail.get("error") if isinstance(exc.detail, dict) else None
            if exc.status == 409 and error_code in {
                "unknown_lease", "expired_lease", "round_closed"
            }:
                print(json.dumps({"status": "result_discarded_lease_lost",
                                  "unit_id": unit["unit_id"],
                                  "error": error_code}), flush=True)
                continue
            raise
        result = json.loads(result_body)
        processed += 1
        print(json.dumps({
            "worker_id": args.worker_id,
            "unit_id": unit["unit_id"],
            "loss_tokens": loss_tokens,
            "mean_loss": loss_sum / loss_tokens,
            "gradient_norm": grad_norm,
            "compute_seconds": metadata["compute_seconds"],
            "accepted": result["status"],
            "round_closed": result["round_closed"],
            "theta_hash": result["theta_hash"],
        }), flush=True)
    final = waiter.get_json("/round")
    if final["run_id"] != run_id:
        raise RuntimeError(f"coordinator run changed after reconnect: {run_id} -> {final['run_id']}")
    print(json.dumps({
        "status": final["status"],
        "training_complete": final["training_complete"],
        "processed_units": processed,
        "received_units": final["received_units"],
        "received_loss_tokens": final["received_loss_tokens"],
        "close_metrics": final["close_metrics"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except CoordinatorWaitExpired as exc:
        print(json.dumps({
            "status": "coordinator_wait_timeout",
            "waited_seconds": round(exc.waited_seconds, 1),
            "action": "worker_exit",
        }), flush=True)
        sys.exit(0)
