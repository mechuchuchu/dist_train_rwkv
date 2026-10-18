#!/usr/bin/env python3
"""Compute pinned G1j SFT unit gradients and return them to the coordinator."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from rwkv_dist_protocol import (
    decode_tensors,
    encode_tensors,
    sha256_bytes,
    tensor_map_sha256,
    unit_digest,
)


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
TARGET_MODULES = ["att.receptance", "att.key", "att.value", "att.output"]


def bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("g1j_dist_worker.py", "rwkv_dist_protocol.py"):
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class Client:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, path: str, *, method="GET", body: bytes | None = None,
                headers: dict[str, str] | None = None):
        request_headers = {"Authorization": f"Bearer {self.token}"}
        request_headers.update(headers or {})
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                detail = json.loads(payload)
            except Exception:
                detail = payload[:1000].decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc

    def get_json(self, path: str):
        _, _, body = self.request(path)
        return json.loads(body)


def load_worker_model() -> torch.nn.Module:
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
    parser.add_argument("--max-units", type=int, default=0, help="0 means drain the open round")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not args.token:
        parser.error("set --token or RWKV_PROTOCOL_TOKEN")
    if args.model_path != MODEL_PATH:
        raise ValueError("this first worker is pinned to the validated G1j 1.5B local model path")
    if not torch.cuda.is_available():
        raise RuntimeError("worker requires a CUDA GPU")

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    client = Client(args.server, args.token)
    bundle_hash = bundle_sha256(Path(__file__).parent)
    descriptor = client.get_json("/round")
    if descriptor["base"]["repo"] != MODEL_ID or descriptor["base"]["revision"] != MODEL_REVISION:
        raise RuntimeError("coordinator base model identity does not match this worker")
    manifest_path = args.model_path / "release-manifest.json"
    actual_model_hash = sha256_bytes(manifest_path.read_bytes())
    if descriptor["base"]["release_manifest_sha256"] != actual_model_hash:
        raise RuntimeError("local base weights do not match the coordinator's pinned manifest")
    if descriptor["worker_code_sha256"] != bundle_hash:
        raise RuntimeError("worker bundle hash does not match coordinator's pinned code")
    if descriptor["status"] == "closed":
        print(json.dumps({"status": "round_already_closed", "run_id": descriptor["run_id"]}))
        return

    print(f"loading {MODEL_ID} on {torch.cuda.get_device_name(0)}", flush=True)
    model = load_worker_model()
    params = trainable_parameters(model)
    loaded_theta_hash = None
    processed = 0
    while args.max_units == 0 or processed < args.max_units:
        descriptor = client.get_json("/round")
        if descriptor["status"] == "closed":
            break
        query = urllib.parse.urlencode({"worker_id": args.worker_id})
        lease_response = client.get_json(f"/lease?{query}")
        if lease_response["status"] == "round_closed":
            break
        if lease_response["status"] == "waiting":
            time.sleep(args.poll_seconds)
            continue

        lease = lease_response
        if lease["theta_hash"] != descriptor["theta_hash"] or lease["theta_version"] != descriptor["theta_version"]:
            raise RuntimeError("lease was issued against a different adapter snapshot")
        if loaded_theta_hash != lease["theta_hash"]:
            _, snapshot_headers, snapshot_payload = client.request("/snapshot")
            header_hash = snapshot_headers.get("X-Theta-Hash")
            if header_hash != lease["theta_hash"]:
                raise RuntimeError("snapshot response does not match leased theta hash")
            params = apply_snapshot(model, snapshot_payload, lease["theta_hash"])
            loaded_theta_hash = lease["theta_hash"]
        unit = lease["unit"]
        heartbeat_stop = threading.Event()
        heartbeat_errors = []

        def heartbeat_loop():
            interval = max(1.0, float(lease["lease_seconds"]) / 3.0)
            while not heartbeat_stop.wait(interval):
                body = json.dumps({
                    "lease_id": lease["lease_id"],
                    "worker_id": args.worker_id,
                }).encode()
                try:
                    client.request("/heartbeat", method="POST", body=body,
                                   headers={"Content-Type": "application/json"})
                except Exception as exc:
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
                    headers={"Content-Type": "application/json"},
                )
                print(json.dumps({"status": "released", **json.loads(release_body)}), flush=True)
            except Exception as exc:
                print(f"Could not release lease explicitly; it will expire: {exc}", flush=True)
            return
        if heartbeat_errors:
            raise RuntimeError(f"lease heartbeat failed: {heartbeat_errors[0]}")
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
        _, _, result_body = client.request(
            "/result",
            method="POST",
            body=gradient_payload,
            headers={
                "Content-Type": "application/vnd.safetensors",
                "X-Result-Metadata": encoded_metadata,
            },
        )
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
        if result["round_closed"]:
            break

    final = client.get_json("/round")
    print(json.dumps({
        "status": final["status"],
        "processed_units": processed,
        "received_units": final["received_units"],
        "received_loss_tokens": final["received_loss_tokens"],
        "close_metrics": final["close_metrics"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
