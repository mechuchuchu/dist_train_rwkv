#!/usr/bin/env python3
"""Development coordinator for a single LoRA-SFT round over authenticated HTTP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from safetensors.torch import load_file, save_file

from rwkv_dist_protocol import (
    ProtocolError,
    RoundCoordinator,
    canonical_json,
    decode_tensors,
    encode_tensors,
    parameter_name_to_peft_saved_name,
    sha256_bytes,
    tensor_map_sha256,
)


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
WORKER_BUNDLE = ("g1j_dist_worker.py", "rwkv_dist_protocol.py")


def worker_bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in WORKER_BUNDLE:
        path = root / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    supplied_hash = manifest.get("manifest_sha256")
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = sha256_bytes(canonical_json(content))
    if supplied_hash != actual_hash:
        raise ValueError(f"data manifest hash mismatch: {supplied_hash} != {actual_hash}")
    if manifest["model"]["repo"] != MODEL_ID or manifest["model"]["revision"] != MODEL_REVISION:
        raise ValueError("tokenized shard belongs to a different base model")
    if not manifest.get("units"):
        raise ValueError("tokenized shard is empty")
    return manifest


def load_initial_theta(adapter_dir: Path) -> dict[str, torch.Tensor]:
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if int(adapter_config.get("r", -1)) != 8 or int(adapter_config.get("lora_alpha", -1)) != 16:
        raise ValueError("initial adapter must be rank 8 / alpha 16")
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    theta = {}
    for name, tensor in state.items():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
            from rwkv_dist_protocol import peft_saved_name_to_parameter_name
            name = peft_saved_name_to_parameter_name(name)
        theta[name] = tensor.float().contiguous()
    if len(theta) != 192 or sum(t.numel() for t in theta.values()) != 3_145_728:
        raise ValueError(f"unexpected initial adapter tensor schema: {len(theta)} tensors")
    return theta


class Application:
    def __init__(self, coordinator: RoundCoordinator, *, token: str, run_dir: Path,
                 adapter_config_dir: Path, manifest: dict):
        self.coordinator = coordinator
        self.token = token
        self.run_dir = run_dir
        self.adapter_config_dir = adapter_config_dir
        self.manifest = manifest
        self.persist_lock = threading.RLock()
        self.event_cursor = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "gradients").mkdir(exist_ok=True)
        shutil.copy2(adapter_config_dir / "adapter_config.json", self.run_dir / "adapter_config.json")
        (self.run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    def persist_events(self) -> None:
        with self.persist_lock:
            events = self.coordinator.events[self.event_cursor:]
            if events:
                with (self.run_dir / "round_log.jsonl").open("a") as f:
                    for event in events:
                        f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
                self.event_cursor = len(self.coordinator.events)

    def persist_result_payload(self, unit_id: str, payload: bytes) -> None:
        safe_name = hashlib.sha256(unit_id.encode()).hexdigest()
        path = self.run_dir / "gradients" / f"{safe_name}.safetensors"
        temp = path.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(path)

    def persist_checkpoint(self) -> None:
        with self.persist_lock:
            state = {parameter_name_to_peft_saved_name(k): v for k, v in self.coordinator.theta.items()}
            adapter_path = self.run_dir / "adapter_model.safetensors"
            temp = self.run_dir / "adapter_model.tmp"
            save_file(state, str(temp))
            temp.replace(adapter_path)
            theta_path = self.run_dir / "theta.safetensors"
            temp_theta = self.run_dir / "theta.tmp"
            save_file(self.coordinator.theta, str(temp_theta))
            temp_theta.replace(theta_path)
            torch.save(self.coordinator.optimizer.state_dict(), self.run_dir / "optimizer.pt")
            summary = {
                "run_id": self.coordinator.run_id,
                "round_id": self.coordinator.round_id,
                "theta_version": self.coordinator.theta_version,
                "theta_hash": self.coordinator.theta_hash,
                "base": self.coordinator.base,
                "tokenizer": self.coordinator.tokenizer,
                "worker_code_sha256": self.coordinator.worker_code_sha256,
                "data_manifest_sha256": self.manifest["manifest_sha256"],
                "accepted_results": [
                    {key: value for key, value in result.items() if key != "gradients"}
                    for result in self.coordinator.accepted.values()
                ],
                "close_metrics": self.coordinator.close_metrics,
            }
            (self.run_dir / "coordinator_state.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            self.persist_events()


def make_handler(app: Application):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RWKV-SFT-DevCoordinator/1"

        def log_message(self, fmt, *args):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

        def _authorized(self) -> bool:
            expected = f"Bearer {app.token}"
            if self.headers.get("Authorization") != expected:
                self._json(401, {"error": "unauthorized"})
                return False
            return True

        def _json(self, status: int, value: dict):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorized():
                return
            route = urlparse(self.path)
            if route.path == "/health":
                self._json(200, {"status": "ok"})
                return
            if route.path == "/round":
                self._json(200, app.coordinator.descriptor())
                return
            if route.path == "/snapshot":
                body = app.coordinator.snapshot_bytes()
                desc = app.coordinator.descriptor()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.safetensors")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Theta-Hash", desc["theta_hash"])
                self.send_header("X-Theta-Version", str(desc["theta_version"]))
                self.end_headers()
                self.wfile.write(body)
                return
            if route.path == "/lease":
                worker_id = parse_qs(route.query).get("worker_id", [""])[0]
                try:
                    lease = app.coordinator.lease(worker_id)
                except ProtocolError as exc:
                    self._json(exc.status, {"error": exc.code, "message": str(exc)})
                    return
                app.persist_events()
                if lease is None:
                    desc = app.coordinator.descriptor()
                    status = "round_closed" if desc["status"] == "closed" else "waiting"
                    self._json(200, {"status": status, "round": desc})
                else:
                    self._json(200, {"status": "leased", **lease})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self):
            if not self._authorized():
                return
            route = urlparse(self.path).path
            if route == "/heartbeat":
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                    if size < 1 or size > 1024 * 1024:
                        raise ProtocolError("invalid_payload_size", "heartbeat body must be 1..1 MiB", 413)
                    metadata = json.loads(self.rfile.read(size))
                    result = app.coordinator.heartbeat(
                        str(metadata.get("lease_id", "")),
                        str(metadata.get("worker_id", "")),
                    )
                    app.persist_events()
                    self._json(200, result)
                except ProtocolError as exc:
                    app.persist_events()
                    self._json(exc.status, {"error": exc.code, "message": str(exc)})
                except Exception as exc:
                    self._json(400, {"error": "bad_request", "message": str(exc)})
                return
            if route == "/release":
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                    if size < 1 or size > 1024 * 1024:
                        raise ProtocolError("invalid_payload_size", "release body must be 1..1 MiB", 413)
                    metadata = json.loads(self.rfile.read(size))
                    result = app.coordinator.release(
                        str(metadata.get("lease_id", "")),
                        str(metadata.get("worker_id", "")),
                        str(metadata.get("reason", "worker_leaving")),
                    )
                    app.persist_events()
                    self._json(200, result)
                except ProtocolError as exc:
                    app.persist_events()
                    self._json(exc.status, {"error": exc.code, "message": str(exc)})
                except Exception as exc:
                    self._json(400, {"error": "bad_request", "message": str(exc)})
                return
            if route != "/result":
                self._json(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 1 or size > 128 * 1024 * 1024:
                    raise ProtocolError("invalid_payload_size", "payload size must be 1..128 MiB", 413)
                encoded_metadata = self.headers.get("X-Result-Metadata", "")
                metadata = json.loads(base64.urlsafe_b64decode(encoded_metadata.encode("ascii")))
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ProtocolError("truncated_payload", "request body shorter than Content-Length", 400)
                result = app.coordinator.submit_result(metadata, payload)
                app.persist_result_payload(str(metadata["unit_id"]), payload)
                app.persist_events()
                if result["round_closed"]:
                    app.persist_checkpoint()
                self._json(200, result)
            except ProtocolError as exc:
                app.persist_events()
                self._json(exc.status, {"error": exc.code, "message": str(exc)})
            except Exception as exc:
                self._json(400, {"error": "bad_request", "message": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--initial-adapter-dir", type=Path,
                        default=Path("artifacts/g1j_1.5b_capybara_sft_protocol"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/g1j_local_protocol/round-0001"))
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17821)
    parser.add_argument("--token", default=os.environ.get("RWKV_PROTOCOL_TOKEN"))
    parser.add_argument("--lease-seconds", type=float, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not args.token:
        parser.error("set --token or RWKV_PROTOCOL_TOKEN")
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: this development server is plain HTTP. Expose it only behind TLS/authenticated networking.", flush=True)
    manifest = read_manifest(args.units)
    base_manifest = json.loads((MODEL_PATH / "release-manifest.json").read_text())
    if manifest["model"]["release_manifest_sha256"] != sha256_bytes((MODEL_PATH / "release-manifest.json").read_bytes()):
        raise ValueError("local base model manifest does not match the tokenized data shard")
    theta = load_initial_theta(args.initial_adapter_dir)
    target_tokens = sum(int(unit["loss_tokens"]) for unit in manifest["units"])
    worker_hash = worker_bundle_sha256(Path(__file__).parent)
    coordinator = RoundCoordinator(
        theta=theta,
        units=manifest["units"],
        run_id=f"g1j1.5b-{int(time.time())}",
        base={
            "repo": MODEL_ID,
            "revision": MODEL_REVISION,
            "release_manifest_sha256": manifest["model"]["release_manifest_sha256"],
            "serialized_parameter_count": base_manifest["conversion"]["serialized_parameter_count"],
            "dtype": "bfloat16",
        },
        tokenizer=manifest["tokenizer"],
        worker_code_sha256=worker_hash,
        target_loss_tokens=target_tokens,
        min_units=min(2, len(manifest["units"])),
        lease_seconds=args.lease_seconds,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        global_grad_clip=None,
    )
    app = Application(coordinator, token=args.token, run_dir=args.run_dir,
                      adapter_config_dir=args.initial_adapter_dir, manifest=manifest)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(app))
    print(json.dumps({
        "listen": f"http://{args.bind}:{args.port}",
        "run_dir": str(args.run_dir.resolve()),
        "run_id": coordinator.run_id,
        "round_id": coordinator.round_id,
        "theta_hash": coordinator.theta_hash,
        "target_loss_tokens": target_tokens,
        "min_units": coordinator.min_units,
        "worker_code_sha256": worker_hash,
        "data_manifest_sha256": manifest["manifest_sha256"],
    }, indent=2), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
