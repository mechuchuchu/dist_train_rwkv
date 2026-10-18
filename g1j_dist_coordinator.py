#!/usr/bin/env python3
"""Development coordinator for a single LoRA-SFT round over authenticated HTTP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
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
from rwkv_dist_client import WORKER_BUNDLE_FILES
from rwkv_hub import ensure_hf_checkpoint, push_adapter_snapshot


MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
MODEL_PATH = Path("/workspace/RWKV7-G1j-1.5B-20260831")
def worker_bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in WORKER_BUNDLE_FILES:
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
                 adapter_config_dir: Path, manifest: dict,
                 auto_training_complete_on_round_close: bool,
                 hf_adapter_repo: str | None = None, hf_token: str | None = None,
                 hf_adapter_private: bool = True):
        self.coordinator = coordinator
        self.token = token
        self.run_dir = run_dir
        self.adapter_config_dir = adapter_config_dir
        self.manifest = manifest
        self.auto_training_complete_on_round_close = auto_training_complete_on_round_close
        self.hf_adapter_repo = hf_adapter_repo
        self.hf_token = hf_token
        self.hf_adapter_private = hf_adapter_private
        self.last_hub_push_fingerprint: tuple[int, str] | None = None
        self.active_hub_push_fingerprint: tuple[int, str] | None = None
        self.hub_push_thread: threading.Thread | None = None
        self.hub_push_status = {
            "enabled": hf_adapter_repo is not None,
            "status": "pending" if hf_adapter_repo is not None else "disabled",
        }
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
                "training_complete": self.coordinator.training_complete,
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
            summary["hub_push"] = self.hub_push_status
            (self.run_dir / "coordinator_state.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            self.persist_events()

    def _write_hub_status_locked(self) -> None:
        state_path = self.run_dir / "coordinator_state.json"
        try:
            state = json.loads(state_path.read_text())
        except FileNotFoundError:
            state = {}
        state["hub_push"] = self.hub_push_status
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temp.replace(state_path)

    def start_hub_push(self) -> dict:
        """Queue an immutable adapter snapshot and upload it without blocking workers."""
        if self.hf_adapter_repo is None:
            return dict(self.hub_push_status)
        snapshot_dir = None
        with self.persist_lock:
            fingerprint = (self.coordinator.theta_version, self.coordinator.theta_hash)
            if fingerprint == self.last_hub_push_fingerprint:
                return dict(self.hub_push_status)
            if fingerprint == self.active_hub_push_fingerprint:
                return dict(self.hub_push_status)

            path_in_repo = (
                f"checkpoints/{self.coordinator.run_id}/"
                f"theta-{self.coordinator.theta_version:06d}"
            )
            try:
                snapshot_dir = Path(tempfile.mkdtemp(prefix="rwkv-hub-push-"))
                shutil.copy2(self.run_dir / "adapter_config.json", snapshot_dir / "adapter_config.json")
                shutil.copy2(self.run_dir / "adapter_model.safetensors", snapshot_dir / "adapter_model.safetensors")
            except Exception as exc:
                if snapshot_dir is not None:
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                error_message = str(exc)
                if self.hf_token:
                    error_message = error_message.replace(self.hf_token, "[REDACTED]")
                self.hub_push_status = {
                    "enabled": True,
                    "status": "failed",
                    "repo_id": self.hf_adapter_repo,
                    "theta_version": self.coordinator.theta_version,
                    "theta_hash": self.coordinator.theta_hash,
                    "error": f"{type(exc).__name__}: {error_message}",
                }
                self._write_hub_status_locked()
                return dict(self.hub_push_status)
            metadata = {
                "run_id": self.coordinator.run_id,
                "round_id": self.coordinator.round_id,
                "theta_version": self.coordinator.theta_version,
                "theta_hash": self.coordinator.theta_hash,
                "base": self.coordinator.base,
                "tokenizer": self.coordinator.tokenizer,
                "data_manifest_sha256": self.manifest["manifest_sha256"],
                "close_metrics": self.coordinator.close_metrics,
            }
            self.active_hub_push_fingerprint = fingerprint
            self.hub_push_status = {
                "enabled": True,
                "status": "queued",
                "repo_id": self.hf_adapter_repo,
                "path_in_repo": path_in_repo,
                "theta_version": self.coordinator.theta_version,
                "theta_hash": self.coordinator.theta_hash,
            }
            self._write_hub_status_locked()
            self.hub_push_thread = threading.Thread(
                target=self._upload_hub_snapshot,
                args=(snapshot_dir, metadata, fingerprint, path_in_repo),
                name="rwkv-hub-adapter-push",
                daemon=True,
            )
            self.hub_push_thread.start()
            return dict(self.hub_push_status)

    def _upload_hub_snapshot(self, snapshot_dir: Path, metadata: dict,
                             fingerprint: tuple[int, str], path_in_repo: str) -> None:
        try:
            commit_url = push_adapter_snapshot(
                repo_id=self.hf_adapter_repo or "",
                token=self.hf_token or "",
                adapter_config_path=snapshot_dir / "adapter_config.json",
                adapter_weights_path=snapshot_dir / "adapter_model.safetensors",
                metadata=metadata,
                path_in_repo=path_in_repo,
                base_model_id=self.coordinator.base["repo"],
                base_model_revision=self.coordinator.base["revision"],
                private=self.hf_adapter_private,
            )
            status = {
                "enabled": True,
                "status": "uploaded",
                "repo_id": self.hf_adapter_repo,
                "path_in_repo": path_in_repo,
                "theta_version": fingerprint[0],
                "theta_hash": fingerprint[1],
                "commit_url": commit_url,
            }
        except Exception as exc:
            error_message = str(exc)
            if self.hf_token:
                error_message = error_message.replace(self.hf_token, "[REDACTED]")
            status = {
                "enabled": True,
                "status": "failed",
                "repo_id": self.hf_adapter_repo,
                "path_in_repo": path_in_repo,
                "theta_version": fingerprint[0],
                "theta_hash": fingerprint[1],
                "error": f"{type(exc).__name__}: {error_message}",
            }
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        with self.persist_lock:
            if status["status"] == "uploaded" and (
                self.last_hub_push_fingerprint is None
                or fingerprint[0] >= self.last_hub_push_fingerprint[0]
            ):
                self.last_hub_push_fingerprint = fingerprint
            if self.active_hub_push_fingerprint == fingerprint:
                self.active_hub_push_fingerprint = None
                self.hub_push_status = status
                self._write_hub_status_locked()
        print(json.dumps({"hub_push": status}, ensure_ascii=False), flush=True)


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
                    if desc["training_complete"]:
                        status = "training_complete"
                    else:
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
            if route == "/hub-push":
                if app.hf_adapter_repo is None:
                    self._json(409, {"error": "hub_push_disabled"})
                    return
                app.persist_checkpoint()
                status = app.start_hub_push()
                self._json(202, {"hub_push": status})
                return
            if route == "/training-complete":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 0 or size > 1024 * 1024:
                        raise ProtocolError("invalid_payload_size", "completion body must be at most 1 MiB", 413)
                    metadata = json.loads(self.rfile.read(size)) if size else {}
                    result = app.coordinator.signal_training_complete(
                        str(metadata.get("reason", "training_finished"))
                    )
                    app.persist_events()
                    app.persist_checkpoint()
                    self._json(200, result)
                    app.start_hub_push()
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
                if result["round_closed"]:
                    if app.auto_training_complete_on_round_close:
                        app.coordinator.signal_training_complete("single_round_budget_satisfied")
                    app.persist_checkpoint()
                else:
                    app.persist_events()
                result["training_complete"] = app.coordinator.training_complete
                self._json(200, result)
                if result["round_closed"]:
                    app.start_hub_push()
            except ProtocolError as exc:
                app.persist_events()
                self._json(exc.status, {"error": exc.code, "message": str(exc)})
            except Exception as exc:
                self._json(400, {"error": "bad_request", "message": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION,
                        help="immutable Hugging Face commit SHA")
    parser.add_argument("--model-dir", type=Path, default=MODEL_PATH)
    parser.add_argument("--fetch-model-from-hf", action="store_true",
                        help="download the pinned model snapshot before starting the coordinator")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--push-adapter-to-hf", action="store_true",
                        help="push each newly saved adapter version to a Hugging Face model repository")
    parser.add_argument("--hf-adapter-repo", default=None,
                        help="destination Hugging Face model repository, e.g. username/rwkv-adapter")
    parser.add_argument("--hf-adapter-public", action="store_true",
                        help="create the destination repository as public (private by default)")
    parser.add_argument("--initial-adapter-dir", type=Path,
                        default=Path("artifacts/g1j_1.5b_capybara_sft_protocol"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/g1j_local_protocol/round-0001"))
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17821)
    parser.add_argument("--token", default=os.environ.get("RWKV_PROTOCOL_TOKEN"))
    parser.add_argument("--lease-seconds", type=float, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--manual-training-complete", action="store_true",
                        help="wait for POST /training-complete after the final round")
    args = parser.parse_args()
    if not args.token:
        parser.error("set --token or RWKV_PROTOCOL_TOKEN")
    if args.lease_seconds <= 0 or args.learning_rate <= 0:
        parser.error("lease seconds and learning rate must be positive")
    if args.push_adapter_to_hf and not args.hf_adapter_repo:
        parser.error("--push-adapter-to-hf requires --hf-adapter-repo")
    if args.push_adapter_to_hf and not args.hf_token:
        parser.error("--push-adapter-to-hf requires --hf-token or HF_TOKEN (coordinator only)")
    if args.hf_adapter_repo and not args.push_adapter_to_hf:
        parser.error("--hf-adapter-repo requires --push-adapter-to-hf")
    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        parser.error("this coordinator prototype is pinned to the G1j 1.5B repo and revision")
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: this development server is plain HTTP. Expose it only behind TLS/authenticated networking.", flush=True)
    manifest = read_manifest(args.units)
    if manifest["model"]["repo"] != args.model_id or manifest["model"]["revision"] != args.model_revision:
        raise ValueError("tokenized data manifest does not match the requested model id/revision")
    model_dir = ensure_hf_checkpoint(
        args.model_id, args.model_revision, args.model_dir,
        fetch_from_hf=args.fetch_model_from_hf,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    release_manifest_path = model_dir / "release-manifest.json"
    base_manifest = json.loads(release_manifest_path.read_text())
    if manifest["model"]["release_manifest_sha256"] != sha256_bytes(release_manifest_path.read_bytes()):
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
    app = Application(
        coordinator, token=args.token, run_dir=args.run_dir,
        adapter_config_dir=args.initial_adapter_dir, manifest=manifest,
        auto_training_complete_on_round_close=not args.manual_training_complete,
        hf_adapter_repo=args.hf_adapter_repo if args.push_adapter_to_hf else None,
        hf_token=args.hf_token if args.push_adapter_to_hf else None,
        hf_adapter_private=not args.hf_adapter_public,
    )
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(app))
    print(json.dumps({
        "listen": f"http://{args.bind}:{args.port}",
        "run_dir": str(args.run_dir.resolve()),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_dir": str(model_dir),
        "fetched_from_hf": args.fetch_model_from_hf,
        "run_id": coordinator.run_id,
        "round_id": coordinator.round_id,
        "theta_hash": coordinator.theta_hash,
        "target_loss_tokens": target_tokens,
        "min_units": coordinator.min_units,
        "worker_code_sha256": worker_hash,
        "data_manifest_sha256": manifest["manifest_sha256"],
        "training_completion": (
            "manual POST /training-complete" if args.manual_training_complete
            else "signaled automatically when this single-round budget closes"
        ),
        "adapter_hub_push": (
            {"enabled": True, "repo_id": args.hf_adapter_repo,
             "private_if_created": not args.hf_adapter_public,
             "cadence": "after each locally persisted adapter version"}
            if args.push_adapter_to_hf else {"enabled": False}
        ),
    }, indent=2), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
