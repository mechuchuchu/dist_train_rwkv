#!/usr/bin/env python3
"""Development coordinator for multi-step LoRA-SFT over authenticated HTTP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from safetensors.torch import load_file, save_file

from rwkv_dist_checkpoint import (
    LoadedCheckpoint,
    ResumeError,
    check_resume_compatibility,
    list_checkpoint_versions,
    load_hub_checkpoint,
    load_latest_checkpoint,
    write_checkpoint,
)
from rwkv_dist_protocol import (
    ProtocolError,
    RoundCoordinator,
    adamw_hyperparameters,
    canonical_json,
    decode_tensors,
    encode_tensors,
    parameter_name_to_peft_saved_name,
    sha256_bytes,
    tensor_map_sha256,
    unit_digest,
)
from rwkv_dist_client import WORKER_BUNDLE_FILES
from rwkv_hub import (
    OPTIMIZER_FILE,
    OPTIMIZER_POLICIES,
    download_hub_checkpoint,
    ensure_hf_checkpoint,
    list_hub_checkpoint_versions,
    push_adapter_snapshot,
)


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


def build_round_plan(
    units: list[dict], *, epochs: int, units_per_step: int, min_units_per_step: int
) -> list[dict]:
    if epochs <= 0 or units_per_step < 0 or min_units_per_step <= 0:
        raise ValueError("epochs and min_units_per_step must be positive; units_per_step cannot be negative")
    plan = []
    for epoch in range(1, epochs + 1):
        epoch_units = []
        for source in units:
            unit = dict(source)
            unit["unit_id"] = f"epoch-{epoch:04d}/{source['unit_id']}"
            unit.pop("unit_sha256", None)
            unit["unit_sha256"] = unit_digest(unit)
            epoch_units.append(unit)
        chunk_size = units_per_step or len(epoch_units)
        for offset in range(0, len(epoch_units), chunk_size):
            step_units = epoch_units[offset:offset + chunk_size]
            plan.append({
                "units": step_units,
                "target_loss_tokens": sum(int(unit["loss_tokens"]) for unit in step_units),
                "min_units": min(min_units_per_step, len(step_units)),
                "epoch": epoch,
            })
    if not plan:
        raise ValueError("training schedule produced no optimizer steps")
    return plan


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


def round_plan_digest(plan: list[dict]) -> str:
    """Identify a training schedule by its steps, unit contents and token budgets.

    A checkpoint's version number only means something against the same plan,
    so resume refuses a checkpoint whose digest differs from this launch's.
    """
    return sha256_bytes(canonical_json([
        {
            "unit_ids": [unit["unit_id"] for unit in step["units"]],
            "unit_sha256": [unit["unit_sha256"] for unit in step["units"]],
            "target_loss_tokens": step["target_loss_tokens"],
            "min_units": step["min_units"],
        }
        for step in plan
    ]))


def build_coordinator(
    *,
    round_plan: list[dict],
    plan_sha256: str,
    manifest: dict,
    base: dict,
    worker_code_sha256: str,
    lease_seconds: float,
    learning_rate: float,
    new_run_id: str,
    initial_theta: dict[str, torch.Tensor] | None = None,
    loaded: LoadedCheckpoint | None = None,
) -> tuple[RoundCoordinator, int]:
    """Create the coordinator for a fresh run, or for the round after `loaded`.

    Returns the coordinator and the plan index of the round it opens. Whatever
    was in flight when the checkpoint was written is not restored: that round
    opens again with every unit pending, and results workers still hold for it
    are refused as belonging to unknown leases.
    """
    if loaded is None:
        if initial_theta is None:
            raise ValueError("a fresh run needs an initial theta")
        start, theta, run_id, optimizer_tensors = 0, initial_theta, new_run_id, None
    else:
        check_resume_compatibility(loaded.state, {
            "data_manifest_sha256": manifest["manifest_sha256"],
            "round_plan_sha256": plan_sha256,
            "planned_rounds": len(round_plan),
            "base": base,
            "tokenizer": manifest["tokenizer"],
            "hyperparameters": adamw_hyperparameters(learning_rate),
        })
        start = int(loaded.state["theta_version"])
        if start >= len(round_plan):
            raise ResumeError(
                f"checkpoint is at version {start} of {len(round_plan)} planned steps; "
                "this run has already finished"
            )
        theta = loaded.theta
        run_id = loaded.state["run_id"]
        optimizer_tensors = loaded.optimizer_tensors
    step = round_plan[start]
    coordinator = RoundCoordinator(
        theta=theta,
        units=step["units"],
        run_id=run_id,
        base=base,
        tokenizer=manifest["tokenizer"],
        worker_code_sha256=worker_code_sha256,
        target_loss_tokens=step["target_loss_tokens"],
        min_units=step["min_units"],
        lease_seconds=lease_seconds,
        learning_rate=learning_rate,
        weight_decay=0.0,
        global_grad_clip=None,
        theta_version=start,
        optimizer_tensors=optimizer_tensors,
    )
    if loaded is not None:
        if coordinator.theta_hash != loaded.state["theta_hash"]:
            raise ResumeError("restored adapter does not reproduce the checkpoint's theta hash")
        coordinator.events.append({
            "event": "coordinator_resumed",
            "run_id": run_id,
            "from_theta_version": start,
            "resumed_round_id": coordinator.round_id,
            "optimizer_restored": bool(loaded.optimizer_tensors),
            "in_flight_round_discarded": True,
            "source": str(loaded.directory),
            "skipped_checkpoints": list(loaded.skipped),
            "at": time.time(),
        })
    return coordinator, start


def load_resume_from_hub(
    *, repo_id: str, run_id: str, version: int | None, token: str | None, work_dir: Path
) -> LoadedCheckpoint:
    """Download and verify a pushed checkpoint, falling back to older versions.

    With `version` given, only that version is tried. Otherwise the newest
    version that verifies is used and the ones passed over are listed in
    `skipped`. Hashes catch damaged or partial uploads; they cannot tell a
    legitimate push from one written by someone else with access to the
    repository, so only resume from a repository you control.
    """
    if version is not None:
        candidates = [version]
    else:
        candidates = list_hub_checkpoint_versions(repo_id=repo_id, run_id=run_id, token=token)
        if not candidates:
            raise ResumeError(f"{repo_id} has no complete checkpoints for run {run_id}")
    skipped: list[str] = []
    for candidate in candidates:
        try:
            folder = download_hub_checkpoint(
                repo_id=repo_id, run_id=run_id, version=candidate, token=token,
                local_dir=work_dir,
            )
            loaded = load_hub_checkpoint(folder)
        except (ResumeError, OSError, ValueError) as exc:
            skipped.append(f"theta-{candidate:06d}: {exc}")
            continue
        if loaded.state["run_id"] != run_id or loaded.state["theta_version"] != candidate:
            skipped.append(
                f"theta-{candidate:06d}: metadata says run {loaded.state['run_id']!r} "
                f"version {loaded.state['theta_version']}"
            )
            continue
        loaded.skipped = skipped
        return loaded
    raise ResumeError(f"no usable Hub checkpoint for run {run_id}: " + "; ".join(skipped))


def _with_status(previous: dict, status: str) -> dict:
    """Copy a hub-push record to a new status, dropping the earlier failure's error.

    The error belongs to the attempt that failed; carrying it into
    "queued"/"uploading"/"uploaded" makes a recovered push look failed.
    """
    return {**{k: v for k, v in previous.items() if k != "error"}, "status": status}


class Application:
    def __init__(self, coordinator: RoundCoordinator, *, token: str, run_dir: Path,
                 adapter_config_dir: Path, manifest: dict,
                 auto_training_complete_on_round_close: bool,
                 hf_adapter_repo: str | None = None, hf_token: str | None = None,
                 hf_adapter_private: bool = True,
                 round_plan: list[dict] | None = None,
                 hf_push_every_steps: int = 10,
                 hf_push_optimizer: str = "auto",
                 start_round_index: int = 0,
                 checkpoint_keep: int = 3,
                 round_plan_sha256: str | None = None):
        self.coordinator = coordinator
        self.token = token
        self.run_dir = run_dir
        self.adapter_config_dir = adapter_config_dir
        self.manifest = manifest
        self.auto_training_complete_on_round_close = auto_training_complete_on_round_close
        self.hf_adapter_repo = hf_adapter_repo
        self.hf_token = hf_token
        self.hf_adapter_private = hf_adapter_private
        self.hf_push_every_steps = int(hf_push_every_steps)
        if hf_push_optimizer not in OPTIMIZER_POLICIES:
            raise ValueError(f"hf_push_optimizer must be one of {', '.join(OPTIMIZER_POLICIES)}")
        self.hf_push_optimizer = hf_push_optimizer
        self.round_plan = round_plan or [{
            "units": list(coordinator.units.values()),
            "target_loss_tokens": coordinator.target_loss_tokens,
            "min_units": coordinator.min_units,
        }]
        self.round_index = start_round_index
        self.checkpoint_keep = int(checkpoint_keep)
        self.round_plan_sha256 = round_plan_sha256 or round_plan_digest(self.round_plan)
        self.hub_push_versions: dict[str, dict] = {}
        self.hub_push_status = {
            "enabled": hf_adapter_repo is not None,
            "status": "pending" if hf_adapter_repo is not None else "disabled",
        }
        self.persist_lock = threading.RLock()
        self.hub_push_queue: queue.Queue = queue.Queue()
        self.hub_push_thread: threading.Thread | None = None
        self.event_cursor = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "gradients").mkdir(exist_ok=True)
        shutil.copy2(adapter_config_dir / "adapter_config.json", self.run_dir / "adapter_config.json")
        (self.run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if self.hf_adapter_repo is not None:
            (self.run_dir / "hub_push_queue").mkdir(exist_ok=True)
            self.hub_push_thread = threading.Thread(
                target=self._hub_push_loop, name="rwkv-hub-push-queue", daemon=True
            )
            self.hub_push_thread.start()

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

    def save_resume_checkpoint(self) -> None:
        """Commit the current adapter version for --resume (a no-op if already saved)."""
        theta, optimizer_tensors, version, theta_hash = self.coordinator.export_state()
        write_checkpoint(
            self.run_dir / "ckpt",
            state={
                "run_id": self.coordinator.run_id,
                "theta_version": version,
                "theta_hash": theta_hash,
                "planned_rounds": len(self.round_plan),
                "round_plan_sha256": self.round_plan_sha256,
                "data_manifest_sha256": self.manifest["manifest_sha256"],
                "base": self.coordinator.base,
                "tokenizer": self.coordinator.tokenizer,
                "hyperparameters": self.coordinator.hyperparameters(),
            },
            theta=theta,
            optimizer_tensors=optimizer_tensors,
            keep=self.checkpoint_keep,
        )

    def persist_checkpoint(self) -> None:
        with self.persist_lock:
            self.save_resume_checkpoint()
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
                "round_index": self.round_index,
                "planned_rounds": len(self.round_plan),
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
        state["hub_push"] = {
            **self.hub_push_status,
            "versions": self.hub_push_versions,
            "every_steps": self.hf_push_every_steps,
            "optimizer_policy": self.hf_push_optimizer,
        }
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temp.replace(state_path)

    def start_hub_push(self, *, final: bool = False, force: bool = False) -> dict:
        """Queue a versioned adapter snapshot when due, without blocking workers."""
        if self.hf_adapter_repo is None:
            return dict(self.hub_push_status)
        with self.persist_lock:
            step = self.coordinator.theta_version
            if step <= 0:
                return dict(self.hub_push_status)
            if not (force or final or step % self.hf_push_every_steps == 0):
                return dict(self.hub_push_status)
            key = f"{step:06d}"
            previous = self.hub_push_versions.get(key, {}).get("status")
            if previous in ("queued", "uploading", "uploaded"):
                self.hub_push_status = self.hub_push_versions[key]
                return dict(self.hub_push_status)
            path_in_repo = (
                f"checkpoints/{self.coordinator.run_id}/"
                f"theta-{step:06d}"
            )
            snapshot_dir = self.run_dir / "hub_push_queue" / f"theta-{step:06d}"
            try:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.run_dir / "adapter_config.json", snapshot_dir / "adapter_config.json")
                shutil.copy2(self.run_dir / "adapter_model.safetensors", snapshot_dir / "adapter_model.safetensors")
                metadata = {
                    "run_id": self.coordinator.run_id,
                    "round_id": self.coordinator.round_id,
                    "theta_version": step,
                    "theta_hash": self.coordinator.theta_hash,
                    "base": self.coordinator.base,
                    "tokenizer": self.coordinator.tokenizer,
                    "data_manifest_sha256": self.manifest["manifest_sha256"],
                    "planned_rounds": len(self.round_plan),
                    "round_plan_sha256": self.round_plan_sha256,
                    "hyperparameters": self.coordinator.hyperparameters(),
                    "close_metrics": self.coordinator.close_metrics,
                }
                if self.hf_push_optimizer != "never":
                    # Read the moments together with the version they belong to; skip
                    # them rather than pair them with a different adapter.
                    _, moments, moments_version, moments_hash = self.coordinator.export_state()
                    if (moments and moments_version == step
                            and moments_hash == self.coordinator.theta_hash):
                        save_file(moments, str(snapshot_dir / OPTIMIZER_FILE))
                (snapshot_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
                )
            except Exception as exc:
                error_message = str(exc)
                if self.hf_token:
                    error_message = error_message.replace(self.hf_token, "[REDACTED]")
                self.hub_push_status = {
                    "enabled": True,
                    "status": "failed",
                    "repo_id": self.hf_adapter_repo,
                    "theta_version": step,
                    "theta_hash": self.coordinator.theta_hash,
                    "error": f"{type(exc).__name__}: {error_message}",
                }
                self.hub_push_versions[key] = dict(self.hub_push_status)
                self._write_hub_status_locked()
                return dict(self.hub_push_status)
            self.hub_push_status = {
                "enabled": True,
                "status": "queued",
                "repo_id": self.hf_adapter_repo,
                "path_in_repo": path_in_repo,
                "theta_version": step,
                "theta_hash": self.coordinator.theta_hash,
            }
            self.hub_push_versions[key] = dict(self.hub_push_status)
            self._write_hub_status_locked()
            self.hub_push_queue.put((key, snapshot_dir, metadata, path_in_repo))
            return dict(self.hub_push_status)

    def retry_hub_pushes(self) -> dict:
        if self.hf_adapter_repo is None:
            return dict(self.hub_push_status)
        with self.persist_lock:
            queue_root = self.run_dir / "hub_push_queue"
            for key, previous in sorted(self.hub_push_versions.items()):
                if previous.get("status") != "failed":
                    continue
                snapshot_dir = queue_root / f"theta-{int(key):06d}"
                metadata_path = snapshot_dir / "metadata.json"
                if not metadata_path.is_file():
                    continue
                metadata = json.loads(metadata_path.read_text())
                path_in_repo = f"checkpoints/{self.coordinator.run_id}/theta-{int(key):06d}"
                queued = _with_status(previous, "queued")
                self.hub_push_versions[key] = queued
                self.hub_push_status = queued
                self.hub_push_queue.put((key, snapshot_dir, metadata, path_in_repo))
            self._write_hub_status_locked()
        return self.start_hub_push(force=True)

    def _hub_push_loop(self) -> None:
        while True:
            key, snapshot_dir, metadata, path_in_repo = self.hub_push_queue.get()
            try:
                with self.persist_lock:
                    status = _with_status(self.hub_push_versions[key], "uploading")
                    self.hub_push_versions[key] = status
                    self.hub_push_status = status
                    self._write_hub_status_locked()
                error = None
                commit_url = None
                report: dict = {}
                optimizer_file = snapshot_dir / OPTIMIZER_FILE
                for attempt in range(3):
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
                            optimizer_weights_path=optimizer_file if optimizer_file.is_file() else None,
                            optimizer_policy=self.hf_push_optimizer,
                            report=report,
                        )
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                if error is None:
                    result = {**status, "status": "uploaded", "commit_url": commit_url,
                              "optimizer_uploaded": bool(report.get("optimizer_uploaded", False))}
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                else:
                    message = str(error)
                    if self.hf_token:
                        message = message.replace(self.hf_token, "[REDACTED]")
                    result = {**status, "status": "failed",
                              "error": f"{type(error).__name__}: {message}"}
                with self.persist_lock:
                    self.hub_push_versions[key] = result
                    if int(key) >= int(self.hub_push_status.get("theta_version", 0)):
                        self.hub_push_status = result
                    self._write_hub_status_locked()
                print(json.dumps({"hub_push": result}, ensure_ascii=False), flush=True)
            except Exception as exc:
                message = str(exc)
                if self.hf_token:
                    message = message.replace(self.hf_token, "[REDACTED]")
                result = {"enabled": True, "status": "failed", "theta_version": int(key),
                          "error": f"{type(exc).__name__}: {message}"}
                with self.persist_lock:
                    self.hub_push_versions[key] = result
                    self.hub_push_status = result
                    try:
                        self._write_hub_status_locked()
                    except Exception:
                        pass
                print(json.dumps({"hub_push": result}, ensure_ascii=False), flush=True)
            finally:
                self.hub_push_queue.task_done()

    def open_next_round(self) -> bool:
        next_index = self.round_index + 1
        if next_index >= len(self.round_plan):
            return False
        spec = self.round_plan[next_index]
        self.coordinator.start_next_round(
            spec["units"],
            target_loss_tokens=spec["target_loss_tokens"],
            min_units=spec["min_units"],
        )
        self.round_index = next_index
        self.persist_checkpoint()
        return True


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
                body, theta_hash, theta_version = app.coordinator.snapshot()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.safetensors")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Theta-Hash", theta_hash)
                self.send_header("X-Theta-Version", str(theta_version))
                self.end_headers()
                self.wfile.write(body)
                return
            if route.path == "/join":
                worker_id = parse_qs(route.query).get("worker_id", [""])[0]
                try:
                    result = app.coordinator.join(worker_id)
                    app.persist_events()
                    self._json(200, result)
                except ProtocolError as exc:
                    app.persist_events()
                    self._json(exc.status, {"error": exc.code, "message": str(exc)})
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
                status = app.retry_hub_pushes()
                self._json(202, {"hub_push": status})
                return
            if route == "/training-complete":
                if app.round_index != len(app.round_plan) - 1:
                    self._json(409, {"error": "training_plan_remaining",
                                     "round_id": app.coordinator.round_id,
                                     "planned_rounds": len(app.round_plan)})
                    return
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
                    app.persist_checkpoint()
                    final_step = app.round_index == len(app.round_plan) - 1
                    app.start_hub_push(final=final_step)
                    if final_step:
                        if app.auto_training_complete_on_round_close:
                            app.coordinator.signal_training_complete("training_plan_satisfied")
                            app.persist_checkpoint()
                    else:
                        app.open_next_round()
                        result["next_round"] = app.coordinator.descriptor()
                else:
                    app.persist_events()
                result["training_complete"] = app.coordinator.training_complete
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
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION,
                        help="immutable Hugging Face commit SHA")
    parser.add_argument("--model-dir", type=Path, default=MODEL_PATH)
    parser.add_argument("--fetch-model-from-hf", action="store_true",
                        help="download the pinned model snapshot before starting the coordinator")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--push-adapter-to-hf", action="store_true",
                        help="push periodic and final adapter snapshots to a Hugging Face model repository")
    parser.add_argument("--hf-adapter-repo", default=None,
                        help="destination Hugging Face model repository, e.g. username/rwkv-adapter")
    parser.add_argument("--hf-adapter-public", action="store_true",
                        help="create the destination repository as public (private by default)")
    parser.add_argument("--hf-push-optimizer", choices=OPTIMIZER_POLICIES, default="auto",
                        help="whether pushed checkpoints include the AdamW state needed for an exact "
                             "resume. auto: only when the repository is private; always: also to a "
                             "public repository (the moments are derived from training gradients); "
                             "never: adapter only")
    parser.add_argument("--resume-from-hf", action="store_true",
                        help="start from a checkpoint pushed to --hf-adapter-repo (needs --hf-run-id). "
                             "Use on a fresh --run-dir; use --resume to continue from local files")
    parser.add_argument("--hf-run-id", default=None,
                        help="run id of the pushed checkpoints to resume from (the folder under "
                             "checkpoints/ in the Hub repository)")
    parser.add_argument("--hf-version", type=int, default=None,
                        help="exact theta version to resume from; default is the newest that verifies")
    parser.add_argument("--initial-adapter-dir", type=Path,
                        default=Path("artifacts/g1j_1.5b_capybara_sft_protocol"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/g1j_local_protocol/round-0001"))
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17821)
    parser.add_argument("--token", default=os.environ.get("RWKV_PROTOCOL_TOKEN"))
    parser.add_argument("--lease-seconds", type=float, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1,
                        help="repeat the pinned unit manifest this many times")
    parser.add_argument("--units-per-step", type=int, default=0,
                        help="units in each optimizer update; 0 puts one full epoch in each step")
    parser.add_argument("--min-units-per-step", type=int, default=2)
    parser.add_argument("--hf-push-every-steps", type=int, default=10,
                        help="push every N optimizer updates and always push the final adapter")
    parser.add_argument("--manual-training-complete", action="store_true",
                        help="wait for POST /training-complete after the final round")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --run-dir from its newest checkpoint that "
                             "verifies. Work in flight in the interrupted round is discarded and "
                             "the round restarts; data, plan, model and optimizer settings must "
                             "match the original launch")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="resume checkpoints to keep under <run-dir>/ckpt (newest first)")
    args = parser.parse_args()
    if not args.token:
        parser.error("set --token or RWKV_PROTOCOL_TOKEN")
    if args.lease_seconds <= 0 or args.learning_rate <= 0:
        parser.error("lease seconds and learning rate must be positive")
    if args.epochs <= 0 or args.units_per_step < 0 or args.min_units_per_step <= 0:
        parser.error("epochs/minimum units must be positive and units-per-step cannot be negative")
    if args.hf_push_every_steps <= 0:
        parser.error("--hf-push-every-steps must be positive")
    if args.keep_checkpoints < 1:
        parser.error("--keep-checkpoints must be at least 1")
    if args.resume and args.resume_from_hf:
        parser.error("--resume and --resume-from-hf are alternatives; pick one source")
    if args.resume_from_hf and not (args.hf_adapter_repo and args.hf_run_id):
        parser.error("--resume-from-hf requires --hf-adapter-repo and --hf-run-id")
    if (args.hf_run_id or args.hf_version is not None) and not args.resume_from_hf:
        parser.error("--hf-run-id and --hf-version only apply with --resume-from-hf")
    if args.hf_version is not None and args.hf_version < 0:
        parser.error("--hf-version cannot be negative")
    if args.push_adapter_to_hf and not args.hf_adapter_repo:
        parser.error("--push-adapter-to-hf requires --hf-adapter-repo")
    if args.push_adapter_to_hf and not args.hf_token:
        parser.error("--push-adapter-to-hf requires --hf-token or HF_TOKEN (coordinator only)")
    if args.hf_adapter_repo and not (args.push_adapter_to_hf or args.resume_from_hf):
        parser.error("--hf-adapter-repo requires --push-adapter-to-hf or --resume-from-hf")
    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        parser.error("this coordinator prototype is pinned to the G1j 1.5B repo and revision")
    if args.push_adapter_to_hf and args.hf_push_optimizer == "always" and args.hf_adapter_public:
        print("WARNING: --hf-push-optimizer always with --hf-adapter-public uploads AdamW moments "
              "(derived from training gradients) to a public repository.", flush=True)
    elif args.push_adapter_to_hf and args.hf_push_optimizer == "auto" and args.hf_adapter_public:
        print("NOTE: optimizer state is withheld from public repositories under --hf-push-optimizer "
              "auto; a later --resume-from-hf will restart the Adam moments cold.", flush=True)
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
    round_plan = build_round_plan(
        manifest["units"], epochs=args.epochs, units_per_step=args.units_per_step,
        min_units_per_step=args.min_units_per_step,
    )
    plan_sha256 = round_plan_digest(round_plan)
    checkpoint_root = args.run_dir / "ckpt"
    loaded = None
    if args.resume:
        try:
            loaded = load_latest_checkpoint(checkpoint_root)
        except ResumeError as exc:
            parser.error(f"cannot resume: {exc}")
        for problem in loaded.skipped:
            print(f"WARNING: skipped an unusable newer checkpoint: {problem}", flush=True)
    elif list_checkpoint_versions(checkpoint_root):
        parser.error(
            f"{checkpoint_root} already holds checkpoints; pass --resume to continue "
            "that run or choose a new --run-dir"
        )
    elif args.resume_from_hf:
        try:
            loaded = load_resume_from_hub(
                repo_id=args.hf_adapter_repo, run_id=args.hf_run_id, version=args.hf_version,
                token=args.hf_token or None, work_dir=args.run_dir / "hub_resume",
            )
        except ResumeError as exc:
            parser.error(f"cannot resume from the Hub: {exc}")
        for problem in loaded.skipped:
            print(f"WARNING: skipped an unusable Hub checkpoint: {problem}", flush=True)
        if not loaded.optimizer_tensors:
            print("WARNING: this Hub checkpoint carries no optimizer state; the adapter is restored "
                  "but the Adam moments start cold, so later steps will differ from an "
                  "uninterrupted run.", flush=True)
    worker_hash = worker_bundle_sha256(Path(__file__).parent)
    base = {
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "release_manifest_sha256": manifest["model"]["release_manifest_sha256"],
        "serialized_parameter_count": base_manifest["conversion"]["serialized_parameter_count"],
        "dtype": "bfloat16",
    }
    try:
        coordinator, start_index = build_coordinator(
            round_plan=round_plan, plan_sha256=plan_sha256, manifest=manifest, base=base,
            worker_code_sha256=worker_hash, lease_seconds=args.lease_seconds,
            learning_rate=args.learning_rate, new_run_id=f"g1j1.5b-{int(time.time())}",
            initial_theta=None if loaded else load_initial_theta(args.initial_adapter_dir),
            loaded=loaded,
        )
    except ResumeError as exc:
        parser.error(f"cannot resume: {exc}")
    target_tokens = round_plan[start_index]["target_loss_tokens"]
    app = Application(
        coordinator, token=args.token, run_dir=args.run_dir,
        adapter_config_dir=args.initial_adapter_dir, manifest=manifest,
        auto_training_complete_on_round_close=not args.manual_training_complete,
        hf_adapter_repo=args.hf_adapter_repo if args.push_adapter_to_hf else None,
        hf_token=args.hf_token if args.push_adapter_to_hf else None,
        hf_adapter_private=not args.hf_adapter_public,
        round_plan=round_plan,
        hf_push_every_steps=args.hf_push_every_steps,
        hf_push_optimizer=args.hf_push_optimizer,
        start_round_index=start_index,
        checkpoint_keep=args.keep_checkpoints,
        round_plan_sha256=plan_sha256,
    )
    app.persist_checkpoint()  # commits version 0 of a fresh run; logs a resume event
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
        "resumed": loaded is not None,
        "resumed_from": (None if loaded is None else "hub" if args.resume_from_hf else "local"),
        "optimizer_restored": bool(loaded and loaded.optimizer_tensors),
        "target_loss_tokens": target_tokens,
        "planned_optimizer_steps": len(round_plan),
        "epochs": args.epochs,
        "units_per_step": args.units_per_step or len(manifest["units"]),
        "hf_push_every_steps": args.hf_push_every_steps,
        "min_units": coordinator.min_units,
        "worker_code_sha256": worker_hash,
        "data_manifest_sha256": manifest["manifest_sha256"],
        "training_completion": (
            "manual POST /training-complete" if args.manual_training_complete
            else f"signaled automatically when all {len(round_plan)} planned optimizer steps close"
        ),
        "adapter_hub_push": (
            {"enabled": True, "repo_id": args.hf_adapter_repo,
             "private_if_created": not args.hf_adapter_public,
             "every_optimizer_steps": args.hf_push_every_steps,
             "optimizer_state": args.hf_push_optimizer,
             "final_snapshot": True}
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
