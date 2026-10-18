#!/usr/bin/env python3
"""Development coordinator for multi-step LoRA-SFT over authenticated HTTP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from safetensors.torch import load_file, save_file

from rwkv_dist_checkpoint import (
    LoadedCheckpoint,
    ResumeError,
    check_resume_compatibility,
    checkpoint_name,
    list_checkpoint_versions,
    load_hub_checkpoint,
    load_latest_checkpoint,
    write_checkpoint,
)
from rwkv_dist_protocol import (
    ANONYMOUS,
    PayloadFile,
    ProtocolError,
    RoundCoordinator,
    adamw_hyperparameters,
    canonical_json,
    default_model_dir,
    model_slug,
    parameter_name_to_peft_saved_name,
    peft_saved_name_to_parameter_name,
    rank_contributions,
    restore_contributions,
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


# The model a run trains unless --model-id/--model-revision say otherwise. Any RWKV7
# size in the same Hugging Face format works; workers must be started for the same one.
DEFAULT_MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
DEFAULT_MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
LORA_RANK = 8
LORA_ALPHA = 16
ATTENTION_MODULES = ("receptance", "key", "value", "output")
UPLOAD_CHUNK_BYTES = 1024 * 1024


def default_initial_adapter_dir(model_id: str) -> Path:
    """Where a fresh run looks for its starting adapter; make_initial_adapter.py writes here.

    One folder per model, because an adapter only fits the model it was made for.
    """
    return Path("artifacts") / f"{model_slug(model_id)}_lora_init"


def worker_bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in WORKER_BUNDLE_FILES:
        path = root / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_manifest(path: Path, *, model_id: str, model_revision: str) -> dict:
    manifest = json.loads(path.read_text())
    supplied_hash = manifest.get("manifest_sha256")
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = sha256_bytes(canonical_json(content))
    if supplied_hash != actual_hash:
        raise ValueError(f"data manifest hash mismatch: {supplied_hash} != {actual_hash}")
    if manifest["model"]["repo"] != model_id or manifest["model"]["revision"] != model_revision:
        raise ValueError(
            f"tokenized shard was made for {manifest['model']['repo']}@{manifest['model']['revision']}, "
            f"not {model_id}@{model_revision}"
        )
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


_LORA_NAME = re.compile(
    r"^(?P<module>.*\.layers\.(?P<layer>\d+)\.att\.(?P<kind>receptance|key|value|output))"
    r"\.lora_(?P<factor>[AB])\.default\.weight$"
)


def expected_lora_shapes(num_layers: int, hidden_size: int) -> dict[str, tuple[int, int]]:
    """Tensor name -> shape of the rank-8 attention LoRA for an RWKV7 of this size."""
    shapes = {}
    for layer in range(num_layers):
        for kind in ATTENTION_MODULES:
            prefix = f"base_model.model.model.layers.{layer}.att.{kind}"
            shapes[f"{prefix}.lora_A.default.weight"] = (LORA_RANK, hidden_size)
            shapes[f"{prefix}.lora_B.default.weight"] = (hidden_size, LORA_RANK)
    return shapes


def read_model_shape(model_dir: Path) -> tuple[int, int] | None:
    """(layers, hidden size) from the snapshot's config.json, or None if it has none."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    try:
        return int(config["num_hidden_layers"]), int(config["hidden_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_path} lacks num_hidden_layers / hidden_size") from exc


def check_lora_structure(theta: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Check that `theta` is a complete rank-8 attention LoRA; return (layers, hidden size)."""
    if not theta:
        raise ValueError("initial adapter holds no LoRA tensors")
    modules: dict[str, dict[str, torch.Tensor]] = {}
    layers: dict[int, set[str]] = {}
    for name, tensor in theta.items():
        match = _LORA_NAME.match(name)
        if match is None:
            raise ValueError(f"unexpected tensor in initial adapter: {name}")
        modules.setdefault(match["module"], {})[match["factor"]] = tensor
        layers.setdefault(int(match["layer"]), set()).add(match["kind"])
    widths = set()
    for module, factors in modules.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(f"{module} has only lora_{next(iter(factors))}; need both factors")
        a, b = factors["A"], factors["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != LORA_RANK or b.shape[1] != LORA_RANK \
                or a.shape[1] != b.shape[0]:
            raise ValueError(f"{module} is not a rank-{LORA_RANK} LoRA pair: "
                             f"A {tuple(a.shape)}, B {tuple(b.shape)}")
        widths.add(int(a.shape[1]))
    if len(widths) != 1:
        raise ValueError(f"attention widths differ across modules: {sorted(widths)}")
    if sorted(layers) != list(range(len(layers))) or any(
            kinds != set(ATTENTION_MODULES) for kinds in layers.values()):
        raise ValueError("initial adapter must cover receptance/key/value/output of layers 0..N-1")
    return len(layers), widths.pop()


def load_initial_theta(adapter_dir: Path, *,
                       model_shape: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
    """Read the starting adapter; with `model_shape` (layers, hidden) also match it to that model."""
    missing = [name for name in ("adapter_config.json", "adapter_model.safetensors")
               if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"initial adapter {adapter_dir} is missing {', '.join(missing)}")
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if int(adapter_config.get("r", -1)) != LORA_RANK or int(adapter_config.get("lora_alpha", -1)) != LORA_ALPHA:
        raise ValueError(f"initial adapter must be rank {LORA_RANK} / alpha {LORA_ALPHA}")
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    theta = {}
    for name, tensor in state.items():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
            name = peft_saved_name_to_parameter_name(name)
        theta[name] = tensor.float().contiguous()
    layers, width = check_lora_structure(theta)
    if model_shape is not None and (layers, width) != tuple(model_shape):
        raise ValueError(
            f"initial adapter was made for a model with {layers} layers of width {width}, but this "
            f"run's base model has {model_shape[0]} layers of width {model_shape[1]}; create a "
            f"matching one with make_initial_adapter.py"
        )
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
        contributions = None
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
        try:
            contributions = restore_contributions(loaded.state.get("contributions"))
        except ValueError as exc:
            raise ResumeError(f"checkpoint has a malformed contribution ledger: {exc}") from exc
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
        contributions=contributions,
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


_QUEUE_ENTRY = re.compile(r"^theta-(\d{6,})$")
_GRADIENT_ROUND = re.compile(r"^round-(\d{6,})(?:\.discarded-\d+)?$")


def _redact(message: str, secret: str | None) -> str:
    return message.replace(secret, "[REDACTED]") if secret else message


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
                 round_plan_sha256: str | None = None,
                 keep_gradient_rounds: int = 1):
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
        # Closed rounds whose per-unit gradient files are kept; -1 keeps all of them.
        # Each file is as large as the adapter, so the default keeps only the latest.
        self.keep_gradient_rounds = int(keep_gradient_rounds)
        self.round_plan_sha256 = round_plan_sha256 or round_plan_digest(self.round_plan)
        self.hub_push_versions: dict[str, dict] = {}
        self.hub_push_status = {
            "enabled": hf_adapter_repo is not None,
            "status": "pending" if hf_adapter_repo is not None else "disabled",
        }
        self.persist_lock = threading.RLock()
        self.hub_push_queue: queue.Queue = queue.Queue()
        self.hub_push_thread: threading.Thread | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "gradients").mkdir(exist_ok=True)
        # Uploads are streamed here before they are validated; anything left from
        # a previous process belongs to a request that never finished.
        self.incoming_dir = self.run_dir / "incoming"
        shutil.rmtree(self.incoming_dir, ignore_errors=True)
        self.incoming_dir.mkdir()
        shutil.copy2(adapter_config_dir / "adapter_config.json", self.run_dir / "adapter_config.json")
        (self.run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if self.hf_adapter_repo is not None:
            (self.run_dir / "hub_push_queue").mkdir(exist_ok=True)
            self.hub_push_thread = threading.Thread(
                target=self._hub_push_loop, name="rwkv-hub-push-queue", daemon=True
            )
            self.hub_push_thread.start()

    def persist_events(self) -> None:
        """Append new events to round_log.jsonl and drop them from memory.

        Only the coordinator appends (at the end), so removing exactly the
        written prefix is safe, and a long run does not accumulate every
        heartbeat in RAM.
        """
        with self.persist_lock:
            events = list(self.coordinator.events)
            if events:
                with (self.run_dir / "round_log.jsonl").open("a") as f:
                    for event in events:
                        f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
                del self.coordinator.events[:len(events)]
                for event in events:
                    line = self.console_line(event)
                    if line:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}", flush=True)

    def console_line(self, event: dict) -> str | None:
        """The one-line console report for the events worth reading while a run is going."""
        try:
            kind = event.get("event")
            if kind == "round_closed":
                metrics = event["metrics"]
                parts = [
                    f"step {event['theta_version']}/{len(self.round_plan)} closed",
                    f"mean_loss {metrics['mean_loss']:.4f}",
                    f"{metrics['unit_count']} units, {metrics['loss_tokens']} loss tokens",
                    f"grad_norm {metrics['preclip_gradient_norm']:.4g}",
                    f"{metrics['round_seconds']:.1f}s",
                ]
                credit = metrics.get("contributors", {})
                if any(name != ANONYMOUS for name in credit):
                    ranked = sorted(credit.items(),
                                    key=lambda item: (-item[1]["units"], item[0] == ANONYMOUS, item[0]))
                    parts.append("contributors: " + ", ".join(
                        f"{name} {entry['units']}" for name, entry in ranked))
                return " | ".join(parts)
            if kind == "worker_joined" and event.get("nickname"):
                waiting = "" if event.get("status") == "active" else " (starts with the next round)"
                return f"contributor \"{event['nickname']}\" joined with worker {event['worker_id']}{waiting}"
        except (KeyError, TypeError, ValueError):
            return None  # a report must never get in the way of saving the log
        return None

    def stream_upload(self, source, size: int) -> PayloadFile:
        """Copy `size` bytes from `source` to a file in incoming/, hashing as they pass.

        Memory use is one chunk however large the upload is or however many
        arrive at once. Raises ProtocolError (and leaves nothing behind) if the
        body ends early.
        """
        path = self.incoming_dir / f"{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        received = 0
        try:
            with path.open("wb") as handle:
                while received < size:
                    chunk = source.read(min(UPLOAD_CHUNK_BYTES, size - received))
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if received != size:
                raise ProtocolError("truncated_payload", "request body shorter than Content-Length", 400)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return PayloadFile(path=path, sha256=digest.hexdigest(), size=size)

    def persist_result_payload(self, unit_id: str, payload: "bytes | PayloadFile",
                               round_id: int | None = None) -> None:
        """Store an accepted unit's gradient under ``gradients/round-NNNNNN/``.

        `round_id` is the round the result was accepted in. The handler passes it
        explicitly because the next round may already be open by the time a
        slower request gets here. A streamed upload is moved into place, not copied.
        """
        round_id = self.coordinator.round_id if round_id is None else int(round_id)
        directory = self.run_dir / "gradients" / f"round-{round_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = hashlib.sha256(unit_id.encode()).hexdigest()
        path = directory / f"{safe_name}.safetensors"
        if isinstance(payload, PayloadFile):
            payload.path.replace(path)
            return
        temp = path.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(path)

    def set_aside_in_flight_gradients(self) -> Path | None:
        """After a resume, move the open round's earlier gradient files out of the live set.

        They belong to work that was discarded, and a reissued round reuses the
        same file names for the same units; leaving them in place would blur the
        old attempt into the new one.
        """
        with self.persist_lock:
            live = self.run_dir / "gradients" / f"round-{self.coordinator.round_id:06d}"
            if not live.is_dir():
                return None
            aside = live.with_name(f"{live.name}.discarded-{int(time.time())}")
            live.rename(aside)
            return aside

    def prune_gradients(self) -> list[str]:
        """Delete gradient folders of closed rounds beyond `keep_gradient_rounds`.

        The open round is never touched. With the default of -1 nothing is deleted.
        """
        if self.keep_gradient_rounds < 0:
            return []
        root = self.run_dir / "gradients"
        closed_through = self.coordinator.round_id - (0 if self.coordinator.closed else 1)
        cutoff = closed_through - self.keep_gradient_rounds
        removed = []
        if not root.is_dir():
            return removed
        for entry in sorted(root.iterdir()):
            match = _GRADIENT_ROUND.match(entry.name)
            if match and entry.is_dir() and int(match.group(1)) <= cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
        return removed

    def save_resume_checkpoint(self) -> None:
        """Commit the current adapter version for --resume (a no-op if already saved)."""
        with self.coordinator.lock:  # the ledger and the adapter version must be of the same round
            theta, optimizer_tensors, version, theta_hash = self.coordinator.export_state()
            contributions = self.coordinator.export_contributions()
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
                "contributions": contributions,
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
            summary["hub_push"] = self._hub_state_locked()
            (self.run_dir / "coordinator_state.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            self.write_contributors_file()
            self.persist_events()
            self.prune_gradients()

    def write_contributors_file(self) -> None:
        """contributors.json: who has been credited for the closed rounds (the saved ledger)."""
        document = {
            "run_id": self.coordinator.run_id,
            "as_of_theta_version": self.coordinator.theta_version,
            "note": "credit for closed optimizer steps; nicknames are self-chosen and unauthenticated",
            "contributors": rank_contributions(self.coordinator.export_contributions()),
        }
        temp = self.run_dir / "contributors.tmp"
        temp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        temp.replace(self.run_dir / "contributors.json")

    def _hub_state_locked(self) -> dict:
        return {
            **self.hub_push_status,
            "versions": self.hub_push_versions,
            "every_steps": self.hf_push_every_steps,
            "optimizer_policy": self.hf_push_optimizer,
        }

    def _write_hub_status_locked(self) -> None:
        state_path = self.run_dir / "coordinator_state.json"
        try:
            state = json.loads(state_path.read_text())
        except FileNotFoundError:
            state = {}
        state["hub_push"] = self._hub_state_locked()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temp.replace(state_path)

    def _queue_entry_verdict(self, entry: Path, version: int, current: int) -> tuple[str, dict | None]:
        """Decide what to do with a push left in hub_push_queue by an earlier process.

        Returns ("replay", metadata), ("drop", None) for an entry that is unusable
        or belongs to a timeline this run no longer follows, or ("ignore", None).
        """
        try:
            metadata = json.loads((entry / "metadata.json").read_text())
            if int(metadata["theta_version"]) != version:
                return "drop", None
        except (OSError, ValueError, KeyError, TypeError):
            return "drop", None
        if metadata.get("run_id") != self.coordinator.run_id:
            return "ignore", None
        if version > current:
            return "drop", None  # pushed by a timeline that resume has rolled back past
        try:
            raw = load_file(str(entry / "adapter_model.safetensors"), device="cpu")
            theta = {
                peft_saved_name_to_parameter_name(name): tensor.float().contiguous()
                for name, tensor in raw.items() if ".lora_A." in name or ".lora_B." in name
            }
            if not theta or tensor_map_sha256(theta) != metadata.get("theta_hash"):
                return "drop", None
        except Exception:
            return "drop", None
        try:  # when the local checkpoint for that version survives, it must agree
            kept = json.loads((self.run_dir / "ckpt" / checkpoint_name(version) / "state.json").read_text())
            if kept.get("theta_hash") != metadata["theta_hash"]:
                return "drop", None
        except (OSError, ValueError):
            pass
        return "replay", metadata

    def recover_hub_pushes(self) -> dict:
        """Pick up Hub pushes an earlier process left unfinished, and learn what the Hub holds.

        Call after the startup checkpoint has been persisted. Versions the Hub
        already has are recorded as uploaded and are never pushed again; leftover
        queue entries that verify are re-queued; the resumed version itself is
        pushed if it is due and missing. An unreachable Hub does not stop startup:
        everything pending is pushed, which at worst repeats an identical upload.
        """
        report: dict = {"found_on_hub": [], "replayed": [], "dropped": [], "ignored": [],
                        "hub_ahead": [], "pushed_resumed_version": None, "hub_error": None}
        if self.hf_adapter_repo is None:
            return report
        run_id = self.coordinator.run_id
        current = self.coordinator.theta_version
        remote: set[int] | None = None
        try:
            remote = set(list_hub_checkpoint_versions(
                repo_id=self.hf_adapter_repo, run_id=run_id, token=self.hf_token))
        except Exception as exc:
            report["hub_error"] = _redact(f"{type(exc).__name__}: {exc}", self.hf_token)
        with self.persist_lock:
            for version in sorted(remote or ()):
                if version > current:
                    report["hub_ahead"].append(version)
                elif version > 0:
                    key = f"{version:06d}"
                    self.hub_push_versions.setdefault(key, {
                        "enabled": True, "status": "uploaded", "repo_id": self.hf_adapter_repo,
                        "path_in_repo": f"checkpoints/{run_id}/theta-{key}",
                        "theta_version": version, "source": "found_on_hub",
                    })
                    report["found_on_hub"].append(version)
            queue_root = self.run_dir / "hub_push_queue"
            entries = sorted(queue_root.iterdir()) if queue_root.is_dir() else []
            for entry in entries:
                match = _QUEUE_ENTRY.match(entry.name)
                if not match or not entry.is_dir():
                    continue
                version = int(match.group(1))
                key = f"{version:06d}"
                verdict, metadata = self._queue_entry_verdict(entry, version, current)
                if verdict == "ignore":
                    report["ignored"].append(entry.name)
                elif verdict == "drop":
                    shutil.rmtree(entry, ignore_errors=True)
                    report["dropped"].append(entry.name)
                elif remote is not None and version in remote:
                    shutil.rmtree(entry, ignore_errors=True)  # already uploaded
                else:
                    path_in_repo = f"checkpoints/{run_id}/theta-{key}"
                    queued = {
                        "enabled": True, "status": "queued", "repo_id": self.hf_adapter_repo,
                        "path_in_repo": path_in_repo, "theta_version": version,
                        "theta_hash": metadata["theta_hash"], "source": "replayed_after_restart",
                    }
                    self.hub_push_versions[key] = queued
                    self.hub_push_queue.put((key, entry, metadata, path_in_repo))
                    report["replayed"].append(version)
            if self.hub_push_versions:
                latest = max(self.hub_push_versions, key=int)
                self.hub_push_status = self.hub_push_versions[latest]
            self._write_hub_status_locked()
            key = f"{current:06d}"
            if current > 0 and current % self.hf_push_every_steps == 0 and key not in self.hub_push_versions:
                report["pushed_resumed_version"] = current
                self.start_hub_push()
        return report

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
                    # The round that produced this version; the open round's id is one more
                    # whenever the push happens after the next round has opened.
                    "round_id": step,
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
            if route.path == "/contributors":
                self._json(200, {
                    "run_id": app.coordinator.run_id,
                    "theta_version": app.coordinator.theta_version,
                    "contributors": app.coordinator.contributors(),
                })
                return
            if route.path == "/join":
                # keep_blank_values: "?nickname=" is an empty nickname (refused), not a missing one
                query = parse_qs(route.query, keep_blank_values=True)
                worker_id = query.get("worker_id", [""])[0]
                try:
                    result = app.coordinator.join(worker_id, query.get("nickname", [None])[0])
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
            upload = None
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 1 or size > 128 * 1024 * 1024:
                    raise ProtocolError("invalid_payload_size", "payload size must be 1..128 MiB", 413)
                encoded_metadata = self.headers.get("X-Result-Metadata", "")
                metadata = json.loads(base64.urlsafe_b64decode(encoded_metadata.encode("ascii")))
                upload = app.stream_upload(self.rfile, size)
                result = app.coordinator.submit_result(metadata, upload)
                app.persist_result_payload(str(metadata["unit_id"]), upload,
                                           round_id=int(metadata["round_id"]))
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
            finally:
                if upload is not None:
                    upload.path.unlink(missing_ok=True)  # rejected uploads; accepted ones were moved

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="base model repository; any RWKV7 size in the same format. Workers "
                             "must be started with the same --model-id/--model-revision "
                             "(default: %(default)s)")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="immutable Hugging Face commit SHA")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="local snapshot of the base model (default: /workspace/<model name>)")
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
    parser.add_argument("--initial-adapter-dir", type=Path, default=None,
                        help="rank-8 PEFT adapter a fresh run starts from; create one with "
                             "make_initial_adapter.py (default: artifacts/<model name>_lora_init)")
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
    parser.add_argument("--keep-gradient-rounds", type=int, default=1,
                        help="closed rounds whose per-unit gradient files under gradients/ are kept "
                             "(the open round's always are); older ones are deleted. Each file is "
                             "4 bytes x LoRA parameters (12 MiB for 1.5B, 32 MiB for 7.2B). "
                             "-1 keeps everything (default: %(default)s)")
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
    if args.keep_gradient_rounds < -1:
        parser.error("--keep-gradient-rounds must be -1 (keep all) or a count of rounds, 0 or more")
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
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        parser.error("--model-revision must be a full 40-character commit hash, not a branch or tag")
    try:
        run_name = model_slug(args.model_id)
    except ValueError as exc:
        parser.error(str(exc))
    if args.model_dir is None:
        args.model_dir = default_model_dir(args.model_id)
    if args.initial_adapter_dir is None:
        args.initial_adapter_dir = default_initial_adapter_dir(args.model_id)
    if args.push_adapter_to_hf and args.hf_push_optimizer == "always" and args.hf_adapter_public:
        print("WARNING: --hf-push-optimizer always with --hf-adapter-public uploads AdamW moments "
              "(derived from training gradients) to a public repository.", flush=True)
    elif args.push_adapter_to_hf and args.hf_push_optimizer == "auto" and args.hf_adapter_public:
        print("NOTE: optimizer state is withheld from public repositories under --hf-push-optimizer "
              "auto; a later --resume-from-hf will restart the Adam moments cold.", flush=True)
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: this development server is plain HTTP. Expose it only behind TLS/authenticated networking.", flush=True)
    try:
        manifest = read_manifest(args.units, model_id=args.model_id,
                                 model_revision=args.model_revision)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"cannot use --units {args.units}: {type(exc).__name__}: {exc}")
    model_dir = ensure_hf_checkpoint(
        args.model_id, args.model_revision, args.model_dir,
        fetch_from_hf=args.fetch_model_from_hf,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    release_manifest_path = model_dir / "release-manifest.json"
    base_manifest = json.loads(release_manifest_path.read_text())
    try:
        serialized_parameter_count = base_manifest["conversion"]["serialized_parameter_count"]
    except (KeyError, TypeError):
        parser.error(f"{release_manifest_path} has no conversion.serialized_parameter_count")
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
        print("NOTE: contributor credit is not stored on the Hub; the ledger of this restored run "
              "starts empty (a later local --resume keeps it).", flush=True)
        if not loaded.optimizer_tensors:
            print("WARNING: this Hub checkpoint carries no optimizer state; the adapter is restored "
                  "but the Adam moments start cold, so later steps will differ from an "
                  "uninterrupted run.", flush=True)
    initial_theta = None
    if loaded is None:
        try:
            model_shape = read_model_shape(model_dir)
            if model_shape is None:
                print(f"NOTE: {model_dir}/config.json not found; the initial adapter is checked for "
                      "structure only, and workers verify the exact tensor names and shapes.", flush=True)
            initial_theta = load_initial_theta(args.initial_adapter_dir, model_shape=model_shape)
        except FileNotFoundError as exc:
            parser.error(
                f"cannot start from {args.initial_adapter_dir}: {exc}. Create a fresh rank-{LORA_RANK} "
                f"adapter for this model (on a worker machine) with: python make_initial_adapter.py "
                f"--model-id {args.model_id} --model-revision {args.model_revision} "
                f"--base-model {model_dir} --output-dir {args.initial_adapter_dir}"
            )
        except ValueError as exc:
            parser.error(f"cannot start from {args.initial_adapter_dir}: {exc}")
    worker_hash = worker_bundle_sha256(Path(__file__).parent)
    base = {
        "repo": args.model_id,
        "revision": args.model_revision,
        "release_manifest_sha256": manifest["model"]["release_manifest_sha256"],
        "serialized_parameter_count": serialized_parameter_count,
        "dtype": "bfloat16",
    }
    try:
        coordinator, start_index = build_coordinator(
            round_plan=round_plan, plan_sha256=plan_sha256, manifest=manifest, base=base,
            worker_code_sha256=worker_hash, lease_seconds=args.lease_seconds,
            learning_rate=args.learning_rate, new_run_id=f"{run_name}-{int(time.time())}",
            initial_theta=initial_theta,
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
        keep_gradient_rounds=args.keep_gradient_rounds,
    )
    if loaded is not None:
        aside = app.set_aside_in_flight_gradients()
        if aside is not None:
            print(f"Set aside the interrupted round's gradient files: {aside.name}", flush=True)
    app.persist_checkpoint()  # commits version 0 of a fresh run; logs a resume event
    if loaded is not None and args.push_adapter_to_hf:
        print("Reconciling Hub pushes with the resumed run...", flush=True)
        recovery = app.recover_hub_pushes()
        print(json.dumps({"hub_push_recovery": recovery}, ensure_ascii=False), flush=True)
        if recovery["hub_error"]:
            print("WARNING: could not list the Hub repository; pending pushes will be sent without "
                  "checking what it already holds.", flush=True)
        if recovery["hub_ahead"]:
            print(f"WARNING: the Hub already holds newer versions of this run "
                  f"({recovery['hub_ahead']}) than the one resumed from. Reaching those versions "
                  f"again will overwrite their folders.", flush=True)
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
        "keep_gradient_rounds": args.keep_gradient_rounds,
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
