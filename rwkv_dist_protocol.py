"""Small, explicit protocol primitives for RWKV distributed SFT experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


def model_name(model_id: str) -> str:
    """'RWKV/RWKV7-G1j-1.5B-20260831' -> 'RWKV7-G1j-1.5B-20260831'."""
    return model_id.rstrip("/").rsplit("/", 1)[-1]


def default_model_dir(model_id: str) -> Path:
    """Where a local snapshot of this model is expected unless a path is given."""
    return Path("/workspace") / model_name(model_id)


def model_slug(model_id: str) -> str:
    """A file-name and run-id safe form of the model name, e.g. 'rwkv7-g1j-1.5b-20260831'."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_name(model_id).lower()).strip("-._")
    if not slug:
        raise ValueError(f"cannot derive a name from model id {model_id!r}")
    return slug


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_map_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(canonical_json(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def unit_digest(unit: dict[str, Any]) -> str:
    content = {key: value for key, value in unit.items() if key != "unit_sha256"}
    return sha256_bytes(canonical_json(content))


def peft_saved_name_to_parameter_name(name: str) -> str:
    for factor in ("lora_A", "lora_B"):
        suffix = f".{factor}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{factor}.default.weight"
    raise ValueError(f"Not a supported PEFT LoRA weight name: {name}")


def parameter_name_to_peft_saved_name(name: str) -> str:
    for factor in ("lora_A", "lora_B"):
        suffix = f".{factor}.default.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{factor}.weight"
    raise ValueError(f"Not a supported PEFT LoRA parameter name: {name}")


def encode_tensors(tensors: dict[str, torch.Tensor]) -> bytes:
    return save_safetensors({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()})


def decode_tensors(payload: bytes) -> dict[str, torch.Tensor]:
    return load_safetensors(payload)


@dataclass(frozen=True)
class PayloadFile:
    """A gradient upload that was streamed to disk instead of held in memory.

    `sha256` is the hash of the file's bytes, computed while it was received.
    submit_result() reads tensors from it one at a time, so a coordinator's
    memory does not grow with the size of an upload or with how many arrive at once.
    """

    path: Path
    sha256: str
    size: int


class _BytesTensors:
    """Tensors of an in-memory safetensors payload."""

    def __init__(self, payload: bytes):
        self._tensors = decode_tensors(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._tensors = {}

    def names(self) -> set[str]:
        return set(self._tensors)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._tensors[name].shape)

    def tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _FileTensors:
    """Tensors of a safetensors file, read one at a time through a memory map."""

    def __init__(self, path: Path):
        self._file = safe_open(str(path), framework="pt", device="cpu")

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._file.__exit__(*exc_info)

    def names(self) -> set[str]:
        return set(self._file.keys())

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._file.get_slice(name).get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        return self._file.get_tensor(name)


def open_gradient_payload(payload: "bytes | PayloadFile"):
    return _FileTensors(payload.path) if isinstance(payload, PayloadFile) else _BytesTensors(payload)


def payload_sha256(payload: "bytes | PayloadFile") -> str:
    return payload.sha256 if isinstance(payload, PayloadFile) else sha256_bytes(payload)


OPTIMIZER_STATE_KEYS = ("exp_avg", "exp_avg_sq", "step")


def optimizer_hyperparameters(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """JSON-safe AdamW settings that must match for saved moments to stay valid."""
    group = optimizer.param_groups[0]
    return {
        "lr": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
    }


def adamw_hyperparameters(learning_rate: float, weight_decay: float = 0.0) -> dict[str, Any]:
    """The settings RoundCoordinator's optimizer will have for these arguments."""
    probe = torch.nn.Parameter(torch.zeros(1))
    return optimizer_hyperparameters(
        torch.optim.AdamW([probe], lr=learning_rate, weight_decay=weight_decay)
    )


def optimizer_state_tensors(
    optimizer: torch.optim.Optimizer, params: dict[str, torch.nn.Parameter]
) -> dict[str, torch.Tensor]:
    """Flatten per-parameter AdamW state into named CPU tensors.

    Plain tensors rather than a pickled state_dict, so the result can be kept
    as safetensors and loaded from an untrusted location without running code.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in params.items():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        missing = [key for key in OPTIMIZER_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"unsupported optimizer state for {name}: missing {missing}")
        for key in OPTIMIZER_STATE_KEYS:
            tensors[f"{key}/{name}"] = (
                torch.as_tensor(state[key]).detach().to("cpu", torch.float32).contiguous().clone()
            )
    return tensors


def load_optimizer_state_tensors(
    optimizer: torch.optim.Optimizer,
    params: dict[str, torch.nn.Parameter],
    tensors: dict[str, torch.Tensor],
) -> None:
    """Install optimizer_state_tensors() output into a freshly built optimizer.

    Hyperparameters stay as the optimizer was constructed; callers compare them
    with the saved ones beforehand. An empty map (nothing stepped yet) is a no-op.
    """
    names = list(params)
    known = set(names)
    for tensor_name in tensors:
        key, _, name = tensor_name.partition("/")
        if key not in OPTIMIZER_STATE_KEYS or name not in known:
            raise ValueError(f"unexpected optimizer state tensor {tensor_name!r}")
    state: dict[int, dict[str, torch.Tensor]] = {}
    for index, name in enumerate(names):
        present = [key for key in OPTIMIZER_STATE_KEYS if f"{key}/{name}" in tensors]
        if not present:
            continue
        if len(present) != len(OPTIMIZER_STATE_KEYS):
            raise ValueError(f"incomplete optimizer state for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if tuple(tensors[f"{key}/{name}"].shape) != tuple(params[name].shape):
                raise ValueError(f"optimizer {key} shape mismatch for {name}")
        state[index] = {key: tensors[f"{key}/{name}"].clone() for key in OPTIMIZER_STATE_KEYS}
    if not state:
        return
    if len(state) != len(names):
        raise ValueError("optimizer state covers only some trainable tensors")
    optimizer.load_state_dict({
        "state": state,
        "param_groups": optimizer.state_dict()["param_groups"],
    })


class ProtocolError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class Lease:
    lease_id: str
    unit_id: str
    worker_id: str
    expires_at: float
    round_id: int
    theta_version: int
    theta_hash: str


class RoundCoordinator:
    """In-memory optimizer-step coordinator with lease, dedup, stale and gradient checks."""

    def __init__(
        self,
        *,
        theta: dict[str, torch.Tensor],
        units: list[dict[str, Any]],
        run_id: str,
        base: dict[str, Any],
        tokenizer: dict[str, Any],
        worker_code_sha256: str,
        target_loss_tokens: int,
        min_units: int,
        lease_seconds: float,
        learning_rate: float,
        weight_decay: float = 0.0,
        global_grad_clip: float | None = None,
        theta_version: int = 0,
        optimizer_tensors: dict[str, torch.Tensor] | None = None,
    ):
        """`theta_version` and `optimizer_tensors` resume from a saved adapter
        version: the round opened is then round `theta_version + 1`, and the
        AdamW moments continue from `optimizer_tensors` instead of starting cold.
        """
        if not theta:
            raise ValueError("theta must contain trainable tensors")
        if theta_version < 0:
            raise ValueError("theta_version cannot be negative")
        if not units:
            raise ValueError("at least one work unit is required")
        self.lock = threading.RLock()
        self.run_id = run_id
        self.round_id = theta_version + 1
        self.theta_version = theta_version
        self.theta = {k: v.detach().to("cpu", torch.float32).contiguous().clone() for k, v in theta.items()}
        self.expected = {k: tuple(v.shape) for k, v in self.theta.items()}
        self.theta_hash = tensor_map_sha256(self.theta)
        self.base = base
        self.tokenizer = tokenizer
        self.worker_code_sha256 = worker_code_sha256
        self.target_loss_tokens = int(target_loss_tokens)
        self.min_units = int(min_units)
        self.lease_seconds = float(lease_seconds)
        self.global_grad_clip = global_grad_clip
        self.params = {k: torch.nn.Parameter(v.clone()) for k, v in self.theta.items()}
        self.optimizer = torch.optim.AdamW(
            list(self.params.values()), lr=learning_rate, weight_decay=weight_decay
        )
        if optimizer_tensors:
            load_optimizer_state_tensors(self.optimizer, self.params, optimizer_tensors)
        self.units: dict[str, dict[str, Any]] = {}
        for supplied in units:
            unit = dict(supplied)
            unit_id = str(unit["unit_id"])
            if unit_id in self.units:
                raise ValueError(f"duplicate unit_id {unit_id}")
            if int(unit["loss_tokens"]) <= 0 or int(unit["total_tokens"]) <= 0:
                raise ValueError(f"invalid token counts for {unit_id}")
            expected_hash = unit_digest(unit)
            if unit.get("unit_sha256") != expected_hash:
                raise ValueError(f"unit hash mismatch for {unit_id}")
            self.units[unit_id] = unit
        self.pending = deque(self.units)
        self.unit_status = {unit_id: "pending" for unit_id in self.units}
        self.leases: dict[str, Lease] = {}
        self.accepted: dict[str, dict[str, Any]] = {}
        # Running sum of the accepted units' gradients, added in acceptance order.
        # Kept instead of the individual gradients, so memory does not grow with
        # the number of units in a step.
        self.grad_sums: dict[str, torch.Tensor] | None = None
        self._snapshot_cache: tuple[int, bytes] | None = None
        self.loss_tokens = 0
        self.loss_sum = 0.0
        self.closed = False
        self.training_complete = False
        self.close_metrics: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.active_workers: set[str] = set()
        self.queued_workers: set[str] = set()
        self.membership_open = True

    def join(self, worker_id: str) -> dict[str, Any]:
        if not worker_id or len(worker_id) > 128:
            raise ProtocolError("invalid_worker", "worker_id must be 1..128 chars")
        with self.lock:
            if self.training_complete:
                return {"status": "training_complete", "round_id": self.round_id}
            new_membership = False
            if worker_id in self.active_workers:
                status = "active"
            elif self.closed or not self.membership_open:
                new_membership = worker_id not in self.queued_workers
                self.queued_workers.add(worker_id)
                status = "waiting_next_round"
            else:
                new_membership = True
                self.active_workers.add(worker_id)
                status = "active"
            if new_membership:
                self.events.append({"event": "worker_joined", "worker_id": worker_id,
                                    "status": status, "round_id": self.round_id,
                                    "at": time.time()})
            return {
                "status": status,
                "round_id": self.round_id if status == "active" else self.round_id + 1,
                "theta_version": self.theta_version,
                "theta_hash": self.theta_hash if status == "active" else None,
            }

    def start_next_round(
        self,
        units: list[dict[str, Any]],
        *,
        target_loss_tokens: int,
        min_units: int,
    ) -> dict[str, Any]:
        if not units:
            raise ValueError("next round requires at least one unit")
        if target_loss_tokens <= 0 or min_units <= 0 or min_units > len(units):
            raise ValueError("invalid next-round budget")
        next_units: dict[str, dict[str, Any]] = {}
        for supplied in units:
            unit = dict(supplied)
            unit_id = str(unit["unit_id"])
            if unit_id in next_units:
                raise ValueError(f"duplicate unit_id {unit_id}")
            if int(unit["loss_tokens"]) <= 0 or int(unit["total_tokens"]) <= 0:
                raise ValueError(f"invalid token counts for {unit_id}")
            if unit.get("unit_sha256") != unit_digest(unit):
                raise ValueError(f"unit hash mismatch for {unit_id}")
            next_units[unit_id] = unit
        with self.lock:
            if not self.closed:
                raise ProtocolError("round_open", "cannot advance before the current round closes", 409)
            if self.training_complete:
                raise ProtocolError("training_complete", "cannot advance after training completion", 409)
            self.round_id += 1
            self.units = next_units
            self.pending = deque(next_units)
            self.unit_status = {unit_id: "pending" for unit_id in next_units}
            self.leases = {}
            self.accepted = {}
            self.grad_sums = None
            self.loss_tokens = 0
            self.loss_sum = 0.0
            self.target_loss_tokens = int(target_loss_tokens)
            self.min_units = int(min_units)
            self.closed = False
            self.close_metrics = None
            self.active_workers.update(self.queued_workers)
            promoted = sorted(self.queued_workers)
            self.queued_workers.clear()
            self.membership_open = True
            self.events.append({"event": "round_opened", "round_id": self.round_id,
                                "theta_version": self.theta_version,
                                "theta_hash": self.theta_hash,
                                "unit_count": len(next_units),
                                "promoted_workers": promoted, "at": time.time()})
            return self.descriptor()

    def descriptor(self) -> dict[str, Any]:
        with self.lock:
            return {
                "protocol_version": "rwkv-sft-http-v1",
                "run_id": self.run_id,
                "round_id": self.round_id,
                "theta_version": self.theta_version,
                "theta_hash": self.theta_hash,
                "status": "closed" if self.closed else "open",
                "training_complete": self.training_complete,
                "base": self.base,
                "tokenizer": self.tokenizer,
                "worker_code_sha256": self.worker_code_sha256,
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": [
                    "att.receptance", "att.key", "att.value", "att.output"
                ]},
                "target_loss_tokens": self.target_loss_tokens,
                "min_units": self.min_units,
                "received_loss_tokens": self.loss_tokens,
                "received_units": len(self.accepted),
                "unit_count": len(self.units),
                "active_workers": len(self.active_workers),
                "queued_workers": len(self.queued_workers),
                "membership_open": self.membership_open,
                "close_metrics": self.close_metrics,
            }

    def _encoded_theta(self) -> bytes:
        """The adapter as safetensors bytes, encoded once per theta version."""
        cache = self._snapshot_cache
        if cache is None or cache[0] != self.theta_version:
            cache = (self.theta_version, encode_tensors(self.theta))
            self._snapshot_cache = cache
        return cache[1]

    def snapshot_bytes(self) -> bytes:
        with self.lock:
            return self._encoded_theta()

    def hyperparameters(self) -> dict[str, Any]:
        with self.lock:
            return optimizer_hyperparameters(self.optimizer)

    def export_state(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, str]:
        """Return (theta, optimizer tensors, version, hash) captured under one lock."""
        with self.lock:
            theta = {name: value.clone() for name, value in self.theta.items()}
            return (
                theta,
                optimizer_state_tensors(self.optimizer, self.params),
                self.theta_version,
                self.theta_hash,
            )

    def snapshot(self) -> tuple[bytes, str, int]:
        """Return adapter bytes with the theta hash/version they encode.

        All three values are read under one lock so that a round closing
        concurrently cannot pair the bytes of one adapter version with the
        hash of another.
        """
        with self.lock:
            return self._encoded_theta(), self.theta_hash, self.theta_version

    def signal_training_complete(self, reason: str = "training_finished") -> dict[str, Any]:
        with self.lock:
            if not self.closed:
                raise ProtocolError("round_open", "cannot finish training while a round is open", 409)
            if not self.training_complete:
                self.training_complete = True
                self.events.append({
                    "event": "training_complete",
                    "run_id": self.run_id,
                    "round_id": self.round_id,
                    "reason": reason[:256],
                    "at": time.time(),
                })
            return {"training_complete": True, "run_id": self.run_id}

    def _expire_leases(self, now: float) -> None:
        for lease_id, lease in list(self.leases.items()):
            if lease.expires_at <= now:
                self.leases.pop(lease_id, None)
                if self.unit_status.get(lease.unit_id) == "leased":
                    self.unit_status[lease.unit_id] = "pending"
                    self.pending.append(lease.unit_id)
                self.events.append({
                    "event": "lease_expired", "unit_id": lease.unit_id,
                    "lease_id": lease.lease_id, "worker_id": lease.worker_id,
                    "at": now,
                })

    def lease(self, worker_id: str) -> dict[str, Any] | None:
        if not worker_id or len(worker_id) > 128:
            raise ProtocolError("invalid_worker", "worker_id must be 1..128 chars")
        with self.lock:
            if self.closed:
                return None
            if worker_id not in self.active_workers:
                raise ProtocolError("join_pending", "worker is queued for the next round", 409)
            self.membership_open = False
            now = time.time()
            self._expire_leases(now)
            while self.pending:
                unit_id = self.pending.popleft()
                if self.unit_status[unit_id] != "pending":
                    continue
                lease = Lease(
                    lease_id=uuid.uuid4().hex,
                    unit_id=unit_id,
                    worker_id=worker_id,
                    expires_at=now + self.lease_seconds,
                    round_id=self.round_id,
                    theta_version=self.theta_version,
                    theta_hash=self.theta_hash,
                )
                self.leases[lease.lease_id] = lease
                self.unit_status[unit_id] = "leased"
                self.events.append({
                    "event": "lease_issued", "unit_id": unit_id,
                    "lease_id": lease.lease_id, "worker_id": worker_id,
                    "expires_at": lease.expires_at,
                })
                unit = self.units[unit_id]
                return {
                    "run_id": self.run_id,
                    "round_id": self.round_id,
                    "theta_version": self.theta_version,
                    "theta_hash": self.theta_hash,
                    "lease_id": lease.lease_id,
                    "worker_id": worker_id,
                    "lease_seconds": self.lease_seconds,
                    "lease_expires_at": lease.expires_at,
                    "unit": unit,
                }
            if len(self.accepted) >= self.min_units and self.loss_tokens >= self.target_loss_tokens:
                self._close_round()
            return None

    def heartbeat(self, lease_id: str, worker_id: str) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                raise ProtocolError("round_closed", "round is closed", 409)
            now = time.time()
            self._expire_leases(now)
            lease = self.leases.get(lease_id)
            if lease is None:
                raise ProtocolError("unknown_lease", "lease is missing or expired", 409)
            if lease.worker_id != worker_id:
                raise ProtocolError("worker_lease_mismatch", "lease belongs to another worker", 403)
            lease.expires_at = now + self.lease_seconds
            event = {"event": "heartbeat", "lease_id": lease_id,
                     "unit_id": lease.unit_id, "worker_id": worker_id,
                     "expires_at": lease.expires_at}
            self.events.append(event)
            return {
                "lease_id": lease_id,
                "lease_expires_at": lease.expires_at,
                "round_id": lease.round_id,
                "theta_version": lease.theta_version,
                "theta_hash": lease.theta_hash,
            }

    def release(self, lease_id: str, worker_id: str, reason: str = "worker_leaving") -> dict[str, Any]:
        with self.lock:
            if self.closed:
                raise ProtocolError("round_closed", "round is closed", 409)
            now = time.time()
            self._expire_leases(now)
            lease = self.leases.get(lease_id)
            if lease is None:
                raise ProtocolError("unknown_lease", "lease is missing or expired", 409)
            if lease.worker_id != worker_id:
                raise ProtocolError("worker_lease_mismatch", "lease belongs to another worker", 403)
            self.leases.pop(lease_id, None)
            self.unit_status[lease.unit_id] = "pending"
            self.pending.appendleft(lease.unit_id)
            event = {"event": "lease_released", "lease_id": lease_id,
                     "unit_id": lease.unit_id, "worker_id": worker_id,
                     "reason": reason[:256], "at": now}
            self.events.append(event)
            return {"status": "released", "unit_id": lease.unit_id}

    def _reject_and_requeue(self, lease: Lease, code: str, message: str, status: int = 409):
        self.leases.pop(lease.lease_id, None)
        if not self.closed and self.unit_status.get(lease.unit_id) == "leased":
            self.unit_status[lease.unit_id] = "pending"
            self.pending.append(lease.unit_id)
        self.events.append({"event": "result_rejected", "code": code,
                            "unit_id": lease.unit_id, "lease_id": lease.lease_id,
                            "worker_id": lease.worker_id, "message": message})
        raise ProtocolError(code, message, status)

    def submit_result(self, metadata: dict[str, Any], payload: "bytes | PayloadFile") -> dict[str, Any]:
        with self.lock:
            unit_id = str(metadata.get("unit_id", ""))
            if unit_id in self.accepted:
                self.events.append({"event": "duplicate_result", "unit_id": unit_id,
                                    "worker_id": metadata.get("worker_id")})
                raise ProtocolError("duplicate_unit", f"unit {unit_id} already counted", 409)
            if self.closed:
                raise ProtocolError("round_closed", "round is closed", 409)
            lease_id = str(metadata.get("lease_id", ""))
            lease = self.leases.get(lease_id)
            if lease is None:
                raise ProtocolError("unknown_lease", "lease is missing, expired, or already used", 409)
            now = time.time()
            if lease.expires_at <= now:
                self._expire_leases(now)
                raise ProtocolError("expired_lease", "lease expired; unit was returned to queue", 409)
            if lease.unit_id != unit_id:
                self._reject_and_requeue(lease, "unit_lease_mismatch", "lease does not belong to unit")
            if lease.worker_id != str(metadata.get("worker_id", "")):
                self._reject_and_requeue(lease, "worker_lease_mismatch", "lease belongs to another worker", 403)
            if lease.round_id != int(metadata.get("round_id", -1)) or lease.round_id != self.round_id:
                self._reject_and_requeue(lease, "stale_round", "result round_id is stale")
            if lease.theta_version != int(metadata.get("theta_version", -1)):
                self._reject_and_requeue(lease, "stale_theta", "result theta_version is stale")
            if lease.theta_hash != str(metadata.get("theta_hash", "")) or lease.theta_hash != self.theta_hash:
                self._reject_and_requeue(lease, "stale_theta", "result theta hash is stale")
            if str(metadata.get("worker_code_sha256", "")) != self.worker_code_sha256:
                self._reject_and_requeue(lease, "worker_code_mismatch", "worker code hash differs", 403)
            if payload_sha256(payload) != str(metadata.get("gradient_sha256", "")):
                self._reject_and_requeue(lease, "gradient_hash_mismatch", "gradient payload hash differs")
            unit = self.units[unit_id]
            if str(metadata.get("unit_sha256", "")) != str(unit["unit_sha256"]):
                self._reject_and_requeue(lease, "unit_hash_mismatch", "result unit hash differs from manifest")
            loss_tokens = int(metadata.get("loss_tokens", -1))
            if loss_tokens != int(unit["loss_tokens"]):
                self._reject_and_requeue(lease, "loss_token_mismatch", "reported loss-token count differs")
            loss_sum = float(metadata.get("loss_sum", float("nan")))
            if not math.isfinite(loss_sum) or loss_sum < 0:
                self._reject_and_requeue(lease, "invalid_loss", "loss_sum must be finite and nonnegative")
            try:
                source = open_gradient_payload(payload)
            except Exception as exc:
                self._reject_and_requeue(lease, "invalid_payload", f"cannot decode safetensors: {exc}")
            with source:
                try:
                    names = source.names()
                except Exception as exc:
                    self._reject_and_requeue(lease, "invalid_payload", f"cannot decode safetensors: {exc}")
                if names != set(self.expected):
                    self._reject_and_requeue(lease, "gradient_names", "gradient tensor names differ from schema")
                # Validate every tensor before adding any, so a bad one cannot leave
                # a half-added unit in the running sum.
                squared_norm = 0.0
                for name, expected_shape in self.expected.items():
                    try:
                        shape = source.shape(name)
                        grad = source.tensor(name) if shape == expected_shape else None
                    except Exception as exc:
                        self._reject_and_requeue(lease, "invalid_payload", f"cannot decode safetensors: {exc}")
                    if grad is None:
                        self._reject_and_requeue(lease, "gradient_shape", f"wrong shape for {name}")
                    grad = grad.to(device="cpu", dtype=torch.float32).contiguous()
                    if not torch.isfinite(grad).all().item():
                        self._reject_and_requeue(lease, "nonfinite_gradient", f"non-finite gradient {name}")
                    squared_norm += float(torch.sum(grad * grad).item())
                if not math.isfinite(squared_norm):
                    self._reject_and_requeue(lease, "invalid_gradient_norm", "gradient norm is not finite")
                if self.grad_sums is None:
                    self.grad_sums = {name: torch.zeros_like(value) for name, value in self.theta.items()}
                for name in self.expected:
                    self.grad_sums[name].add_(
                        source.tensor(name).to(device="cpu", dtype=torch.float32).contiguous()
                    )

            self.leases.pop(lease_id, None)
            self.unit_status[unit_id] = "done"
            result = {
                "unit_id": unit_id,
                "lease_id": lease_id,
                "worker_id": lease.worker_id,
                "round_id": self.round_id,
                "theta_version": self.theta_version,
                "theta_hash": self.theta_hash,
                "loss_tokens": loss_tokens,
                "loss_sum": loss_sum,
                "gradient_sha256": metadata["gradient_sha256"],
                "gradient_norm": math.sqrt(squared_norm),
                "accepted_at": now,
            }
            self.accepted[unit_id] = result
            self.loss_tokens += loss_tokens
            self.loss_sum += loss_sum
            self.events.append(dict(result) | {"event": "result_accepted"})
            closed = False
            if self.loss_tokens >= self.target_loss_tokens and len(self.accepted) >= self.min_units:
                self._close_round()
                closed = True
            return {
                "status": "accepted",
                "unit_id": unit_id,
                "received_units": len(self.accepted),
                "received_loss_tokens": self.loss_tokens,
                "round_closed": closed,
                "theta_version": self.theta_version,
                "theta_hash": self.theta_hash,
                "close_metrics": self.close_metrics,
            }

    def _close_round(self) -> None:
        if self.closed:
            return
        if self.loss_tokens <= 0 or len(self.accepted) < self.min_units:
            raise ProtocolError("round_not_ready", "minimum units and loss tokens are required", 409)
        grad_sums, self.grad_sums = self.grad_sums, None
        if grad_sums is None:
            raise ProtocolError("round_not_ready", "no accepted gradients to apply", 409)
        mean_grads = {name: value / self.loss_tokens for name, value in grad_sums.items()}
        del grad_sums
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in mean_grads.values()])).item()
        if self.global_grad_clip is not None and norm > self.global_grad_clip:
            scale = self.global_grad_clip / max(norm, 1e-12)
            mean_grads = {name: grad * scale for name, grad in mean_grads.items()}
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.params.items():
            parameter.grad = mean_grads[name]
        self.optimizer.step()
        self.theta = {name: parameter.detach().clone() for name, parameter in self.params.items()}
        self._snapshot_cache = None
        old_hash = self.theta_hash
        self.theta_version += 1
        self.theta_hash = tensor_map_sha256(self.theta)
        self.closed = True
        self.membership_open = False
        self.close_metrics = {
            "unit_count": len(self.accepted),
            "loss_tokens": self.loss_tokens,
            "loss_sum": self.loss_sum,
            "mean_loss": self.loss_sum / self.loss_tokens,
            "preclip_gradient_norm": norm,
            "global_grad_clip": self.global_grad_clip,
            "theta_hash_before": old_hash,
            "theta_hash_after": self.theta_hash,
        }
        self.events.append({"event": "round_closed", "round_id": self.round_id,
                            "theta_version": self.theta_version, "metrics": self.close_metrics})

