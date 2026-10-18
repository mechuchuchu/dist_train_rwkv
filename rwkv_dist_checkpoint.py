"""Crash-safe, versioned coordinator checkpoints for --resume.

One checkpoint is one directory, ``ckpt/theta-NNNNNN/``. It is built under a
temporary name and renamed into place only when complete, so a crash can never
leave a half-written version visible. Resume loads the newest directory that
verifies and falls back to an older one when it does not.

A checkpoint holds the adapter (theta), the AdamW moments and a ``state.json``
recording what the run was (data, plan, base model, optimizer settings) with a
SHA-256 for every file. Optimizer state is stored as safetensors, never as a
pickle, so a checkpoint fetched from the Hub cannot execute code when loaded.

Work in progress inside a round is deliberately not checkpointed: after a
restart the interrupted round is issued again from its first unit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rwkv_dist_protocol import is_saved_theta_name, tensor_map_sha256, theta_name_from_saved


CHECKPOINT_FORMAT = "rwkv-coordinator-ckpt-v1"
THETA_FILE = "theta.safetensors"
OPTIMIZER_FILE = "optimizer.safetensors"
STATE_FILE = "state.json"
HUB_ADAPTER_FILE = "adapter_model.safetensors"
HUB_METADATA_FILE = "adapter_metadata.json"

# Everything that must be identical for a saved version to continue this run.
COMPATIBILITY_KEYS = (
    "data_manifest_sha256",
    "round_plan_sha256",
    "planned_rounds",
    "base",
    "tokenizer",
    "hyperparameters",
)
# What a Hub push must record for its checkpoint to be resumable.
HUB_RESUME_KEYS = ("run_id", "theta_version", "theta_hash") + COMPATIBILITY_KEYS

_NAME = re.compile(r"^theta-(\d{6,})$")


class ResumeError(RuntimeError):
    """A checkpoint cannot be used to resume this launch."""


@dataclass
class LoadedCheckpoint:
    state: dict[str, Any]
    theta: dict[str, torch.Tensor]
    optimizer_tensors: dict[str, torch.Tensor]
    directory: Path
    # Newer versions that were passed over because they failed verification.
    skipped: list[str] = field(default_factory=list)


def checkpoint_name(version: int) -> str:
    return f"theta-{version:06d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def list_checkpoint_versions(root: Path) -> list[int]:
    """Versions with a committed directory under `root`, newest first."""
    if not Path(root).is_dir():
        return []
    versions = []
    for entry in Path(root).iterdir():
        match = _NAME.match(entry.name)
        if match and entry.is_dir():
            versions.append(int(match.group(1)))
    return sorted(versions, reverse=True)


def prune_checkpoints(root: Path, keep: int) -> list[int]:
    """Delete all but the newest `keep` versions; return the versions removed."""
    removed = []
    for version in list_checkpoint_versions(root)[keep:]:
        shutil.rmtree(Path(root) / checkpoint_name(version), ignore_errors=True)
        removed.append(version)
    return removed


def write_checkpoint(
    root: Path,
    *,
    state: dict[str, Any],
    theta: dict[str, torch.Tensor],
    optimizer_tensors: dict[str, torch.Tensor],
    keep: int = 3,
) -> Path:
    """Write ``root/theta-<version>`` atomically and return its path.

    A version that already exists is left untouched: its contents are fixed by
    the version number, and rewriting it could only introduce a window in which
    it is incomplete.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    root = Path(root)
    version = int(state["theta_version"])
    root.mkdir(parents=True, exist_ok=True)
    final = root / checkpoint_name(version)
    if final.exists():
        return final
    for leftover in root.glob(".tmp-*"):  # a previous write that never finished
        shutil.rmtree(leftover, ignore_errors=True)
    staging = root / f".tmp-{checkpoint_name(version)}-{os.getpid()}"
    staging.mkdir()
    try:
        save_file(theta, str(staging / THETA_FILE))
        save_file(optimizer_tensors, str(staging / OPTIMIZER_FILE))
        for name in (THETA_FILE, OPTIMIZER_FILE):
            _fsync_file(staging / name)
        document = {
            **state,
            "format": CHECKPOINT_FORMAT,
            "files": {
                THETA_FILE: file_sha256(staging / THETA_FILE),
                OPTIMIZER_FILE: file_sha256(staging / OPTIMIZER_FILE),
            },
        }
        state_path = staging / STATE_FILE
        state_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        _fsync_file(state_path)
        _fsync_dir(staging)
        os.replace(staging, final)  # the commit point
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _fsync_dir(root)
    prune_checkpoints(root, keep)
    return final


def load_checkpoint(directory: Path) -> LoadedCheckpoint:
    """Load one checkpoint directory, raising ResumeError on any inconsistency."""
    directory = Path(directory)
    match = _NAME.match(directory.name)
    try:
        state = json.loads((directory / STATE_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ResumeError(f"{directory.name}: cannot read {STATE_FILE}: {exc}") from exc
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ResumeError(f"{directory.name}: unsupported checkpoint format {state.get('format')!r}")
    if match is None or int(match.group(1)) != state.get("theta_version"):
        raise ResumeError(f"{directory.name}: directory name does not match its recorded version")
    files = state.get("files") or {}
    for name in (THETA_FILE, OPTIMIZER_FILE):
        path = directory / name
        if not path.is_file():
            raise ResumeError(f"{directory.name}: missing {name}")
        if file_sha256(path) != files.get(name):
            raise ResumeError(f"{directory.name}: {name} does not match its recorded hash")
    try:
        theta = load_file(str(directory / THETA_FILE), device="cpu")
        optimizer_tensors = load_file(str(directory / OPTIMIZER_FILE), device="cpu")
    except Exception as exc:
        raise ResumeError(f"{directory.name}: cannot decode tensors: {exc}") from exc
    if tensor_map_sha256(theta) != state.get("theta_hash"):
        raise ResumeError(f"{directory.name}: adapter does not match its recorded theta hash")
    return LoadedCheckpoint(state, theta, optimizer_tensors, directory)


def load_latest_checkpoint(root: Path) -> LoadedCheckpoint:
    """Load the newest verifying checkpoint under `root`."""
    versions = list_checkpoint_versions(root)
    if not versions:
        raise ResumeError(f"no checkpoints found under {root}")
    skipped: list[str] = []
    for version in versions:
        try:
            loaded = load_checkpoint(Path(root) / checkpoint_name(version))
        except ResumeError as exc:
            skipped.append(str(exc))
            continue
        loaded.skipped = skipped
        return loaded
    raise ResumeError(f"no usable checkpoint under {root}: " + "; ".join(skipped))


def load_hub_checkpoint(directory: Path) -> LoadedCheckpoint:
    """Verify and load a folder downloaded from a pushed Hub checkpoint.

    The optimizer file is optional: a push made without it (public repository,
    or --hf-push-optimizer never) still resumes the adapter, with the Adam
    moments starting cold.
    """
    directory = Path(directory)
    try:
        metadata = json.loads((directory / HUB_METADATA_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ResumeError(f"{directory}: cannot read {HUB_METADATA_FILE}: {exc}") from exc
    missing = [key for key in HUB_RESUME_KEYS if key not in metadata]
    if missing:
        raise ResumeError(
            f"{directory}: pushed without resume metadata (missing {', '.join(missing)}); "
            "it cannot be checked against this run, so it is not used"
        )
    try:
        raw = load_file(str(directory / HUB_ADAPTER_FILE), device="cpu")
    except Exception as exc:
        raise ResumeError(f"{directory}: cannot read {HUB_ADAPTER_FILE}: {exc}") from exc
    theta = {}
    for name, tensor in raw.items():
        if not is_saved_theta_name(name):
            continue
        theta[theta_name_from_saved(name)] = tensor.float().contiguous()
    if not theta or tensor_map_sha256(theta) != metadata["theta_hash"]:
        raise ResumeError(f"{directory}: adapter does not match the theta hash it was pushed with")
    optimizer_tensors: dict[str, torch.Tensor] = {}
    optimizer_path = directory / OPTIMIZER_FILE
    if optimizer_path.is_file():
        recorded = metadata.get("optimizer_sha256")
        if not recorded or file_sha256(optimizer_path) != recorded:
            raise ResumeError(f"{directory}: optimizer file does not match its recorded hash")
        try:
            optimizer_tensors = load_file(str(optimizer_path), device="cpu")
        except Exception as exc:
            raise ResumeError(f"{directory}: cannot decode optimizer tensors: {exc}") from exc
    state = {key: metadata[key] for key in HUB_RESUME_KEYS}
    return LoadedCheckpoint(state, theta, optimizer_tensors, directory)


def _short(value: Any, limit: int = 48) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def check_resume_compatibility(state: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse a checkpoint made for different data, plan, model or optimizer settings."""
    problems = [
        f"{key} (checkpoint {_short(state.get(key))}, this launch {_short(expected.get(key))})"
        for key in COMPATIBILITY_KEYS
        if state.get(key) != expected.get(key)
    ]
    if problems:
        raise ResumeError(
            "checkpoint belongs to a different run configuration: " + "; ".join(problems)
        )
