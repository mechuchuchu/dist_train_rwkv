"""Explicit, revision-pinned Hugging Face checkpoint materialization."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

OPTIMIZER_POLICIES = ("auto", "always", "never")
OPTIMIZER_FILE = "optimizer.safetensors"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION_DIR = re.compile(r"^theta-(\d{6,})$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_hf_checkpoint(
    repo_id: str,
    revision: str,
    local_dir: Path,
    *,
    fetch_from_hf: bool,
    token: str | bool | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Return a local snapshot, downloading it only when explicitly requested."""
    destination = Path(local_dir).expanduser()
    if fetch_from_hf:
        from huggingface_hub import snapshot_download

        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            cache_dir=str(cache_dir.expanduser()) if cache_dir is not None else None,
            token=token,
        )
        destination = Path(downloaded)
    if not destination.is_dir():
        action = "HF download was requested but did not create" if fetch_from_hf else "Missing local checkpoint"
        raise FileNotFoundError(f"{action}: {destination}")
    if not (destination / "release-manifest.json").is_file():
        raise FileNotFoundError(
            f"{destination} lacks release-manifest.json; expected the pinned HF Transformers snapshot"
        )
    return destination.resolve()


def push_adapter_snapshot(
    *,
    repo_id: str,
    token: str,
    adapter_config_path: Path,
    adapter_weights_path: Path,
    metadata: dict[str, Any],
    path_in_repo: str,
    base_model_id: str,
    base_model_revision: str,
    private: bool = True,
    optimizer_weights_path: Path | None = None,
    optimizer_policy: str = "auto",
    report: dict[str, Any] | None = None,
) -> str | None:
    """Upload only a PEFT adapter snapshot and its provenance to the Hub.

    `optimizer_weights_path` is the AdamW state as safetensors. Whether it goes
    up is decided by `optimizer_policy` against the repository's actual
    visibility, read from the Hub rather than inferred from the `private` flag
    (an existing private repo stays private even when public was requested):

    * ``never``  - the optimizer state is not uploaded.
    * ``auto``   - uploaded only when the repository is confirmed private.
    * ``always`` - uploaded even to a public repository. The moments are a
      function of the training gradients, so this is an explicit choice.

    When `report` is given it receives ``repo_private`` and ``optimizer_uploaded``.
    """
    if optimizer_policy not in OPTIMIZER_POLICIES:
        raise ValueError(f"optimizer_policy must be one of {', '.join(OPTIMIZER_POLICIES)}")
    if not token:
        raise ValueError("a Hugging Face write token is required to push an adapter")
    if not repo_id.strip():
        raise ValueError("an adapter Hub repository id is required")
    if Path(path_in_repo).is_absolute() or ".." in Path(path_in_repo).parts:
        raise ValueError("path_in_repo must be a relative path inside the Hub repository")
    config_source = Path(adapter_config_path)
    weights_source = Path(adapter_weights_path)
    if not config_source.is_file() or not weights_source.is_file():
        raise FileNotFoundError("adapter config and weights must exist before pushing")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    repo_info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
    repo_private = getattr(repo_info, "private", None)
    if private and repo_private is False:
        raise PermissionError(
            f"destination repository {repo_id} is public; use a private repo or explicitly opt into public upload"
        )
    optimizer_source = Path(optimizer_weights_path) if optimizer_weights_path is not None else None
    upload_optimizer = (
        optimizer_source is not None
        and optimizer_source.is_file()
        and (optimizer_policy == "always" or (optimizer_policy == "auto" and repo_private is True))
    )
    if report is not None:
        report["repo_private"] = repo_private
        report["optimizer_uploaded"] = upload_optimizer
    with tempfile.TemporaryDirectory(prefix="rwkv-adapter-hub-") as temp_dir:
        staging = Path(temp_dir)
        config = json.loads(config_source.read_text())
        config["base_model_name_or_path"] = base_model_id
        config["revision"] = base_model_revision
        config["inference_mode"] = True
        (staging / "adapter_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        shutil.copy2(weights_source, staging / "adapter_model.safetensors")
        metadata = dict(metadata)
        metadata["optimizer_included"] = upload_optimizer
        if upload_optimizer:
            shutil.copy2(optimizer_source, staging / OPTIMIZER_FILE)
            metadata["optimizer_sha256"] = _file_sha256(staging / OPTIMIZER_FILE)
        (staging / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        result = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=path_in_repo,
            commit_message=f"Upload RWKV adapter snapshot {metadata.get('theta_hash', '')[:12]}",
            allow_patterns=["adapter_config.json", "adapter_model.safetensors",
                            "adapter_metadata.json", OPTIMIZER_FILE],
            token=token,
        )
    return getattr(result, "commit_url", None)


def _checked_run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id or ""):
        raise ValueError(f"unsafe run id for a Hub path: {run_id!r}")
    return run_id


def list_hub_checkpoint_versions(*, repo_id: str, run_id: str, token: str | None) -> list[int]:
    """Versions of `run_id` that are fully uploaded to the Hub repository, newest first."""
    prefix = f"checkpoints/{_checked_run_id(run_id)}/"
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    found: dict[int, set[str]] = {}
    for path in files:
        if not path.startswith(prefix):
            continue
        folder, _, name = path[len(prefix):].partition("/")
        match = _VERSION_DIR.match(folder)
        if match:
            found.setdefault(int(match.group(1)), set()).add(name)
    needed = {"adapter_model.safetensors", "adapter_metadata.json"}
    return sorted((version for version, names in found.items() if needed <= names), reverse=True)


def download_hub_checkpoint(
    *, repo_id: str, run_id: str, version: int, token: str | None, local_dir: Path
) -> Path:
    """Download one pushed checkpoint folder and return its local path."""
    if version < 0:
        raise ValueError("version cannot be negative")
    relative = f"checkpoints/{_checked_run_id(run_id)}/theta-{version:06d}"
    from huggingface_hub import snapshot_download

    destination = Path(local_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{relative}/*"],
        local_dir=str(destination),
        token=token,
    )
    folder = destination / relative
    if not folder.is_dir():
        raise FileNotFoundError(f"Hub download did not produce {relative} for {repo_id}")
    return folder
