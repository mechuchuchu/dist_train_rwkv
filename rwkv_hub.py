"""Explicit, revision-pinned Hugging Face checkpoint materialization."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


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
) -> str | None:
    """Upload only a PEFT adapter snapshot and its provenance to the Hub."""
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
    if private:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        if getattr(repo_info, "private", None) is False:
            raise PermissionError(
                f"destination repository {repo_id} is public; use a private repo or explicitly opt into public upload"
            )
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
        (staging / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        result = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=path_in_repo,
            commit_message=f"Upload RWKV adapter snapshot {metadata.get('theta_hash', '')[:12]}",
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "adapter_metadata.json"],
            token=token,
        )
    return getattr(result, "commit_url", None)
