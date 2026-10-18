#!/usr/bin/env python3
"""Write (or check) release-manifest.json for a locally converted HF checkpoint.

The training protocol identifies a base model by the SHA-256 of the
`release-manifest.json` inside its folder: the data shard records it, the
coordinator pins it, and every worker refuses to join unless its own copy hashes
the same. The upstream RWKV repositories ship that file; a checkpoint made with
convert_pth_to_hf.py does not, so this script writes one.

For the manifest to pin the weights and not just its own name it lists the size
and SHA-256 of every file in the folder, and the parameter count read from the
safetensors headers (`conversion.serialized_parameter_count`, which the
coordinator requires). The output is deterministic: no timestamps, sorted keys,
so the same folder always gives the same manifest hash.

Needs only the standard library. Run it once after converting, then copy or push
the folder together with the manifest. On every other machine,

    python make_release_manifest.py --check <folder>

re-hashes the files and fails if any differ from the manifest, which is the only
check that a copied folder really holds the same weights: workers compare the
manifest's hash, not the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

MANIFEST_NAME = "release-manifest.json"
SCHEMA = "dist-train-rwkv-release-manifest-v1"
BASE_VOCAB_SIZE = 65536  # RWKV World tokenizer ids 0..65535; a converted checkpoint appends after them
HEADER_LIMIT = 100 * 1024 * 1024  # a safetensors header is a small JSON; refuse anything absurd


class ManifestError(Exception):
    """The folder cannot be described, or does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listed_files(folder: Path) -> list[Path]:
    """Every regular file except the manifest, hidden entries and Python caches, in a fixed order."""
    found = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_file() and relative.as_posix() != MANIFEST_NAME:
            found.append(path)
    return found


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ManifestError(f"{path.name}: too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > HEADER_LIMIT:
            raise ManifestError(f"{path.name}: implausible safetensors header size {length}")
        body = stream.read(length)
    if len(body) != length:
        raise ManifestError(f"{path.name}: safetensors header is cut off")
    try:
        header = json.loads(body)
    except ValueError as exc:
        raise ManifestError(f"{path.name}: unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ManifestError(f"{path.name}: safetensors header is not an object")
    return header


def read_config(folder: Path) -> dict:
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise ManifestError(f"{folder} has no config.json")
    try:
        config = json.loads(config_path.read_text())
        return {
            "vocab_size": int(config["vocab_size"]),
            "hidden_size": int(config["hidden_size"]),
            "num_hidden_layers": int(config["num_hidden_layers"]),
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"{config_path} lacks a usable vocab_size / hidden_size / num_hidden_layers: {exc!r}") from exc


def describe_weights(folder: Path, config: dict) -> dict:
    """Parameter count and tensor layout from the safetensors headers; no weights are read."""
    shards = [path for path in listed_files(folder) if path.suffix == ".safetensors"]
    if not shards:
        raise ManifestError(f"{folder} holds no .safetensors weights")
    tensors: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, int] = {}
    for shard in shards:
        for name, entry in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ManifestError(f"tensor {name} appears in more than one shard")
            try:
                shape = tuple(int(dim) for dim in entry["shape"])
                dtype = str(entry["dtype"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestError(f"{shard.name}: bad header entry for {name}") from exc
            tensors[name] = shape
            dtypes[dtype] = dtypes.get(dtype, 0) + 1

    index_path = folder / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            indexed = set(json.loads(index_path.read_text())["weight_map"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ManifestError(f"{index_path.name} is unreadable: {exc!r}") from exc
        if indexed != set(tensors):
            missing = sorted(indexed - set(tensors))[:3]
            extra = sorted(set(tensors) - indexed)[:3]
            raise ManifestError(
                f"the shards do not match {index_path.name} (listed but absent: {missing}; present but unlisted: {extra}); "
                "a shard is missing or from another conversion"
            )

    # The converter keeps the reference key names: the embedding is rwkv7.emb.weight and the head head.weight.
    for name in ("rwkv7.emb.weight", "head.weight"):
        if name in tensors and tensors[name] != (config["vocab_size"], config["hidden_size"]):
            raise ManifestError(
                f"{name} has shape {tensors[name]} but config.json says vocab_size {config['vocab_size']}, "
                f"hidden_size {config['hidden_size']}; the weights and config come from different conversions"
            )
    if config["vocab_size"] < BASE_VOCAB_SIZE:
        raise ManifestError(f"vocab_size {config['vocab_size']} is below the RWKV World vocabulary of {BASE_VOCAB_SIZE}")

    count = 0
    for shape in tensors.values():
        size = 1
        for dim in shape:
            size *= dim
        count += size
    return {
        "serialized_parameter_count": count,
        "tensor_count": len(tensors),
        "dtypes": dict(sorted(dtypes.items())),
    }


def build_manifest(folder: Path) -> dict:
    config = read_config(folder)
    files = {
        path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in listed_files(folder)
    }
    return {
        "schema": SCHEMA,
        "model": {
            **config,
            "base_vocab_size": BASE_VOCAB_SIZE,
            "new_token_count": config["vocab_size"] - BASE_VOCAB_SIZE,
        },
        "conversion": describe_weights(folder, config),
        "files": files,
    }


def render(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def differences(recorded: dict, actual: dict) -> list[str]:
    """Human-readable differences between a manifest on disk and one built from the folder now."""
    problems = []
    recorded_files, actual_files = recorded.get("files", {}), actual["files"]
    for name in sorted(set(recorded_files) | set(actual_files)):
        if name not in actual_files:
            problems.append(f"{name}: listed in the manifest but missing from the folder")
        elif name not in recorded_files:
            problems.append(f"{name}: in the folder but not in the manifest")
        elif recorded_files[name] != actual_files[name]:
            problems.append(f"{name}: contents differ from the manifest")
    for section in ("model", "conversion"):
        if recorded.get(section) != actual[section]:
            problems.append(f"{section}: {recorded.get(section)} in the manifest, {actual[section]} in the folder")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="the converted HF checkpoint folder")
    parser.add_argument("--check", action="store_true",
                        help="verify the folder against its existing manifest instead of writing one")
    parser.add_argument("--force", action="store_true", help="replace an existing, different manifest")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    folder = args.folder.expanduser()
    if not folder.is_dir():
        raise ManifestError(f"{folder} is not a folder")
    manifest_path = folder / MANIFEST_NAME
    actual = build_manifest(folder)
    rendered = render(actual)

    if args.check:
        if not manifest_path.is_file():
            raise ManifestError(f"{manifest_path} does not exist; run without --check to create it")
        try:
            recorded = json.loads(manifest_path.read_text())
        except ValueError as exc:
            raise ManifestError(f"{manifest_path} is not valid JSON: {exc}") from exc
        problems = differences(recorded, actual)
        if problems:
            print(f"{folder} does NOT match its {MANIFEST_NAME}:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"OK: {len(actual['files'])} files match {MANIFEST_NAME}")
        print(f"manifest sha256: {hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
        return 0

    if manifest_path.is_file() and manifest_path.read_bytes() != rendered and not args.force:
        raise ManifestError(
            f"{manifest_path} already exists and differs from what this folder produces now. If the folder was "
            "changed on purpose, pass --force (the new manifest hash invalidates data shards made for the old one); "
            "otherwise run with --check to see what changed"
        )
    manifest_path.write_bytes(rendered)
    manifest_sha256 = hashlib.sha256(rendered).hexdigest()
    model = actual["model"]
    print(f"wrote {manifest_path}")
    print(f"  files: {len(actual['files'])}, parameters: {actual['conversion']['serialized_parameter_count']:,}, "
          f"tensors: {actual['conversion']['tensor_count']}, dtypes: {actual['conversion']['dtypes']}")
    print(f"  vocab {model['vocab_size']} = {BASE_VOCAB_SIZE} RWKV World ids + {model['new_token_count']} new "
          "(their embedding and head rows are trained with the LoRA)")
    print(f"manifest sha256: {manifest_sha256}")
    print()
    print("Next: the protocol needs --model-id and a 40-hex --model-revision.")
    print("  * If you push this folder to a Hugging Face repo, use that repo id and the commit sha the push returns.")
    print("    Workers can then download it with --fetch-model-from-hf only if the repo is public (they send no token).")
    print("  * If you copy the folder to each machine by hand, any repo-style id works, and this value is a stable")
    print(f"    stand-in revision derived from the manifest:  {manifest_sha256[:40]}")
    print("    Do not use --fetch-model-from-hf with it. Run --check on each machine after copying.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
