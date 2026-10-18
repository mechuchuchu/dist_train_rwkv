#!/usr/bin/env python3
"""Build a small immutable, pre-tokenized RWKV7 SFT shard for protocol runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL_ID = "RWKV/RWKV7-G1j-1.5B-20260831"
DEFAULT_MODEL_REVISION = "2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
DATASET_ID = "trl-lib/Capybara"
DATASET_REVISION = "e235e846458bff3398a88aed812347f7f0756520"
ROW_IDS = [1, 16, 21, 25]
SCHEMA_VERSION = "rwkv-token-units-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/g1j_local_protocol/units.json"))
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="base model repository whose tokenizer is used (default: %(default)s)")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="immutable commit hash of --model-id")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="local snapshot of the model (default: /workspace/<model name>)")
    args = parser.parse_args()
    model_path = args.model_path or Path("/workspace") / args.model_id.rstrip("/").rsplit("/", 1)[-1]
    if not (model_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing local model/tokenizer: {model_path}")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID,
        split="train[:32]",
        revision=DATASET_REVISION,
    ).select(ROW_IDS)
    units = []
    for row_id, row in zip(ROW_IDS, dataset):
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(encoded["input_ids"][: args.max_length])
        assistant_mask = list(encoded["assistant_masks"][: args.max_length])
        labels = [token_id if is_assistant else -100 for token_id, is_assistant in zip(ids, assistant_mask)]
        loss_tokens = sum(label != -100 for label in labels[1:])
        if loss_tokens == 0:
            raise ValueError(f"Row {row_id} has no assistant labels after truncation")
        unit = {
            "unit_id": f"capybara-train-row-{row_id:05d}",
            "source_row": row_id,
            "input_ids": ids,
            "labels": labels,
            "total_tokens": len(ids),
            "loss_tokens": loss_tokens,
        }
        unit["unit_sha256"] = sha256(canonical_json(unit))
        units.append(unit)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "repo": args.model_id,
            "revision": args.model_revision,
            "release_manifest_sha256": file_sha256(model_path / "release-manifest.json"),
        },
        "dataset": {
            "repo": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train[:32]",
            "row_ids": ROW_IDS,
        },
        "tokenizer": {
            "tokenizer_json_sha256": file_sha256(model_path / "tokenizer.json"),
            "chat_template_sha256": sha256((tokenizer.chat_template or "").encode()),
            "max_length": args.max_length,
            "assistant_only_loss": True,
        },
        "units": units,
    }
    manifest["manifest_sha256"] = sha256(canonical_json(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "unit_count": len(units),
        "total_tokens": sum(unit["total_tokens"] for unit in units),
        "loss_tokens": sum(unit["loss_tokens"] for unit in units),
        "unit_ids": [unit["unit_id"] for unit in units],
    }, indent=2))


if __name__ == "__main__":
    main()
