#!/usr/bin/env python3
"""Convert an upstream RWKV-7 G1 .pth checkpoint to a HF checkpoint with the K2 Horizon chat format.

The RWKV World token vocabulary and trie tokenizer stay intact. Only tokens
marked special in the K2 Horizon checkpoint and missing from RWKV are
appended. Their embedding and LM-head rows are initialized from the mean of the
existing RWKV rows for the literal special-token spelling; SFT can then teach
the model how to use them.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AddedToken, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.initialization import no_init_weights


DEFAULT_PTH = (
    "/workspace/.hf_home/hub/models--BlinkDL--rwkv7-g1/snapshots/"
    "2a17e911155a5be23a4ed2792e978c77038f14e2/"
    "rwkv7-g1k-1.5b-20260930-ctx25600.pth"
)
DEFAULT_OUTPUT = "/workspace/RWKV-LM/RWKV-v7/RWKV7-G1k-1.5B-k2-horizon-hf"
DEFAULT_RWKV_REFERENCE = "RWKV/RWKV7-G1k-1.5B-20260930"
DEFAULT_K2_CHECKPOINT = "IFM/K2-Horizon-32B"
K2_VISION_TOKENS = {"<image>", "</image>", "<video>", "</video>"}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def download_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def checkpoint_special_entries(tokenizer_json: dict) -> list[dict]:
    """Read only special tokens; ordinary checkpoint tokens are ignored."""
    result = []
    seen = set()
    for item in sorted(tokenizer_json.get("added_tokens", []), key=lambda x: x["id"]):
        content = item.get("content")
        if item.get("special") and isinstance(content, str) and content not in seen:
            result.append(item)
            seen.add(content)
    return result


def selected_special_entries(tokenizer_json: dict) -> list[dict]:
    entries = checkpoint_special_entries(tokenizer_json)
    # This conversion is text-only. Do not add image or video marker IDs.
    return [item for item in entries if item["content"] not in K2_VISION_TOKENS]


def make_added_token(item: dict) -> AddedToken:
    return AddedToken(
        item["content"],
        single_word=bool(item.get("single_word", False)),
        lstrip=bool(item.get("lstrip", False)),
        rstrip=bool(item.get("rstrip", False)),
        normalized=bool(item.get("normalized", False)),
        special=True,
    )


def make_hf_state_dict(source_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The HF RWKV7 wrapper nests the reference model under ``rwkv7``."""
    converted = {}
    for key, value in source_state.items():
        if key == "head.weight" or key.startswith("rwkv7."):
            target_key = key
        else:
            target_key = f"rwkv7.{key}"
        if target_key in converted:
            raise ValueError(f"duplicate target parameter after key mapping: {target_key}")
        converted[target_key] = value
    return converted


def resize_vocab_and_initialize_specials(
    model,
    old_vocab_size: int,
    new_vocab_size: int,
    new_token_init_ids: dict[int, list[int]],
) -> None:
    if new_vocab_size == old_vocab_size:
        model.config.vocab_size = new_vocab_size
        return

    old_input = model.get_input_embeddings()
    old_output = model.get_output_embeddings()
    if old_input.weight.shape[0] != old_vocab_size:
        raise ValueError(
            f"input embedding has {old_input.weight.shape[0]} rows, expected {old_vocab_size}"
        )
    if old_output.weight.shape[0] != old_vocab_size:
        raise ValueError(
            f"LM head has {old_output.weight.shape[0]} rows, expected {old_vocab_size}"
        )

    hidden_size = old_input.weight.shape[1]
    new_input = torch.nn.Embedding(
        new_vocab_size,
        hidden_size,
        device=old_input.weight.device,
        dtype=old_input.weight.dtype,
    )
    new_output = torch.nn.Linear(
        hidden_size,
        new_vocab_size,
        bias=old_output.bias is not None,
        device=old_output.weight.device,
        dtype=old_output.weight.dtype,
    )

    with torch.no_grad():
        new_input.weight[:old_vocab_size].copy_(old_input.weight)
        new_output.weight[:old_vocab_size].copy_(old_output.weight)
        if old_output.bias is not None:
            new_output.bias[:old_vocab_size].copy_(old_output.bias)

        for new_id in range(old_vocab_size, new_vocab_size):
            source_ids = new_token_init_ids.get(new_id)
            if not source_ids:
                raise ValueError(f"no RWKV token pieces available to initialize new token id {new_id}")
            source_index = torch.tensor(source_ids, dtype=torch.long, device=old_input.weight.device)
            new_input.weight[new_id].copy_(old_input.weight[source_index].float().mean(0).to(old_input.weight.dtype))

            source_index = source_index.to(old_output.weight.device)
            new_output.weight[new_id].copy_(old_output.weight[source_index].float().mean(0).to(old_output.weight.dtype))
            if old_output.bias is not None:
                new_output.bias[new_id].copy_(old_output.bias[source_index].float().mean().to(old_output.bias.dtype))

    model.set_input_embeddings(new_input)
    model.set_output_embeddings(new_output)
    model.config.vocab_size = new_vocab_size


def save_trie_tokenizer(
    tokenizer,
    rwkv_tokenizer_json: dict,
    source_special_entries: list[dict],
    missing_entries: list[dict],
    output_dir: Path,
    chat_template: str,
) -> dict[str, int]:
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(output_dir)

    # Keep the RWKV WordPiece trie and all 65,536 ordinary IDs untouched.
    # The custom RWKV tokenizer reads these appended entries as atomic specials.
    tokenizer_json = rwkv_tokenizer_json
    added_items = list(tokenizer_json.get("added_tokens", []))
    by_content = {item.get("content"): item for item in added_items}
    new_ids = {}

    for item in missing_entries:
        content = item["content"]
        token_id = tokenizer.convert_tokens_to_ids(content)
        if token_id < tokenizer.vocab_size:
            # This case means the spelling is already represented by a base
            # vocabulary token; preserve its existing ID rather than append.
            continue
        record = {
            "id": int(token_id),
            "content": content,
            "single_word": bool(item.get("single_word", False)),
            "lstrip": bool(item.get("lstrip", False)),
            "rstrip": bool(item.get("rstrip", False)),
            "normalized": bool(item.get("normalized", False)),
            "special": True,
        }
        prior = by_content.get(content)
        if prior is not None:
            if int(prior["id"]) != token_id:
                raise ValueError(f"special token {content!r} has conflicting tokenizer IDs")
            prior.update(record)
        else:
            added_items.append(record)
            by_content[content] = record
        new_ids[content] = int(token_id)

    tokenizer_json["added_tokens"] = sorted(added_items, key=lambda item: int(item["id"]))
    write_json(output_dir / "tokenizer.json", tokenizer_json)

    config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = read_json(config_path)
    tokenizer_config["chat_template"] = chat_template
    tokenizer_config["bos_token"] = tokenizer.bos_token
    tokenizer_config["eos_token"] = tokenizer.eos_token
    tokenizer_config["pad_token"] = tokenizer.pad_token
    tokenizer_config["unk_token"] = tokenizer.unk_token
    tokenizer_config["padding_side"] = tokenizer.padding_side
    # Transformers 5 exposes custom tokenizer specials through
    # ``extra_special_tokens``; the RWKV tokenizer JSON remains authoritative.
    tokenizer_config["extra_special_tokens"] = [
        str(token) for token in getattr(tokenizer, "extra_special_tokens", [])
    ]

    # Keep an explicit decoder map for stable added-token IDs on reload.
    decoder = tokenizer_config.setdefault("added_tokens_decoder", {})
    for item in source_special_entries:
        content = item["content"]
        token_id = tokenizer.convert_tokens_to_ids(content)
        if token_id >= tokenizer.vocab_size:
            decoder[str(token_id)] = {
                "content": content,
                "single_word": bool(item.get("single_word", False)),
                "lstrip": bool(item.get("lstrip", False)),
                "rstrip": bool(item.get("rstrip", False)),
                "normalized": bool(item.get("normalized", False)),
                "special": True,
            }
    write_json(config_path, tokenizer_config)
    (output_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")
    return new_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pth", type=Path, default=Path(DEFAULT_PTH), help="Upstream RWKV-7 .pth checkpoint")
    parser.add_argument("--output", type=Path, default=None, help="Output HF checkpoint directory")
    parser.add_argument(
        "--rwkv-reference-model",
        default=DEFAULT_RWKV_REFERENCE,
        help="HF RWKV7 repo providing config, remote model code, and the matching RWKV trie tokenizer",
    )
    parser.add_argument(
        "--k2-checkpoint",
        default=DEFAULT_K2_CHECKPOINT,
        help="Hugging Face repo providing the K2 Horizon special tokens and chat template; only special tokens are merged",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--chat-template-file", type=Path, default=None, help="Optional Jinja template override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pth.is_file():
        raise FileNotFoundError(args.pth)
    if args.output is None:
        args.output = Path(DEFAULT_OUTPUT)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load only config/code and tokenizer files from the matching HF RWKV7
    # reference repo. No reference model weights are downloaded.
    config = AutoConfig.from_pretrained(args.rwkv_reference_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.rwkv_reference_model, trust_remote_code=True)
    rwkv_tok_json_path = download_file(args.rwkv_reference_model, "tokenizer.json")
    rwkv_tokenizer_json = read_json(rwkv_tok_json_path)
    base_vocab_size = len(tokenizer)

    k2_checkpoint = args.k2_checkpoint
    source_tokenizer_json_path = download_file(k2_checkpoint, "tokenizer.json")
    source_tokenizer_json = read_json(source_tokenizer_json_path)
    source_special_entries = selected_special_entries(source_tokenizer_json)
    known_specials = set(tokenizer.all_special_tokens)
    missing_entries = [item for item in source_special_entries if item["content"] not in known_specials]

    # Tokenize the literal marker spellings BEFORE registering them as atomic
    # tokens; the mean rows provide a sane initialization for SFT to refine.
    pieces_by_content = {
        item["content"]: tokenizer.encode(item["content"], add_special_tokens=False)
        for item in missing_entries
    }
    if any(not pieces for pieces in pieces_by_content.values()):
        empty = [key for key, pieces in pieces_by_content.items() if not pieces]
        raise ValueError(f"RWKV trie returned no pieces for special tokens: {empty}")

    new_token_count = tokenizer.add_special_tokens(
        {"additional_special_tokens": [make_added_token(item) for item in missing_entries]}
    )
    if new_token_count != len(missing_entries):
        print(f"Tokenizer reported {new_token_count} new IDs for {len(missing_entries)} missing specials.")

    # Match K2 Horizon's text-only chat markers. Keep the original RWKV
    # id 0 untouched; K2's BOS/EOS spellings receive appended IDs.
    tokenizer.bos_token = "<|ifm|begin_of_text|>"
    tokenizer.eos_token = "<|ifm|endoftext|>"
    tokenizer.pad_token = None
    k2_eos_ids = [
        tokenizer.convert_tokens_to_ids("<|ifm|endoftext|>"),
        tokenizer.convert_tokens_to_ids("<|ifm|im_end|>"),
    ]

    if args.chat_template_file:
        chat_template = args.chat_template_file.read_text(encoding="utf-8")
    else:
        # This is the K2 text chat template; no image/video processor or model
        # components are copied into the RWKV checkpoint.
        chat_template = download_file(k2_checkpoint, "chat_template.jinja").read_text(
            encoding="utf-8"
        )

    # The upstream .pth uses reference model keys (emb.*, blocks.*, ln_out.*).
    # HF wraps those under rwkv7.* while leaving the LM head at head.weight.
    source_state = torch.load(args.pth, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(source_state, dict) or "emb.weight" not in source_state or "head.weight" not in source_state:
        raise ValueError("Expected an RWKV-7 state dict containing emb.weight and head.weight")
    source_dtype = source_state["emb.weight"].dtype
    source_vocab_size, hidden_size = source_state["emb.weight"].shape
    if source_vocab_size != base_vocab_size:
        raise ValueError(f".pth vocab is {source_vocab_size}; RWKV tokenizer has {base_vocab_size} base IDs")
    if config.hidden_size != hidden_size or config.vocab_size != source_vocab_size:
        raise ValueError(
            "The selected RWKV reference config does not match this .pth: "
            f"checkpoint is vocab={source_vocab_size}, hidden={hidden_size}; "
            f"reference is vocab={config.vocab_size}, hidden={config.hidden_size}. "
            "Pass the matching --rwkv-reference-model."
        )

    config.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    config.eos_token_id = k2_eos_ids
    config.pad_token_id = (
        None if tokenizer.pad_token is None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )

    # Skip expensive random initialization of the 1.5B parameter model; every
    # parameter is filled from the upstream .pth immediately after creation.
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=source_dtype,
        )
    if model.get_input_embeddings().weight.dtype != source_dtype:
        model.to(dtype=source_dtype)
    hf_state = make_hf_state_dict(source_state)
    model.load_state_dict(hf_state, strict=True)
    del hf_state, source_state
    gc.collect()

    new_token_init_ids = {}
    for content, pieces in pieces_by_content.items():
        token_id = tokenizer.convert_tokens_to_ids(content)
        if token_id >= base_vocab_size:
            new_token_init_ids[token_id] = pieces

    resize_vocab_and_initialize_specials(
        model,
        old_vocab_size=base_vocab_size,
        new_vocab_size=len(tokenizer),
        new_token_init_ids=new_token_init_ids,
    )
    model.config.bos_token_id = config.bos_token_id
    model.config.eos_token_id = config.eos_token_id
    model.config.pad_token_id = config.pad_token_id
    model.generation_config.bos_token_id = config.bos_token_id
    model.generation_config.eos_token_id = config.eos_token_id
    model.generation_config.pad_token_id = config.pad_token_id

    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    save_trie_tokenizer(
        tokenizer,
        rwkv_tokenizer_json,
        source_special_entries,
        missing_entries,
        args.output,
        chat_template,
    )

    # Keep the dynamic HF architecture/tokenizer implementations alongside the
    # weights so AutoModel/AutoTokenizer can reload with trust_remote_code=True.
    for filename in ("configuration_rwkv7.py", "modeling_rwkv7.py", "tokenization_rwkv7.py"):
        shutil.copy2(download_file(args.rwkv_reference_model, filename), args.output / filename)

    print(f"Saved HF RWKV7 checkpoint: {args.output}")
    print(f"RWKV base vocab preserved: {base_vocab_size} IDs")
    print(f"Final tokenizer/model vocab size: {len(tokenizer)}")
    print(f"K2 Horizon source: {k2_checkpoint}")
    print("Added K2 Horizon special tokens:")
    for item in missing_entries:
        content = item["content"]
        token_id = tokenizer.convert_tokens_to_ids(content)
        if token_id >= base_vocab_size:
            print(f"  {content!r}: {token_id}")
    print(f"EOS id: {model.config.eos_token_id}; PAD id: {model.config.pad_token_id}")


if __name__ == "__main__":
    main()