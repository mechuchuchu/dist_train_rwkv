"""Trainable rows for tokens a converted checkpoint appends to the RWKV vocabulary.

The RWKV World tokenizer has 65,536 ids. A checkpoint converted to another chat
format (for example the K2 Horizon variant) appends special tokens after them and
grows the embedding and the LM head by one row each. Those rows start as an
average of existing rows and are exactly what SFT has to teach, so they are
trained alongside the LoRA.

Only the appended rows are trained, as an additive delta that starts at zero:

    embedding(id)   = base_embedding(id)   + embedding_delta[id - 65536]
    logit(id)       = base_logit(id)       + hidden @ head_delta[id - 65536]

for ids >= 65536, and unchanged below. The base weights stay frozen, so a worker
needs nothing but the pinned checkpoint, and the coordinator's state is two small
float32 matrices of shape (new tokens, hidden size) next to the LoRA tensors.
Training the whole embedding and head instead would move two
(65,536 + new) x hidden matrices every round (over 1 GB for the 1.5B model).

The delta is applied with forward hooks on `get_input_embeddings()` and
`get_output_embeddings()`, so it does not depend on the model's internals.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BASE_VOCAB_SIZE = 65536

# Tensor names in theta. They match the parameter names of `NewTokenRows` once it
# is attached to the model as `new_token_rows`.
MODULE_NAME = "new_token_rows"
EMBEDDING_DELTA = f"{MODULE_NAME}.embedding_delta"
HEAD_DELTA = f"{MODULE_NAME}.head_delta"
NEW_TOKEN_TENSORS = (EMBEDDING_DELTA, HEAD_DELTA)


def new_token_count(vocab_size: int) -> int:
    """How many ids a model of this vocabulary size has beyond the RWKV World tokenizer."""
    vocab_size = int(vocab_size)
    if vocab_size < BASE_VOCAB_SIZE:
        raise ValueError(f"vocab_size {vocab_size} is below the RWKV World vocabulary of {BASE_VOCAB_SIZE}")
    return vocab_size - BASE_VOCAB_SIZE


def new_token_shapes(count: int, hidden_size: int) -> dict[str, tuple[int, int]]:
    """Tensor name -> shape of the trainable rows; empty when the vocabulary was not extended."""
    if count <= 0:
        return {}
    return {EMBEDDING_DELTA: (count, hidden_size), HEAD_DELTA: (count, hidden_size)}


class NewTokenRows(torch.nn.Module):
    """The two delta matrices, plus a count of how often the model used them."""

    def __init__(self, count: int, hidden_size: int, device: torch.device | str = "cpu"):
        super().__init__()
        if count <= 0:
            raise ValueError("there are no new token rows to train")
        self.count = int(count)
        self.hidden_size = int(hidden_size)
        self.embedding_delta = torch.nn.Parameter(
            torch.zeros(count, hidden_size, dtype=torch.float32, device=device))
        self.head_delta = torch.nn.Parameter(
            torch.zeros(count, hidden_size, dtype=torch.float32, device=device))
        self.embedding_calls = 0
        self.head_calls = 0

    def reset_calls(self) -> None:
        self.embedding_calls = 0
        self.head_calls = 0

    def require_used(self) -> None:
        """Raise unless the last forward went through both hooks.

        A model that reads `lm_head.weight` itself (a fused cross-entropy, say) would
        skip the head hook; its rows would then get no gradient and, because the worker
        reports a missing gradient as zero, would silently never train.
        """
        if self.embedding_calls == 0 or self.head_calls == 0:
            raise RuntimeError(
                "the model did not call its input embedding and LM head modules during the forward "
                f"pass (embedding {self.embedding_calls}x, head {self.head_calls}x), so the new "
                "token rows would not be trained"
            )

    def add_to_embedding(self, ids: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        self.embedding_calls += 1
        # Row 0 of the padded table is zero: every id below the new ones lands on it.
        padded = torch.cat([self.embedding_delta.new_zeros(1, self.hidden_size), self.embedding_delta], dim=0)
        index = (ids - (BASE_VOCAB_SIZE - 1)).clamp(min=0)
        extra = F.embedding(index, padded)
        # Sum in float32 and round once; for the old ids this returns `output` unchanged.
        return (output.float() + extra).to(output.dtype)

    def add_to_logits(self, hidden: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        self.head_calls += 1
        if output.shape[-1] != BASE_VOCAB_SIZE + self.count:
            raise RuntimeError(
                f"the LM head has {output.shape[-1]} outputs but the new token rows cover ids "
                f"{BASE_VOCAB_SIZE}..{BASE_VOCAB_SIZE + self.count - 1}"
            )
        extra = F.linear(hidden.float(), self.head_delta)
        # float32 logits: a bfloat16 sum would round away updates smaller than its resolution.
        return torch.cat(
            [output[..., :BASE_VOCAB_SIZE].float(), output[..., BASE_VOCAB_SIZE:].float() + extra], dim=-1)


def _first_input(args: tuple, kwargs: dict) -> torch.Tensor:
    if args:
        return args[0]
    return kwargs["input"]


def attach_new_token_rows(model: torch.nn.Module, *, hf_model: torch.nn.Module | None = None) -> NewTokenRows | None:
    """Add trainable rows for every id past the RWKV World vocabulary; None if there are none.

    `model` receives the parameters (as `new_token_rows`, so they appear in its
    `named_parameters()`); `hf_model` is the module whose input embedding and LM head
    are hooked, `model` itself unless a wrapper such as PEFT is around it.
    """
    hf_model = hf_model if hf_model is not None else model
    embedding = hf_model.get_input_embeddings()
    head = hf_model.get_output_embeddings()
    if embedding is None or head is None:
        raise RuntimeError("the model exposes no input embedding or LM head to extend")
    vocab_size, hidden_size = embedding.weight.shape
    if head.weight.shape != (vocab_size, hidden_size):
        raise RuntimeError(
            f"embedding {tuple(embedding.weight.shape)} and LM head {tuple(head.weight.shape)} "
            "do not describe the same vocabulary"
        )
    if vocab_size <= BASE_VOCAB_SIZE:  # a plain RWKV7 (or a stand-in smaller than one): nothing to add
        return None
    rows = NewTokenRows(vocab_size - BASE_VOCAB_SIZE, hidden_size, device=embedding.weight.device)
    model.add_module(MODULE_NAME, rows)
    embedding.register_forward_hook(
        lambda module, args, kwargs, output: rows.add_to_embedding(_first_input(args, kwargs), output),
        with_kwargs=True,
    )
    head.register_forward_hook(
        lambda module, args, kwargs, output: rows.add_to_logits(_first_input(args, kwargs), output),
        with_kwargs=True,
    )
    return rows
