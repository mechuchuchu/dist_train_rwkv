"""Trainable embedding and LM-head rows for tokens past the RWKV World vocabulary.

A checkpoint converted to another chat format (the K2 Horizon variant, say) appends
special tokens after id 65535. Their rows are trained along with the LoRA, as zero
-initialised additive deltas over the frozen base rows. The test that matters most
is the first one: training the delta must give the same gradients as training the
rows themselves.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import g1j_dist_coordinator as coordinator_module
import test_g1j_dist_coordinator as coordinator_tests
from g1j_dist_coordinator import load_initial_theta, read_new_token_count
from rwkv_dist_protocol import (
    is_saved_theta_name,
    saved_name_from_theta,
    tensor_map_sha256,
    theta_name_from_saved,
)
from rwkv_new_tokens import (
    BASE_VOCAB_SIZE,
    EMBEDDING_DELTA,
    HEAD_DELTA,
    NEW_TOKEN_TENSORS,
    NewTokenRows,
    attach_new_token_rows,
    new_token_count,
    new_token_shapes,
)
from test_g1j_dist_coordinator_cli import ReturningServer, write_fixtures, write_initial_adapter

HIDDEN = 6
EXTRA = 3


class TinyLM(torch.nn.Module):
    """embedding -> one mixing layer -> LM head, with the HF accessors the hooks use."""

    def __init__(self, extra: int = EXTRA, hidden: int = HIDDEN, bypass_head: bool = False):
        super().__init__()
        self.emb = torch.nn.Embedding(BASE_VOCAB_SIZE + extra, hidden)
        self.mix = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, BASE_VOCAB_SIZE + extra, bias=False)
        self.bypass_head = bypass_head

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head

    def forward(self, ids):
        hidden = torch.tanh(self.mix(self.emb(ids)))
        if self.bypass_head:  # reads the weight itself, the way a fused cross-entropy would
            return hidden @ self.head.weight.T
        return self.head(hidden)


def freeze(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def next_token_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1),
                           reduction="sum")


IDS = torch.tensor([[5, BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + 2, 7, BASE_VOCAB_SIZE + 1, 9]])


class VocabularyArithmeticTests(unittest.TestCase):
    def test_the_rows_are_the_ids_past_the_rwkv_world_vocabulary(self):
        self.assertEqual(BASE_VOCAB_SIZE, 65536)
        self.assertEqual(new_token_count(65536), 0)
        self.assertEqual(new_token_count(65536 + 200), 200)

    def test_a_smaller_vocabulary_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "below the RWKV World vocabulary"):
            new_token_count(65535)

    def test_the_tensors_are_two_count_by_hidden_matrices(self):
        self.assertEqual(new_token_shapes(5, 2048), {EMBEDDING_DELTA: (5, 2048), HEAD_DELTA: (5, 2048)})
        self.assertEqual(new_token_shapes(0, 2048), {})

    def test_the_names_are_the_parameter_names_of_the_attached_module(self):
        self.assertEqual(NEW_TOKEN_TENSORS, ("new_token_rows.embedding_delta", "new_token_rows.head_delta"))
        names = {name for name, _ in NewTokenRows(2, 4).named_parameters()}
        self.assertEqual({f"new_token_rows.{name}" for name in names}, set(NEW_TOKEN_TENSORS))


class EquivalenceTests(unittest.TestCase):
    """Training the deltas is training the rows themselves."""

    def setUp(self):
        torch.manual_seed(0)
        self.base = TinyLM()
        self.reference = copy.deepcopy(self.base)
        freeze(self.base)
        self.rows = attach_new_token_rows(self.base)
        with torch.no_grad():
            self.rows.embedding_delta.normal_()
            self.rows.head_delta.normal_()

    def reference_with_rows_trained(self):
        with torch.no_grad():
            self.reference.emb.weight[BASE_VOCAB_SIZE:] += self.rows.embedding_delta
            self.reference.head.weight[BASE_VOCAB_SIZE:] += self.rows.head_delta
        self.reference.emb.weight.requires_grad_(True)
        self.reference.head.weight.requires_grad_(True)
        return self.reference

    def test_logits_equal_those_of_a_model_whose_rows_were_changed(self):
        reference = self.reference_with_rows_trained()
        torch.testing.assert_close(self.base(IDS), reference(IDS), rtol=1e-5, atol=1e-5)

    def test_gradients_equal_those_of_the_rows(self):
        loss = next_token_loss(self.base(IDS), IDS)
        loss.backward()
        reference = self.reference_with_rows_trained()
        reference_loss = next_token_loss(reference(IDS), IDS)
        reference_loss.backward()
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(self.rows.embedding_delta.grad,
                                   reference.emb.weight.grad[BASE_VOCAB_SIZE:], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(self.rows.head_delta.grad,
                                   reference.head.weight.grad[BASE_VOCAB_SIZE:], rtol=1e-4, atol=1e-6)
        self.assertGreater(self.rows.embedding_delta.grad.abs().sum().item(), 0)
        self.assertGreater(self.rows.head_delta.grad.abs().sum().item(), 0)

    def test_the_base_weights_receive_no_gradient(self):
        next_token_loss(self.base(IDS), IDS).backward()
        for name, parameter in self.base.named_parameters():
            if name.startswith("new_token_rows."):
                continue
            self.assertIsNone(parameter.grad, name)

    def test_an_embedding_row_gets_gradient_only_when_its_token_is_in_the_input(self):
        ids = torch.tensor([[5, BASE_VOCAB_SIZE + 1, 7, 9]])
        next_token_loss(self.base(ids), ids).backward()
        grad = self.rows.embedding_delta.grad
        self.assertGreater(grad[1].abs().sum().item(), 0)
        self.assertEqual(grad[0].abs().sum().item(), 0)
        self.assertEqual(grad[2].abs().sum().item(), 0)
        # ...while every head row is pushed down or up by every position's softmax
        self.assertTrue((self.rows.head_delta.grad.abs().sum(dim=1) > 0).all())


class ZeroDeltaTests(unittest.TestCase):
    """At the start of training the model is exactly the converted checkpoint."""

    def test_zero_deltas_change_nothing_for_any_id(self):
        torch.manual_seed(1)
        plain = TinyLM()
        hooked = copy.deepcopy(plain)
        freeze(hooked)
        attach_new_token_rows(hooked)
        self.assertTrue(torch.equal(hooked(IDS), plain(IDS)))

    def test_in_bfloat16_the_embedding_stays_bfloat16_and_unchanged_at_zero(self):
        torch.manual_seed(2)
        plain = TinyLM().to(torch.bfloat16)
        hooked = copy.deepcopy(plain)
        freeze(hooked)
        attach_new_token_rows(hooked)
        self.assertEqual(hooked.emb(IDS).dtype, torch.bfloat16)
        self.assertTrue(torch.equal(hooked.emb(IDS), plain.emb(IDS)))
        self.assertTrue(torch.equal(hooked(IDS), plain(IDS).float()))

    def test_the_head_delta_is_applied_in_float32_so_small_updates_are_not_rounded_away(self):
        torch.manual_seed(3)
        model = TinyLM().to(torch.bfloat16)
        freeze(model)
        rows = attach_new_token_rows(model)
        before = model(IDS).clone()
        with torch.no_grad():
            rows.head_delta[0].fill_(1e-4)  # far below bfloat16's resolution at these magnitudes
        after = model(IDS)
        self.assertEqual(after.dtype, torch.float32)
        self.assertFalse(torch.equal(after[..., BASE_VOCAB_SIZE], before[..., BASE_VOCAB_SIZE]))
        self.assertTrue(torch.equal(after[..., :BASE_VOCAB_SIZE], before[..., :BASE_VOCAB_SIZE]))
        self.assertTrue(torch.equal(after[..., BASE_VOCAB_SIZE + 1:], before[..., BASE_VOCAB_SIZE + 1:]))

    def test_the_embedding_sum_is_rounded_once_not_twice(self):
        """bfloat16(1.0078125 + 0.0039) is 1.0078125; rounding the delta first (to 0.00390625) would give 1.015625."""
        rows = NewTokenRows(1, 2)
        with torch.no_grad():
            rows.embedding_delta.fill_(0.0039)
        output = torch.full((1, 1, 2), 1.0078125, dtype=torch.bfloat16)
        result = rows.add_to_embedding(torch.tensor([[BASE_VOCAB_SIZE]]), output)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result[0, 0, 0].item(), 1.0078125)
        untouched = rows.add_to_embedding(torch.tensor([[7]]), output)
        self.assertTrue(torch.equal(untouched, output))

    def test_old_ids_keep_their_exact_embedding_when_the_rows_are_nonzero(self):
        torch.manual_seed(4)
        plain = TinyLM()
        hooked = copy.deepcopy(plain)
        freeze(hooked)
        rows = attach_new_token_rows(hooked)
        with torch.no_grad():
            rows.embedding_delta.normal_()
        old_ids = torch.tensor([[0, 1, 65535]])
        self.assertTrue(torch.equal(hooked.emb(old_ids), plain.emb(old_ids)))
        new_id = torch.tensor([[BASE_VOCAB_SIZE + 1]])
        expected = plain.emb(new_id) + rows.embedding_delta[1]
        torch.testing.assert_close(hooked.emb(new_id), expected)

    def test_the_embedding_may_be_called_with_a_keyword(self):
        model = TinyLM()
        freeze(model)
        rows = attach_new_token_rows(model)
        with torch.no_grad():
            rows.embedding_delta.fill_(1.0)
        positional = model.emb(IDS)
        keyword = model.emb(input=IDS)
        self.assertTrue(torch.equal(positional, keyword))


class AttachTests(unittest.TestCase):
    def test_a_plain_rwkv_vocabulary_gets_no_rows(self):
        model = TinyLM(extra=0)
        freeze(model)
        self.assertIsNone(attach_new_token_rows(model))
        self.assertFalse(hasattr(model, "new_token_rows"))
        self.assertEqual([p for p in model.parameters() if p.requires_grad], [])

    def test_a_model_smaller_than_the_rwkv_world_vocabulary_gets_no_rows(self):
        model = torch.nn.Sequential()
        model.emb = torch.nn.Embedding(8, 4)
        model.head = torch.nn.Linear(4, 8, bias=False)
        model.get_input_embeddings = lambda: model.emb
        model.get_output_embeddings = lambda: model.head
        self.assertIsNone(attach_new_token_rows(model))

    def test_the_rows_are_the_only_trainable_parameters_and_are_float32_zeros(self):
        model = TinyLM().to(torch.bfloat16)
        freeze(model)
        attach_new_token_rows(model)
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(set(trainable), set(NEW_TOKEN_TENSORS))
        for name, parameter in trainable.items():
            self.assertEqual(tuple(parameter.shape), (EXTRA, HIDDEN), name)
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertEqual(parameter.abs().sum().item(), 0, name)

    def test_parameters_go_on_the_wrapper_and_hooks_on_the_wrapped_model(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def get_base_model(self):
                return self.inner

            def forward(self, ids):
                return self.inner(ids)

        wrapper = Wrapper(TinyLM())
        freeze(wrapper)
        rows = attach_new_token_rows(wrapper, hf_model=wrapper.get_base_model())
        self.assertEqual({n for n, p in wrapper.named_parameters() if p.requires_grad}, set(NEW_TOKEN_TENSORS))
        self.assertIs(wrapper.new_token_rows, rows)
        wrapper(IDS)
        rows.require_used()

    def test_an_embedding_and_head_that_disagree_are_refused(self):
        model = TinyLM()
        model.head = torch.nn.Linear(HIDDEN, BASE_VOCAB_SIZE + EXTRA + 1, bias=False)
        with self.assertRaisesRegex(RuntimeError, "do not describe the same vocabulary"):
            attach_new_token_rows(model)

    def test_a_model_without_accessors_is_refused(self):
        class Bare(torch.nn.Module):
            def get_input_embeddings(self):
                return None

            def get_output_embeddings(self):
                return None

        with self.assertRaisesRegex(RuntimeError, "exposes no input embedding"):
            attach_new_token_rows(Bare())

    def test_a_head_of_the_wrong_width_is_refused_at_the_forward_pass(self):
        model = TinyLM()
        freeze(model)
        rows = attach_new_token_rows(model)
        with self.assertRaisesRegex(RuntimeError, "LM head has"):
            rows.add_to_logits(torch.zeros(1, 2, HIDDEN), torch.zeros(1, 2, BASE_VOCAB_SIZE))

    def test_no_rows_to_train_is_an_error_for_the_module_itself(self):
        with self.assertRaises(ValueError):
            NewTokenRows(0, 8)


class ForwardGuardTests(unittest.TestCase):
    def test_a_forward_that_goes_through_both_modules_passes(self):
        model = TinyLM()
        freeze(model)
        rows = attach_new_token_rows(model)
        rows.reset_calls()
        model(IDS)
        rows.require_used()
        self.assertEqual((rows.embedding_calls, rows.head_calls), (1, 1))

    def test_a_model_that_reads_the_head_weight_itself_is_caught(self):
        model = TinyLM(bypass_head=True)
        freeze(model)
        rows = attach_new_token_rows(model)
        rows.reset_calls()
        model(IDS)
        with self.assertRaisesRegex(RuntimeError, "new token rows would not be trained"):
            rows.require_used()

    def test_the_counters_reset_between_units(self):
        model = TinyLM()
        freeze(model)
        rows = attach_new_token_rows(model)
        model(IDS)
        model(IDS)
        self.assertEqual((rows.embedding_calls, rows.head_calls), (2, 2))
        rows.reset_calls()
        self.assertEqual((rows.embedding_calls, rows.head_calls), (0, 0))
        with self.assertRaises(RuntimeError):
            rows.require_used()


class SavedNameTests(unittest.TestCase):
    def test_new_token_tensors_keep_their_names_in_adapter_files(self):
        for name in NEW_TOKEN_TENSORS:
            self.assertTrue(is_saved_theta_name(name))
            self.assertEqual(saved_name_from_theta(name), name)
            self.assertEqual(theta_name_from_saved(name), name)

    def test_lora_names_still_convert(self):
        theta = "base_model.model.model.layers.0.att.key.lora_A.default.weight"
        saved = "base_model.model.model.layers.0.att.key.lora_A.weight"
        self.assertEqual(saved_name_from_theta(theta), saved)
        self.assertEqual(theta_name_from_saved(saved), theta)
        self.assertTrue(is_saved_theta_name(saved))

    def test_other_tensors_are_not_part_of_theta(self):
        for name in ("base_model.model.lm_head.weight", "new_token_rows.other", "embedding_delta"):
            self.assertFalse(is_saved_theta_name(name), name)


class InitialThetaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "adapter"
        write_initial_adapter(self.dir, 2, 64)

    def test_the_rows_start_at_zero_with_the_models_hidden_size(self):
        theta = load_initial_theta(self.dir, model_shape=(2, 64), new_tokens=5)
        self.assertEqual(len(theta), 16 + 2)
        for name in NEW_TOKEN_TENSORS:
            self.assertEqual(tuple(theta[name].shape), (5, 64))
            self.assertEqual(theta[name].dtype, torch.float32)
            self.assertEqual(theta[name].abs().sum().item(), 0)

    def test_without_new_tokens_theta_is_the_lora_alone(self):
        self.assertEqual(len(load_initial_theta(self.dir, model_shape=(2, 64))), 16)
        self.assertEqual(len(load_initial_theta(self.dir, model_shape=(2, 64), new_tokens=0)), 16)

    def test_rows_in_the_adapter_file_do_not_override_the_zero_start(self):
        path = self.dir / "adapter_model.safetensors"
        save_file({**load_file(str(path)),
                   EMBEDDING_DELTA: torch.ones(5, 64), HEAD_DELTA: torch.ones(5, 64)}, str(path))
        theta = load_initial_theta(self.dir, model_shape=(2, 64), new_tokens=5)
        self.assertEqual(theta[EMBEDDING_DELTA].abs().sum().item(), 0)
        self.assertEqual(len(load_initial_theta(self.dir, model_shape=(2, 64))), 16)


class ReadNewTokenCountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def config(self, content):
        (self.dir / "config.json").write_text(json.dumps(content))

    def test_it_is_the_vocabulary_beyond_65536(self):
        self.config({"vocab_size": 65536})
        self.assertEqual(read_new_token_count(self.dir), 0)
        self.config({"vocab_size": 65536 + 187})
        self.assertEqual(read_new_token_count(self.dir), 187)

    def test_no_config_or_no_vocab_size_means_none(self):
        self.assertEqual(read_new_token_count(self.dir), 0)
        self.config({"hidden_size": 2048})
        self.assertEqual(read_new_token_count(self.dir), 0)
        self.config([])
        self.assertEqual(read_new_token_count(self.dir), 0)

    def test_a_vocabulary_smaller_than_rwkv_world_is_refused(self):
        self.config({"vocab_size": 50000})
        with self.assertRaisesRegex(ValueError, "below the RWKV World vocabulary"):
            read_new_token_count(self.dir)

    def test_a_vocab_size_that_is_not_a_number_is_refused(self):
        self.config({"vocab_size": "many"})
        with self.assertRaises(ValueError):
            read_new_token_count(self.dir)


class CoordinatorStartTests(unittest.TestCase):
    """main() on a snapshot whose vocabulary was extended."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixtures = write_fixtures(self.root, layers=2, hidden=64)

    def start(self, vocab_size: int | None) -> tuple[str, Path]:
        config_path = self.fixtures["model"] / "config.json"
        config = json.loads(config_path.read_text())
        if vocab_size is not None:
            config["vocab_size"] = vocab_size
        config_path.write_text(json.dumps(config))
        run_dir = self.root / f"run-{vocab_size}"
        argv = [
            "g1j_dist_coordinator.py", "--units", str(self.fixtures["units"]),
            "--model-dir", str(self.fixtures["model"]),
            "--initial-adapter-dir", str(self.fixtures["adapter"]),
            "--run-dir", str(run_dir), "--token", "protocol-token", "--port", "0",
            "--units-per-step", "2", "--min-units-per-step", "2",
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch.object(coordinator_module, "ThreadingHTTPServer", ReturningServer), \
                contextlib.redirect_stdout(output):
            coordinator_module.main()
        return output.getvalue(), run_dir

    def test_an_extended_vocabulary_adds_zero_rows_to_the_first_theta_and_says_so(self):
        text, run_dir = self.start(65536 + 4)
        self.assertIn("4 token ids beyond the RWKV World vocabulary (ids 65536..65539)", text)
        theta = load_file(str(run_dir / "theta.safetensors"))
        self.assertEqual(len(theta), 16 + 2)
        for name in NEW_TOKEN_TENSORS:
            self.assertEqual(tuple(theta[name].shape), (4, 64))
            self.assertEqual(theta[name].abs().sum().item(), 0)
        saved = load_file(str(run_dir / "adapter_model.safetensors"))
        self.assertTrue(set(NEW_TOKEN_TENSORS) <= set(saved))
        self.assertIn("base_model.model.model.layers.0.att.key.lora_A.weight", saved)

    def test_a_plain_vocabulary_is_unchanged(self):
        text, run_dir = self.start(65536)
        self.assertNotIn("beyond the RWKV World vocabulary", text)
        theta = load_file(str(run_dir / "theta.safetensors"))
        self.assertEqual(len(theta), 16)
        self.assertFalse(set(NEW_TOKEN_TENSORS) & set(theta))

    def test_a_snapshot_without_vocab_size_is_unchanged(self):
        _text, run_dir = self.start(None)
        self.assertEqual(len(load_file(str(run_dir / "theta.safetensors"))), 16)


def theta_with_rows():
    """The names the resume tests train, plus both new-token tensors."""
    return (*coordinator_tests.THETA_NAMES, *NEW_TOKEN_TENSORS)


class RowsSurviveResumeMixin:
    """Run another module's resume tests with the new-token tensors in theta.

    The coordinator's optimizer, checkpoints, adapter files, Hub pushes and Hub
    replay take theta as it comes; this checks that they do so for these tensors too.
    """

    def setUp(self):
        patcher = patch.object(coordinator_tests, "THETA_NAMES", theta_with_rows())
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()


class LocalResumeWithRows(RowsSurviveResumeMixin, coordinator_tests.CoordinatorResumeTests):
    pass


class HubResumeWithRows(RowsSurviveResumeMixin, coordinator_tests.CoordinatorHubResumeTests):
    def test_the_pushed_adapter_carries_the_rows_next_to_the_lora(self):
        hub, origin = self.push_first_rounds(rounds=1, repo_private=True, policy="auto")
        pushed = load_file(str(hub.root / "checkpoints" / "run-resume" / "theta-000001"
                               / "adapter_model.safetensors"))
        self.assertTrue(set(NEW_TOKEN_TENSORS) <= set(pushed))
        self.assertEqual({name for name in pushed if name not in NEW_TOKEN_TENSORS},
                         {"m.layers.0.att.receptance.lora_A.weight",
                          "m.layers.0.att.receptance.lora_B.weight"})
        expected = {name: tensor for name, tensor in origin.coordinator.theta.items()}
        loaded = self.load_from(hub)
        self.assertEqual(tensor_map_sha256(loaded.theta), tensor_map_sha256(expected))
        self.assertEqual(set(loaded.theta), set(theta_with_rows()))


class HubRecoveryWithRows(RowsSurviveResumeMixin, coordinator_tests.HubPushRecoveryTests):
    pass


class RealLibrariesTests(unittest.TestCase):
    """The worker's own model recipe (frozen base, checkpointing, PEFT LoRA) on a tiny transformers model."""

    def setUp(self):
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("peft/transformers are not installed")

    def tiny_llama(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(5)
        config = LlamaConfig(
            vocab_size=BASE_VOCAB_SIZE + EXTRA, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, tie_word_embeddings=False,
        )
        return LlamaForCausalLM(config).float()

    def test_rows_train_through_peft_and_gradient_checkpointing_like_real_rows(self):
        from peft import LoraConfig, get_peft_model

        plain = self.tiny_llama()
        reference = copy.deepcopy(plain)

        for parameter in plain.parameters():
            parameter.requires_grad_(False)
        plain.config.use_cache = False
        plain.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        plain.enable_input_require_grads()
        model = get_peft_model(plain, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                                 target_modules=["q_proj", "v_proj"]))
        rows = attach_new_token_rows(model, hf_model=model.get_base_model())
        model.train()
        with torch.no_grad():
            rows.embedding_delta.normal_(std=0.5)
            rows.head_delta.normal_(std=0.5)

        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(set(NEW_TOKEN_TENSORS) <= trainable)
        self.assertTrue(all("lora_" in name for name in trainable - set(NEW_TOKEN_TENSORS)))

        rows.reset_calls()
        loss = next_token_loss(model(input_ids=IDS, use_cache=False).logits, IDS)
        loss.backward()
        rows.require_used()

        # the same step on a copy whose embedding and head rows are the trainable parameters
        # (LoRA B starts at zero, so the adapted model equals its base and the copy needs no LoRA)
        with torch.no_grad():
            reference.get_input_embeddings().weight[BASE_VOCAB_SIZE:] += rows.embedding_delta
            reference.get_output_embeddings().weight[BASE_VOCAB_SIZE:] += rows.head_delta
        reference.get_input_embeddings().weight.requires_grad_(True)
        reference.get_output_embeddings().weight.requires_grad_(True)
        reference_loss = next_token_loss(reference(input_ids=IDS, use_cache=False).logits, IDS)
        reference_loss.backward()

        torch.testing.assert_close(loss, reference_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            rows.embedding_delta.grad,
            reference.get_input_embeddings().weight.grad[BASE_VOCAB_SIZE:], rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            rows.head_delta.grad,
            reference.get_output_embeddings().weight.grad[BASE_VOCAB_SIZE:], rtol=1e-3, atol=1e-5)
        self.assertGreater(rows.embedding_delta.grad.abs().sum().item(), 0)


class PeftIgnoresTheRowsTests(unittest.TestCase):
    """The rows ride in adapter_model.safetensors; a stock PEFT loader must not choke on them."""

    def test_loading_an_adapter_file_with_the_extra_tensors_works(self):
        try:
            from peft import LoraConfig, get_peft_model
            from peft.utils import set_peft_model_state_dict
        except ImportError:
            self.skipTest("peft is not installed")

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.key = torch.nn.Linear(8, 8, bias=False)

            def forward(self, x):
                return self.key(x)

        model = get_peft_model(Block(), LoraConfig(r=8, lora_alpha=16, target_modules=["key"]))
        state = {name.replace(".default", ""): torch.ones_like(p)
                 for name, p in model.named_parameters() if "lora_" in name}
        state[EMBEDDING_DELTA] = torch.zeros(3, 8)
        state[HEAD_DELTA] = torch.zeros(3, 8)
        set_peft_model_state_dict(model, state)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
