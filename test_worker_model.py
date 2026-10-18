"""The worker's model recipe, run for real on a tiny transformers model.

test_g1j_dist_worker.py replaces the model load and the gradient step to exercise the
worker's control flow. These tests run the real `load_worker_model`, `compute_unit`
and `apply_snapshot` instead, with a tiny Llama standing in for the RWKV checkpoint and
the CUDA device mapped to the CPU, to check what a worker does with the extra
embedding and LM-head rows of a converted checkpoint.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rwkv_dist_client import WORKER_BUNDLE_FILES
from rwkv_dist_protocol import decode_tensors, encode_tensors, tensor_map_sha256, unit_digest
from rwkv_new_tokens import BASE_VOCAB_SIZE, EMBEDDING_DELTA, HEAD_DELTA, NEW_TOKEN_TENSORS, attach_new_token_rows

ROOT = Path(__file__).parent
EXTRA = 3
UNIT_IDS = [5, BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + 2, 7, BASE_VOCAB_SIZE + 1, 9]


class BundleTests(unittest.TestCase):
    def test_the_bundle_is_exactly_the_modules_a_worker_imports(self):
        """A module missing from the bundle could change without workers and coordinator noticing."""
        local = {path.stem for path in ROOT.glob("*.py") if not path.name.startswith("test_")}
        seen: set[str] = set()
        queue = ["g1j_dist_worker"]
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            for node in ast.walk(ast.parse((ROOT / f"{name}.py").read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    modules = [node.module]
                else:
                    continue
                queue.extend(module.split(".")[0] for module in modules if module.split(".")[0] in local)
        self.assertEqual({f"{name}.py" for name in seen}, set(WORKER_BUNDLE_FILES))


def load_real_worker():
    spec = importlib.util.spec_from_file_location("g1j_dist_worker_with_real_libraries", ROOT / "g1j_dist_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkerModelTests(unittest.TestCase):
    def setUp(self):
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("peft/transformers are not installed")
        self.worker = load_real_worker()

    def tiny_llama(self, extra: int):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(6)
        config = LlamaConfig(
            vocab_size=BASE_VOCAB_SIZE + extra, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, tie_word_embeddings=False,
        )
        return LlamaForCausalLM(config).float()

    def load(self, extra: int):
        with patch.object(self.worker.AutoModelForCausalLM, "from_pretrained",
                          return_value=self.tiny_llama(extra)), \
                patch.object(self.worker, "TARGET_MODULES", ["q_proj", "v_proj"]):
            return self.worker.load_worker_model(Path("unused"))

    @contextlib.contextmanager
    def cpu_for_cuda(self):
        real = torch.tensor

        def tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda:0":
                kwargs = {**kwargs, "device": "cpu"}
            return real(*args, **kwargs)

        with patch.object(torch, "tensor", tensor):
            yield

    def unit(self) -> dict:
        unit = {"unit_id": "u0", "input_ids": UNIT_IDS, "labels": [-100] + UNIT_IDS[1:],
                "loss_tokens": len(UNIT_IDS) - 1, "total_tokens": len(UNIT_IDS)}
        unit["unit_sha256"] = unit_digest(unit)
        return unit

    def test_the_loaded_model_trains_the_lora_and_the_new_rows(self):
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        self.assertTrue(set(NEW_TOKEN_TENSORS) <= set(params))
        self.assertTrue(all("lora_" in name for name in set(params) - set(NEW_TOKEN_TENSORS)))
        for name in NEW_TOKEN_TENSORS:
            self.assertEqual(tuple(params[name].shape), (EXTRA, 16))
            self.assertEqual(params[name].dtype, torch.float32)
            self.assertEqual(params[name].abs().sum().item(), 0)

    def test_a_plain_vocabulary_trains_the_lora_alone(self):
        model = self.load(0)
        params = self.worker.trainable_parameters(model)
        self.assertFalse(set(NEW_TOKEN_TENSORS) & set(params))
        self.assertFalse(hasattr(model, "new_token_rows"))
        self.assertTrue(all("lora_" in name for name in params))

    def test_a_unit_returns_a_gradient_for_every_trainable_tensor_including_the_rows(self):
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        with self.cpu_for_cuda():
            payload, loss, loss_tokens, grad_norm = self.worker.compute_unit(model, params, self.unit())
        gradients = decode_tensors(payload)
        self.assertEqual(set(gradients), set(params))
        self.assertEqual(loss_tokens, 5)
        self.assertGreater(grad_norm, 0)
        for name in NEW_TOKEN_TENSORS:
            self.assertEqual(tuple(gradients[name].shape), (EXTRA, 16))
            self.assertGreater(gradients[name].abs().sum().item(), 0, name)
        # only the new tokens that appear in the input reach the embedding rows
        embedding_rows = gradients[EMBEDDING_DELTA].abs().sum(dim=1)
        self.assertTrue((embedding_rows > 0).all())

    def test_the_counters_are_reset_and_checked_around_each_forward_pass(self):
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        rows = model.new_token_rows
        with self.cpu_for_cuda(), \
                patch.object(rows, "reset_calls", wraps=rows.reset_calls) as reset, \
                patch.object(rows, "require_used", wraps=rows.require_used) as used:
            self.worker.compute_unit(model, params, self.unit())
        reset.assert_called_once_with()
        used.assert_called_once_with()

    def test_a_forward_that_skips_the_head_module_stops_the_worker(self):
        """Otherwise the missing gradient would be sent as zeros and the head rows would never train."""
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        hf_model = model.get_base_model()
        real_forward = hf_model.forward

        def forward_without_the_head(*args, **kwargs):
            hidden = hf_model.model(input_ids=kwargs["input_ids"]).last_hidden_state
            return type("Output", (), {"logits": hidden @ hf_model.lm_head.weight.T})()

        with self.cpu_for_cuda():
            self.worker.compute_unit(model, params, self.unit())  # a normal unit first
            with patch.object(hf_model, "forward", forward_without_the_head):
                with self.assertRaisesRegex(RuntimeError, "new token rows would not be trained"):
                    self.worker.compute_unit(model, params, self.unit())
            self.worker.compute_unit(model, params, self.unit())  # and a normal one again
        self.assertIs(hf_model.forward.__func__, real_forward.__func__)

    def test_a_snapshot_with_the_rows_is_applied_to_them(self):
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        generator = torch.Generator().manual_seed(1)
        snapshot = {name: torch.randn(tuple(p.shape), generator=generator) for name, p in params.items()}
        applied = self.worker.apply_snapshot(model, encode_tensors(snapshot), tensor_map_sha256(snapshot))
        for name in NEW_TOKEN_TENSORS:
            self.assertTrue(torch.equal(applied[name].detach(), snapshot[name]), name)
        self.assertTrue(torch.equal(model.new_token_rows.head_delta.detach(), snapshot[HEAD_DELTA]))

    def test_a_snapshot_from_a_coordinator_without_the_rows_is_refused(self):
        model = self.load(EXTRA)
        params = self.worker.trainable_parameters(model)
        snapshot = {name: torch.zeros(tuple(p.shape)) for name, p in params.items() if name not in NEW_TOKEN_TENSORS}
        with self.assertRaisesRegex(RuntimeError, "tensor schema mismatch"):
            self.worker.apply_snapshot(model, encode_tensors(snapshot), tensor_map_sha256(snapshot))

    def test_a_snapshot_with_rows_is_refused_by_a_plain_model(self):
        model = self.load(0)
        params = self.worker.trainable_parameters(model)
        snapshot = {name: torch.zeros(tuple(p.shape)) for name, p in params.items()}
        snapshot[EMBEDDING_DELTA] = torch.zeros(EXTRA, 16)
        snapshot[HEAD_DELTA] = torch.zeros(EXTRA, 16)
        with self.assertRaisesRegex(RuntimeError, "tensor schema mismatch"):
            self.worker.apply_snapshot(model, encode_tensors(snapshot), tensor_map_sha256(snapshot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
