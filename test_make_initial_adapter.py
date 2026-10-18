"""make_initial_adapter.py against real PEFT and a stand-in for the RWKV base.

The base model itself is not available on CPU, but the script only needs its
module structure: 24 layers whose `att` block holds receptance/key/value/output
projections of width 2048 (the sizes behind the protocol's 192-tensor,
3,145,728-parameter schema). The stand-in has exactly that structure, with
uninitialised bf16 weights so it costs almost no memory, and is returned from
`AutoModelForCausalLM.from_pretrained` so the worker's own `load_worker_model`
runs unchanged on top of real PEFT.

These tests need transformers and peft; without them they are skipped.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from torch import nn

import test_g1j_dist_coordinator_cli as coordinator_cli
from rwkv_dist_protocol import parameter_name_to_peft_saved_name, tensor_map_sha256


def import_against_real_libraries():
    """Import the script on real peft/transformers.

    test_g1j_dist_worker caches a g1j_dist_worker built on stubs; set that
    one aside so the script binds to a worker module that has the real PEFT.
    """
    names = ("g1j_dist_worker", "make_initial_adapter")
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        import make_initial_adapter as script
        import g1j_dist_worker as worker
        return script, worker
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


try:
    from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
    from transformers.modeling_layers import GradientCheckpointingLayer
    SCRIPT, WORKER = import_against_real_libraries()
    SKIP = None
except ImportError as exc:  # transformers / peft not installed
    SCRIPT = WORKER = None
    SKIP = f"needs transformers and peft: {exc}"

HIDDEN = 2048


def projection() -> nn.Linear:
    # Uninitialised: the base weights are never read, only their shapes.
    return nn.utils.skip_init(nn.Linear, HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)


if SKIP is None:
    class StandInConfig(PretrainedConfig):
        model_type = "rwkv-structure-standin"

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.receptance, self.key = projection(), projection()
            self.value, self.output = projection(), projection()

    class FeedForward(nn.Module):
        # Also has a `key`/`value`: only `att.*` may be picked up.
        def __init__(self):
            super().__init__()
            self.key, self.value = projection(), projection()

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.att, self.ffn = Attention(), FeedForward()

    class Backbone(nn.Module):
        def __init__(self, layers: int):
            super().__init__()
            self.embeddings = nn.Embedding(8, HIDDEN, dtype=torch.bfloat16)
            self.layers = nn.ModuleList(Block() for _ in range(layers))

    class StandInModel(PreTrainedModel, GenerationMixin):
        config_class = StandInConfig
        supports_gradient_checkpointing = True

        def __init__(self, config, layers: int = 24):
            super().__init__(config)
            self.model = Backbone(layers)
            self.lm_head = nn.Linear(HIDDEN, 8, bias=False, dtype=torch.bfloat16)

        def get_input_embeddings(self):
            return self.model.embeddings

        def set_input_embeddings(self, value):
            self.model.embeddings = value

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}


class FakeLoader:
    """Stands in for AutoModelForCausalLM.from_pretrained."""

    def __init__(self, layers: int = 24, draw_random: int = 0, fail: bool = False):
        self.layers, self.draw_random, self.fail = layers, draw_random, fail
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.fail:
            raise OSError("model files are unreadable")
        if self.draw_random:
            torch.rand(self.draw_random)  # a loader that consumes the global RNG
        model = StandInModel(StandInConfig(), layers=self.layers)
        model.name_or_path = str(path)
        return model


@unittest.skipIf(SKIP is not None, SKIP or "")
class InitialAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base-model"
        self.out = self.root / "adapters" / "init"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, loader=None, seed=7, out=None, force=False):
        loader = loader or FakeLoader()
        with patch.object(WORKER.AutoModelForCausalLM, "from_pretrained", loader), \
                warnings.catch_warnings():
            # PEFT looks for the base model's config.json when saving; the stand-in has none.
            warnings.filterwarnings("ignore", message="Could not find a config file")
            record = SCRIPT.build_initial_adapter(
                self.base, out or self.out, seed=seed, force=force)
        return record, loader

    def leftovers(self):
        return sorted(p.name for p in self.out.parent.iterdir()) if self.out.parent.exists() else []

    def test_the_coordinator_loads_what_the_script_writes(self):
        record, _ = self.build()
        theta = SCRIPT.load_initial_theta(self.out)
        self.assertEqual(len(theta), 192)
        self.assertEqual(sum(t.numel() for t in theta.values()), 3_145_728)
        self.assertEqual(record["theta_hash"], tensor_map_sha256(theta))
        self.assertEqual((record["tensors"], record["parameters"]), (192, 3_145_728))
        self.assertTrue(all(t.dtype == torch.float32 for t in theta.values()))

    def test_tensor_names_are_exactly_the_ones_a_worker_trains(self):
        self.build()
        with patch.object(WORKER.AutoModelForCausalLM, "from_pretrained", FakeLoader()):
            worker_model = WORKER.load_worker_model(self.base)
        expected = {parameter_name_to_peft_saved_name(name)
                    for name in WORKER.trainable_parameters(worker_model)}
        written = set(load_file(str(self.out / "adapter_model.safetensors")))
        self.assertEqual(written, expected)
        self.assertIn("base_model.model.model.layers.23.att.output.lora_B.weight", written)
        self.assertFalse(any(".ffn." in name for name in written))

    def test_the_coordinator_starts_from_the_adapter_and_reports_the_printed_hash(self):
        record, _ = self.build()
        case = coordinator_cli.CoordinatorCommandLineTests(
            "test_fresh_start_commits_version_zero_and_reports_it")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.fixtures["adapter"] = self.out
        status, _ = case.run_main(case.root / "run")
        self.assertEqual(status["theta_hash"], record["theta_hash"])
        self.assertEqual(status["round_id"], 1)
        saved = json.loads((case.root / "run" / "adapter_config.json").read_text())
        self.assertEqual(saved["r"], 8)

    def test_a_worker_can_apply_the_initial_theta_as_its_snapshot(self):
        self.build()
        theta = SCRIPT.load_initial_theta(self.out)
        payload = WORKER.encode_tensors(theta)
        with patch.object(WORKER.AutoModelForCausalLM, "from_pretrained", FakeLoader()):
            worker_model = WORKER.load_worker_model(self.base)
        params = WORKER.apply_snapshot(worker_model, payload, tensor_map_sha256(theta))
        self.assertEqual(len(params), 192)

    def test_the_start_is_the_base_model_lora_b_zero_and_lora_a_random(self):
        self.build()
        theta = SCRIPT.load_initial_theta(self.out)
        for name, tensor in theta.items():
            if ".lora_B." in name:
                self.assertEqual(int(tensor.count_nonzero()), 0, name)
                self.assertEqual(tuple(tensor.shape), (HIDDEN, 8))
            else:
                self.assertEqual(tuple(tensor.shape), (8, HIDDEN))
                self.assertGreater(int(tensor.count_nonzero()), 0, name)
                self.assertLessEqual(float(tensor.abs().max()), HIDDEN ** -0.5 * (1 + 1e-6))
        self.assertGreater(float(theta["base_model.model.model.layers.0.att.key.lora_A.default.weight"].std()), 0.01)

    def test_config_is_rank_8_attention_only_and_names_the_base(self):
        self.build()
        config = json.loads((self.out / "adapter_config.json").read_text())
        self.assertEqual(config["peft_type"], "LORA")
        self.assertEqual((config["r"], config["lora_alpha"], config["lora_dropout"]), (8, 16, 0.0))
        self.assertEqual(sorted(config["target_modules"]),
                         ["att.key", "att.output", "att.receptance", "att.value"])
        self.assertEqual(config["base_model_name_or_path"], str(self.base))

    def test_the_worker_loader_is_used_unchanged(self):
        _, loader = self.build()
        (path, kwargs), = loader.calls
        self.assertEqual(path, str(self.base))
        self.assertIs(kwargs["dtype"], torch.bfloat16)
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["local_files_only"])

    def test_the_same_seed_gives_the_same_adapter_and_another_seed_a_different_one(self):
        first, _ = self.build(seed=11, out=self.root / "a")
        again, _ = self.build(seed=11, out=self.root / "b")
        other, _ = self.build(seed=12, out=self.root / "c")
        self.assertEqual(first["theta_hash"], again["theta_hash"])
        self.assertNotEqual(first["theta_hash"], other["theta_hash"])
        self.assertEqual(first["seed"], 11)

    def test_values_do_not_depend_on_how_the_loader_used_the_global_rng(self):
        quiet, _ = self.build(seed=5, out=self.root / "quiet")
        torch.manual_seed(123)
        noisy, _ = self.build(loader=FakeLoader(draw_random=1000), seed=5, out=self.root / "noisy")
        self.assertEqual(quiet["theta_hash"], noisy["theta_hash"])

    def test_the_record_matches_the_files(self):
        record, _ = self.build()
        on_disk = json.loads((self.out / "adapter_init.json").read_text())
        self.assertEqual(on_disk, record)
        self.assertEqual(on_disk["base_model"]["revision"], WORKER.MODEL_REVISION)
        self.assertEqual(on_disk["versions"]["peft"], SCRIPT.peft.__version__)

    def test_existing_weights_are_not_replaced_without_force(self):
        first, _ = self.build(seed=1)
        before = (self.out / "adapter_model.safetensors").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build(seed=2)
        self.assertEqual((self.out / "adapter_model.safetensors").read_bytes(), before)
        self.assertEqual(self.leftovers(), ["init"])
        replaced, _ = self.build(seed=2, force=True)
        self.assertNotEqual(replaced["theta_hash"], first["theta_hash"])
        self.assertEqual(SCRIPT.tensor_map_sha256(SCRIPT.load_initial_theta(self.out)),
                         replaced["theta_hash"])
        self.assertEqual(self.leftovers(), ["init"])

    def test_a_directory_that_only_holds_a_config_is_filled_in(self):
        self.out.mkdir(parents=True)
        (self.out / "adapter_config.json").write_text('{"note": "tracked placeholder"}')
        self.build()
        self.assertTrue((self.out / "adapter_model.safetensors").is_file())
        self.assertEqual(json.loads((self.out / "adapter_config.json").read_text())["r"], 8)
        self.assertEqual(self.leftovers(), ["init"])

    def test_a_failed_run_leaves_nothing_behind(self):
        with self.assertRaises(OSError):
            self.build(loader=FakeLoader(fail=True))
        self.assertFalse(self.out.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_model_with_a_different_layer_count_is_not_published(self):
        with self.assertRaisesRegex(ValueError, "unexpected initial adapter tensor schema"):
            self.build(loader=FakeLoader(layers=23))
        self.assertFalse(self.out.exists())
        self.assertEqual(self.leftovers(), [])

    def test_non_float32_adapter_tensors_are_refused(self):
        real = WORKER.trainable_parameters

        def bf16_params(model):
            return {name: nn.Parameter(p.detach().to(torch.bfloat16))
                    for name, p in real(model).items()}

        with patch.object(SCRIPT, "trainable_parameters", bf16_params):
            with self.assertRaisesRegex(RuntimeError, "not float32"):
                self.build()
        self.assertFalse(self.out.exists())

    def test_tensors_named_differently_from_the_workers_are_not_published(self):
        real = WORKER.trainable_parameters

        def renamed(model):
            return {name.replace(".default.", ".renamed."): p for name, p in real(model).items()}

        with patch.object(SCRIPT, "trainable_parameters", renamed):
            with self.assertRaisesRegex(RuntimeError, "does not match the worker's"):
                self.build()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_non_zero_lora_b_stops_the_build(self):
        def leaves_b_nonzero(params, seed):
            with torch.no_grad():
                for parameter in params.values():
                    parameter.fill_(0.01)

        with patch.object(SCRIPT, "initialise_lora", leaves_b_nonzero):
            with self.assertRaisesRegex(RuntimeError, "not zero"):
                self.build()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.leftovers(), [])

    def test_initialisation_zeroes_lora_b_whatever_the_model_started_with(self):
        # PEFT zeroes lora_B by default; the script must not depend on that.
        params = {"x.lora_A.default.weight": nn.Parameter(torch.ones(8, 64)),
                  "x.lora_B.default.weight": nn.Parameter(torch.ones(64, 8))}
        SCRIPT.initialise_lora(params, seed=0)
        self.assertEqual(int(params["x.lora_B.default.weight"].count_nonzero()), 0)
        a = params["x.lora_A.default.weight"].detach()
        self.assertLessEqual(float(a.abs().max()), 64 ** -0.5 * (1 + 1e-6))
        self.assertGreater(float(a.std()), 0.01)

    def test_a_trainable_tensor_outside_the_lora_schema_is_refused(self):
        params = {"base_model.model.lm_head.weight": nn.Parameter(torch.zeros(2))}
        with self.assertRaisesRegex(RuntimeError, "outside the LoRA schema"):
            SCRIPT.initialise_lora(params, seed=0)

    def test_a_non_zero_lora_b_is_caught(self):
        theta = {"x.lora_A.default.weight": torch.full((8, 4), 0.1),
                 "x.lora_B.default.weight": torch.zeros(4, 8)}
        SCRIPT.check_clean_init(theta)
        theta["x.lora_B.default.weight"][0, 0] = 1e-9
        with self.assertRaisesRegex(RuntimeError, "not zero"):
            SCRIPT.check_clean_init(theta)

    def test_a_lora_a_outside_the_init_bound_or_all_zero_is_caught(self):
        zero = {"x.lora_A.default.weight": torch.zeros(8, 4)}
        with self.assertRaisesRegex(RuntimeError, "Kaiming-uniform"):
            SCRIPT.check_clean_init(zero)
        large = {"x.lora_A.default.weight": torch.full((8, 4), 0.6)}  # bound is 0.5
        with self.assertRaisesRegex(RuntimeError, "Kaiming-uniform"):
            SCRIPT.check_clean_init(large)

    def test_names_or_shapes_that_differ_from_the_worker_are_caught(self):
        params = {"a": nn.Parameter(torch.zeros(2, 3)), "b": nn.Parameter(torch.zeros(1))}
        SCRIPT.check_matches_worker({"a": torch.zeros(2, 3), "b": torch.zeros(1)}, params)
        with self.assertRaisesRegex(RuntimeError, "missing=\\['b'\\]"):
            SCRIPT.check_matches_worker({"a": torch.zeros(2, 3)}, params)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch for a"):
            SCRIPT.check_matches_worker({"a": torch.zeros(3, 2), "b": torch.zeros(1)}, params)

    def run_main(self, *args, cuda=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                patch.object(torch.cuda, "is_available", return_value=cuda), \
                patch.object(WORKER.AutoModelForCausalLM, "from_pretrained", FakeLoader()):
            try:
                code = SCRIPT.main(list(args))
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_main_reports_the_hash_and_how_to_use_the_adapter(self):
        self.base.mkdir()
        code, out, _ = self.run_main("--base-model", str(self.base),
                                     "--output-dir", str(self.out), "--seed", "3")
        self.assertEqual(code, 0)
        record = json.loads((self.out / "adapter_init.json").read_text())
        self.assertIn(f"theta_hash {record['theta_hash']}", out)
        self.assertIn(f"--initial-adapter-dir {self.out}", out)
        self.assertIn("192 tensors", out)

    def test_main_without_cuda_fails_before_touching_the_output(self):
        self.base.mkdir()
        code, _, err = self.run_main("--base-model", str(self.base),
                                     "--output-dir", str(self.out), cuda=False)
        self.assertEqual(code, 2)
        self.assertIn("CUDA is required", err)
        self.assertFalse(self.out.parent.exists())

    def test_main_rejects_a_missing_base_model_and_existing_weights_up_front(self):
        code, _, err = self.run_main("--base-model", str(self.base), "--output-dir", str(self.out))
        self.assertEqual(code, 2)
        self.assertIn("base model directory not found", err)
        self.base.mkdir()
        self.assertEqual(self.run_main("--base-model", str(self.base),
                                       "--output-dir", str(self.out))[0], 0)
        code, _, err = self.run_main("--base-model", str(self.base), "--output-dir", str(self.out))
        self.assertEqual(code, 2)
        self.assertIn("--force", err)


if __name__ == "__main__":
    unittest.main()
