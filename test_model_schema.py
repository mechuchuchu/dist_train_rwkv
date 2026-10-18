"""How a run's model identity and LoRA schema are derived, for any RWKV7 size.

Nothing here is specific to the 1.5B model: the layer count and width come from
the model's own config.json, and the paths and run names from its repository id.
The sizes below are the published RWKV7-G1j/G1k HF-format models.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import g1j_dist_coordinator as coordinator
from g1j_dist_coordinator import (
    check_lora_structure,
    default_initial_adapter_dir,
    expected_lora_shapes,
    load_initial_theta,
    read_model_shape,
)
from rwkv_dist_protocol import default_model_dir, model_name, model_slug
from test_g1j_dist_coordinator_cli import write_initial_adapter

# name -> (layers, hidden size, LoRA tensors, LoRA parameters)
SIZES = {
    "1.5B": (24, 2048, 192, 3_145_728),
    "2.9B": (32, 2560, 256, 5_242_880),
    "7.2B": (32, 4096, 256, 8_388_608),
    "13.3B": (61, 4096, 488, 15_990_784),
}


def zero_theta(layers: int, hidden: int) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(shape) for name, shape in expected_lora_shapes(layers, hidden).items()}


class ModelNameTests(unittest.TestCase):
    def test_the_name_is_the_part_after_the_organisation(self):
        self.assertEqual(model_name("RWKV/RWKV7-G1j-1.5B-20260831"), "RWKV7-G1j-1.5B-20260831")
        self.assertEqual(model_name("RWKV/RWKV7-G1k-2.9B-20260930/"), "RWKV7-G1k-2.9B-20260930")
        self.assertEqual(model_name("local-model"), "local-model")

    def test_the_default_snapshot_location_follows_the_model(self):
        self.assertEqual(default_model_dir("RWKV/RWKV7-G1j-1.5B-20260831"),
                         Path("/workspace/RWKV7-G1j-1.5B-20260831"))
        self.assertEqual(default_model_dir("RWKV/RWKV7-G1k-2.9B-20260930"),
                         Path("/workspace/RWKV7-G1k-2.9B-20260930"))

    def test_the_slug_is_a_safe_lowercase_file_name(self):
        self.assertEqual(model_slug("RWKV/RWKV7-G1j-1.5B-20260831"), "rwkv7-g1j-1.5b-20260831")
        self.assertEqual(model_slug("RWKV/RWKV7-G1k-2.9B-20260930"), "rwkv7-g1k-2.9b-20260930")
        self.assertEqual(model_slug("org/My Model (v2)"), "my-model-v2")
        self.assertEqual(model_slug("org/.hidden_"), "hidden")

    def test_a_name_that_could_not_be_a_folder_is_refused(self):
        for model_id in ("", "/", "org/..", "org/.", "org/---", "org/ ()"):
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError, "cannot derive a name"):
                    model_slug(model_id)

    def test_the_slug_never_contains_a_path_separator(self):
        for model_id in ("a/b", "a\\b", "../../etc", "a/b/c d"):
            with self.subTest(model_id=model_id):
                try:
                    slug = model_slug(model_id)
                except ValueError:
                    continue
                self.assertNotIn("/", slug)
                self.assertNotIn("\\", slug)
                self.assertFalse(slug.startswith("."))

    def test_each_model_gets_its_own_initial_adapter_folder(self):
        self.assertEqual(default_initial_adapter_dir("RWKV/RWKV7-G1j-1.5B-20260831"),
                         Path("artifacts/rwkv7-g1j-1.5b-20260831_lora_init"))
        self.assertEqual(default_initial_adapter_dir("RWKV/RWKV7-G1k-2.9B-20260930"),
                         Path("artifacts/rwkv7-g1k-2.9b-20260930_lora_init"))


class LoraSchemaTests(unittest.TestCase):
    def test_the_schema_follows_layers_and_width(self):
        for label, (layers, hidden, tensors, parameters) in SIZES.items():
            with self.subTest(size=label):
                shapes = expected_lora_shapes(layers, hidden)
                self.assertEqual(len(shapes), tensors)
                self.assertEqual(sum(a * b for a, b in shapes.values()), parameters)
                self.assertEqual(set(shapes.values()), {(8, hidden), (hidden, 8)})
                self.assertIn(f"base_model.model.model.layers.{layers - 1}.att.value.lora_B.default.weight",
                              shapes)
                self.assertNotIn(f"base_model.model.model.layers.{layers}.att.value.lora_B.default.weight",
                                 shapes)

    def test_a_complete_adapter_reports_its_own_size(self):
        for label, (layers, hidden, *_rest) in SIZES.items():
            with self.subTest(size=label):
                self.assertEqual(check_lora_structure(zero_theta(layers, hidden)), (layers, hidden))

    def test_a_malformed_adapter_is_refused(self):
        def without(prefix):
            return {k: v for k, v in zero_theta(4, 64).items() if not k.startswith(prefix)}

        base = zero_theta(4, 64)
        cases = {
            "empty": ({}, "no LoRA tensors"),
            "a whole layer missing": (without("base_model.model.model.layers.2."), "layers 0..N-1"),
            "layers not starting at 0": (
                {k.replace("layers.0.", "layers.4."): v for k, v in base.items()}, "layers 0..N-1"),
            "a projection missing": (without("base_model.model.model.layers.1.att.key"), "receptance/key/value/output"),
            "lora_B missing": (
                {k: v for k, v in base.items() if k != "base_model.model.model.layers.0.att.key.lora_B.default.weight"},
                "only lora_A"),
            "wrong rank": (
                {**base, "base_model.model.model.layers.0.att.key.lora_A.default.weight": torch.zeros(16, 64),
                 "base_model.model.model.layers.0.att.key.lora_B.default.weight": torch.zeros(64, 16)},
                "not a rank-8 LoRA pair"),
            "A and B disagree about the width": (
                {**base, "base_model.model.model.layers.0.att.key.lora_B.default.weight": torch.zeros(32, 8)},
                "not a rank-8 LoRA pair"),
            "one module of another width": (
                {**base, "base_model.model.model.layers.0.att.key.lora_A.default.weight": torch.zeros(8, 128),
                 "base_model.model.model.layers.0.att.key.lora_B.default.weight": torch.zeros(128, 8)},
                "widths differ"),
            "an extra tensor": ({**base, "base_model.model.lm_head.weight": torch.zeros(2)}, "unexpected tensor"),
            "a feed-forward adapter": (
                {**base, "base_model.model.model.layers.0.ffn.key.lora_A.default.weight": torch.zeros(8, 64)},
                "unexpected tensor"),
            "not matrices": (
                {**base, "base_model.model.model.layers.0.att.key.lora_A.default.weight": torch.zeros(8)},
                "not a rank-8 LoRA pair"),
        }
        for label, (theta, message) in cases.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    check_lora_structure(theta)


class ModelShapeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_the_shape_is_read_from_config_json(self):
        for label, (layers, hidden, *_rest) in SIZES.items():
            with self.subTest(size=label):
                (self.dir / "config.json").write_text(json.dumps(
                    {"num_hidden_layers": layers, "hidden_size": hidden, "head_dim": 64}))
                self.assertEqual(read_model_shape(self.dir), (layers, hidden))

    def test_a_snapshot_without_config_json_has_no_known_shape(self):
        self.assertIsNone(read_model_shape(self.dir))

    def test_a_config_without_the_size_is_an_error(self):
        for content in ({"hidden_size": 2048}, {"num_hidden_layers": 24}, {}, []):
            with self.subTest(content=content):
                (self.dir / "config.json").write_text(json.dumps(content))
                with self.assertRaisesRegex(ValueError, "num_hidden_layers / hidden_size"):
                    read_model_shape(self.dir)

    def test_a_config_that_is_not_json_is_an_error(self):
        (self.dir / "config.json").write_text("{not json")
        with self.assertRaises(ValueError):
            read_model_shape(self.dir)


class LoadInitialThetaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "adapter"

    def test_adapters_of_every_size_load_as_float32(self):
        for label, (layers, hidden, tensors, _parameters) in SIZES.items():
            with self.subTest(size=label):
                directory = self.dir.parent / label
                write_initial_adapter(directory, layers, hidden)
                theta = load_initial_theta(directory, model_shape=(layers, hidden))
                self.assertEqual(len(theta), tensors)
                self.assertTrue(all(t.dtype == torch.float32 for t in theta.values()))
                self.assertTrue(all(".default." in name for name in theta))

    def test_half_precision_files_are_widened_to_float32(self):
        write_initial_adapter(self.dir, 2, 64)
        path = self.dir / "adapter_model.safetensors"
        from safetensors.torch import load_file
        save_file({k: v.to(torch.bfloat16) for k, v in load_file(str(path)).items()}, str(path))
        theta = load_initial_theta(self.dir)
        self.assertTrue(all(t.dtype == torch.float32 for t in theta.values()))

    def test_tensors_that_are_not_lora_weights_are_ignored(self):
        write_initial_adapter(self.dir, 2, 64)
        path = self.dir / "adapter_model.safetensors"
        from safetensors.torch import load_file
        save_file({**load_file(str(path)), "base_model.model.lm_head.weight": torch.zeros(2)}, str(path))
        self.assertEqual(len(load_initial_theta(self.dir)), 16)

    def test_the_model_shape_is_enforced_when_given(self):
        write_initial_adapter(self.dir, 32, 2560)
        load_initial_theta(self.dir)  # structure alone is fine
        load_initial_theta(self.dir, model_shape=(32, 2560))
        for wrong in ((24, 2048), (32, 4096), (31, 2560), (61, 2560)):
            with self.subTest(model=wrong):
                with self.assertRaisesRegex(
                        ValueError, rf"32 layers of width 2560.*{wrong[0]} layers of width {wrong[1]}"):
                    load_initial_theta(self.dir, model_shape=wrong)

    def test_the_wrong_rank_or_alpha_is_refused(self):
        write_initial_adapter(self.dir, 2, 64)
        for config in ({"r": 16, "lora_alpha": 16}, {"r": 8, "lora_alpha": 32}, {}):
            with self.subTest(config=config):
                (self.dir / "adapter_config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "rank 8 / alpha 16"):
                    load_initial_theta(self.dir)

    def test_missing_files_are_named(self):
        write_initial_adapter(self.dir, 2, 64)
        (self.dir / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "adapter_model.safetensors"):
            load_initial_theta(self.dir)
        with self.assertRaisesRegex(FileNotFoundError, "adapter_config.json"):
            load_initial_theta(self.dir.parent / "nowhere")

    def test_the_rank_and_alpha_the_protocol_uses_are_the_documented_ones(self):
        self.assertEqual((coordinator.LORA_RANK, coordinator.LORA_ALPHA), (8, 16))
        self.assertEqual(coordinator.ATTENTION_MODULES, ("receptance", "key", "value", "output"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
