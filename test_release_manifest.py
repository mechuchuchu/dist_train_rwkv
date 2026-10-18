"""convert_k2_horizon/make_release_manifest.py on tiny converted-checkpoint folders.

The script's output has to satisfy the rest of the project: the coordinator reads
`conversion.serialized_parameter_count`, pins the file's hash, and a data shard made
against the folder is accepted for that hash only. The last tests run the real
coordinator start-up on a folder this script describes.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

import g1j_dist_coordinator as coordinator_module
from rwkv_dist_protocol import canonical_json, sha256_bytes
from rwkv_hub import ensure_hf_checkpoint
from test_g1j_dist_coordinator_cli import ReturningServer, write_fixtures

SCRIPT_PATH = Path(__file__).parent / "convert_k2_horizon" / "make_release_manifest.py"
spec = importlib.util.spec_from_file_location("make_release_manifest", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SCRIPT)

BASE = 65536
EXTRA = 3
HIDDEN = 8
LAYERS = 2


def write_checkpoint(folder: Path, *, extra: int = EXTRA, hidden: int = HIDDEN, shards: int = 1) -> int:
    """A converted-looking folder; returns its true parameter count."""
    folder.mkdir(parents=True, exist_ok=True)
    vocab = BASE + extra
    (folder / "config.json").write_text(json.dumps(
        {"vocab_size": vocab, "hidden_size": hidden, "num_hidden_layers": LAYERS, "model_type": "rwkv7"}))
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "rwkv7.emb.weight": torch.randn(vocab, hidden, generator=generator).to(torch.bfloat16),
        "head.weight": torch.randn(vocab, hidden, generator=generator).to(torch.bfloat16),
        "rwkv7.ln_out.weight": torch.randn(hidden, generator=generator),
        "rwkv7.blocks.0.att.r_proj.weight": torch.randn(hidden, hidden, generator=generator),
    }
    names = list(tensors)
    if shards == 1:
        save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    else:
        weight_map = {}
        for index in range(shards):
            part = {name: tensors[name] for name in names[index::shards]}
            filename = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
            save_file(part, str(folder / filename), metadata={"format": "pt"})
            weight_map.update({name: filename for name in part})
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "tokenizer.json").write_text('{"added_tokens": []}')
    (folder / "tokenizer_config.json").write_text("{}")
    (folder / "chat_template.jinja").write_text("{{ messages }}")
    (folder / "modeling_rwkv7.py").write_text("# remote code\n")
    return sum(tensor.numel() for tensor in tensors.values())


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = SCRIPT.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "converted"
        self.parameters = write_checkpoint(self.folder)

    def manifest(self) -> dict:
        return json.loads((self.folder / "release-manifest.json").read_text())


class WritingTests(ManifestTestCase):
    def test_the_parameter_count_is_read_from_the_headers(self):
        code, out, _ = run(str(self.folder))
        self.assertEqual(code, 0)
        conversion = self.manifest()["conversion"]
        self.assertEqual(conversion["serialized_parameter_count"], self.parameters)
        self.assertEqual(conversion["tensor_count"], 4)
        self.assertEqual(conversion["dtypes"], {"BF16": 2, "F32": 2})
        self.assertIn(f"{self.parameters:,}", out)

    def test_the_model_section_says_how_many_tokens_are_new(self):
        run(str(self.folder))
        self.assertEqual(self.manifest()["model"], {
            "vocab_size": BASE + EXTRA, "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
            "base_vocab_size": BASE, "new_token_count": EXTRA})

    def test_every_file_is_listed_with_its_size_and_hash_except_the_manifest_and_hidden_entries(self):
        (self.folder / ".cache").mkdir()
        (self.folder / ".cache" / "noise").write_text("x")
        (self.folder / "__pycache__").mkdir()
        (self.folder / "__pycache__" / "m.pyc").write_bytes(b"x")
        (self.folder / ".gitattributes").write_text("x")
        run(str(self.folder))
        files = self.manifest()["files"]
        self.assertEqual(sorted(files), ["chat_template.jinja", "config.json", "model.safetensors",
                                         "modeling_rwkv7.py", "tokenizer.json", "tokenizer_config.json"])
        weights = (self.folder / "model.safetensors").read_bytes()
        self.assertEqual(files["model.safetensors"],
                         {"bytes": len(weights), "sha256": hashlib.sha256(weights).hexdigest()})

    def test_it_is_deterministic_and_prints_the_hash_of_the_file_it_wrote(self):
        _, first_out, _ = run(str(self.folder))
        first = (self.folder / "release-manifest.json").read_bytes()
        code, second_out, _ = run(str(self.folder))
        self.assertEqual(code, 0)
        self.assertEqual((self.folder / "release-manifest.json").read_bytes(), first)
        digest = sha256_bytes(first)
        self.assertIn(f"manifest sha256: {digest}", first_out)
        self.assertIn(f"manifest sha256: {digest}", second_out)

    def test_the_same_files_in_another_place_give_the_same_manifest(self):
        run(str(self.folder))
        copy = self.root / "elsewhere" / "other-name"
        shutil.copytree(self.folder, copy)
        (copy / "release-manifest.json").unlink()
        run(str(copy))
        self.assertEqual((copy / "release-manifest.json").read_bytes(),
                         (self.folder / "release-manifest.json").read_bytes())

    def test_a_changed_folder_does_not_silently_replace_its_manifest(self):
        run(str(self.folder))
        before = (self.folder / "release-manifest.json").read_bytes()
        (self.folder / "tokenizer_config.json").write_text('{"changed": true}')
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("--force", err)
        self.assertEqual((self.folder / "release-manifest.json").read_bytes(), before)
        code, _, _ = run(str(self.folder), "--force")
        self.assertEqual(code, 0)
        self.assertNotEqual((self.folder / "release-manifest.json").read_bytes(), before)

    def test_it_suggests_a_stable_stand_in_revision_of_forty_hex_characters(self):
        _, out, _ = run(str(self.folder))
        digest = sha256_bytes((self.folder / "release-manifest.json").read_bytes())
        suggested = re.search(r"derived from the manifest:\s+(\S+)", out).group(1)
        self.assertEqual(suggested, digest[:40])
        self.assertRegex(suggested, r"^[0-9a-f]{40}$")

    def test_the_manifest_is_valid_json_with_sorted_keys_and_a_final_newline(self):
        run(str(self.folder))
        text = (self.folder / "release-manifest.json").read_text()
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(text, json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


class CheckTests(ManifestTestCase):
    def setUp(self):
        super().setUp()
        run(str(self.folder))

    def test_an_unchanged_folder_passes_and_reports_the_same_hash(self):
        code, out, _ = run(str(self.folder), "--check")
        self.assertEqual(code, 0)
        self.assertIn("OK", out)
        self.assertIn(sha256_bytes((self.folder / "release-manifest.json").read_bytes()), out)

    def test_a_single_flipped_byte_in_the_weights_is_caught_by_name(self):
        path = self.folder / "model.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x01
        path.write_bytes(bytes(data))
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 1)
        self.assertIn("model.safetensors: contents differ", err)

    def test_a_missing_and_an_extra_file_are_both_reported(self):
        (self.folder / "tokenizer.json").unlink()
        (self.folder / "surprise.txt").write_text("x")
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 1)
        self.assertIn("tokenizer.json: listed in the manifest but missing", err)
        self.assertIn("surprise.txt: in the folder but not in the manifest", err)

    def test_a_changed_config_is_caught(self):
        config = json.loads((self.folder / "config.json").read_text())
        config["rope"] = 1
        (self.folder / "config.json").write_text(json.dumps(config))
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 1)
        self.assertIn("config.json: contents differ", err)

    def test_a_changed_model_section_is_reported_even_when_no_listed_file_differs(self):
        recorded = self.manifest()
        recorded["model"]["num_hidden_layers"] += 1
        recorded["conversion"]["tensor_count"] += 1
        (self.folder / "release-manifest.json").write_text(json.dumps(recorded))
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 1)
        self.assertIn("model:", err)
        self.assertIn("conversion:", err)
        self.assertNotIn("contents differ", err)

    def test_check_without_a_manifest_says_so(self):
        (self.folder / "release-manifest.json").unlink()
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_check_does_not_modify_anything(self):
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        run(str(self.folder), "--check")
        self.assertEqual({p.name: p.read_bytes() for p in self.folder.iterdir()}, before)

    def test_a_manifest_that_is_not_json_is_an_error_not_a_crash(self):
        (self.folder / "release-manifest.json").write_text("{nope")
        code, _, err = run(str(self.folder), "--check")
        self.assertEqual(code, 2)
        self.assertIn("not valid JSON", err)


class RefusalTests(ManifestTestCase):
    def test_a_folder_that_does_not_exist(self):
        code, _, err = run(str(self.root / "nowhere"))
        self.assertEqual(code, 2)
        self.assertIn("is not a folder", err)

    def test_no_weights(self):
        (self.folder / "model.safetensors").unlink()
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("no .safetensors weights", err)
        self.assertFalse((self.folder / "release-manifest.json").exists())

    def test_no_config(self):
        (self.folder / "config.json").unlink()
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("no config.json", err)

    def test_a_config_without_the_sizes(self):
        (self.folder / "config.json").write_text(json.dumps({"vocab_size": BASE}))
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("hidden_size", err)

    def test_a_vocabulary_smaller_than_rwkv_world(self):
        shutil.rmtree(self.folder)
        write_checkpoint(self.folder, extra=-10)
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("below the RWKV World vocabulary", err)

    def test_weights_that_disagree_with_the_config(self):
        config = json.loads((self.folder / "config.json").read_text())
        config["vocab_size"] += 1
        (self.folder / "config.json").write_text(json.dumps(config))
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("different conversions", err)

    def test_a_file_that_is_not_safetensors(self):
        (self.folder / "model.safetensors").write_bytes(b"short")
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("too short", err)

    def test_an_absurd_header_length(self):
        (self.folder / "model.safetensors").write_bytes((2 ** 40).to_bytes(8, "little") + b"{}")
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("implausible", err)

    def test_a_truncated_header(self):
        (self.folder / "model.safetensors").write_bytes((100).to_bytes(8, "little") + b"{}")
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("cut off", err)


class ShardedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "sharded"
        self.parameters = write_checkpoint(self.folder, shards=3)

    def test_parameters_are_summed_across_shards(self):
        self.assertEqual(run(str(self.folder))[0], 0)
        manifest = json.loads((self.folder / "release-manifest.json").read_text())
        self.assertEqual(manifest["conversion"]["serialized_parameter_count"], self.parameters)
        self.assertEqual(sum(name.endswith(".safetensors") for name in manifest["files"]), 3)
        self.assertIn("model.safetensors.index.json", manifest["files"])

    def test_a_missing_shard_is_caught_against_the_index(self):
        (self.folder / "model-00002-of-00003.safetensors").unlink()
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("do not match model.safetensors.index.json", err)

    def test_a_tensor_present_in_two_shards_is_refused(self):
        one = load_file(str(self.folder / "model-00001-of-00003.safetensors"))
        two = load_file(str(self.folder / "model-00002-of-00003.safetensors"))
        two["head.weight" if "head.weight" in one else next(iter(one))] = next(iter(one.values()))
        save_file(two, str(self.folder / "model-00002-of-00003.safetensors"))
        code, _, err = run(str(self.folder))
        self.assertEqual(code, 2)
        self.assertIn("more than one shard", err)


class ProjectConsumerTests(unittest.TestCase):
    """What the training code does with a folder this script described."""

    HIDDEN = 64
    LAYERS = 2

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_the_hub_helper_accepts_the_folder_only_once_it_has_a_manifest(self):
        folder = self.root / "converted"
        write_checkpoint(folder)
        with self.assertRaisesRegex(FileNotFoundError, "release-manifest.json"):
            ensure_hf_checkpoint("local/converted", "a" * 40, folder, fetch_from_hf=False)
        run(str(folder))
        self.assertEqual(ensure_hf_checkpoint("local/converted", "a" * 40, folder, fetch_from_hf=False),
                         folder.resolve())

    def test_the_coordinator_starts_on_it_and_trains_the_new_rows(self):
        fixtures = write_fixtures(self.root / "fixtures", layers=self.LAYERS, hidden=self.HIDDEN)
        shutil.rmtree(fixtures["model"])
        parameters = write_checkpoint(fixtures["model"], hidden=self.HIDDEN)
        run(str(fixtures["model"]))
        manifest_hash = sha256_bytes((fixtures["model"] / "release-manifest.json").read_bytes())

        shard = json.loads(fixtures["units"].read_text())
        shard["model"]["release_manifest_sha256"] = manifest_hash
        content = {key: value for key, value in shard.items() if key != "manifest_sha256"}
        shard["manifest_sha256"] = sha256_bytes(canonical_json(content))
        fixtures["units"].write_text(json.dumps(shard))

        run_dir = self.root / "run"
        argv = ["g1j_dist_coordinator.py", "--units", str(fixtures["units"]),
                "--model-dir", str(fixtures["model"]), "--initial-adapter-dir", str(fixtures["adapter"]),
                "--run-dir", str(run_dir), "--token", "protocol-token", "--port", "0",
                "--units-per-step", "2", "--min-units-per-step", "2"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch.object(coordinator_module, "ThreadingHTTPServer", ReturningServer), \
                contextlib.redirect_stdout(output):
            coordinator_module.main()

        self.assertIn(f"{EXTRA} token ids beyond the RWKV World vocabulary", output.getvalue())
        theta = load_file(str(run_dir / "theta.safetensors"))
        self.assertEqual(tuple(theta["new_token_rows.head_delta"].shape), (EXTRA, self.HIDDEN))
        saved = json.loads(next((run_dir / "ckpt").glob("theta-*/state.json")).read_text())
        self.assertEqual(saved["base"]["serialized_parameter_count"], parameters)
        self.assertEqual(saved["base"]["release_manifest_sha256"], manifest_hash)

    def test_a_folder_edited_after_the_manifest_was_made_is_not_the_pinned_model(self):
        folder = self.root / "converted"
        write_checkpoint(folder)
        run(str(folder))
        pinned = sha256_bytes((folder / "release-manifest.json").read_bytes())
        (folder / "chat_template.jinja").write_text("{{ different }}")
        run(str(folder), "--force")
        self.assertNotEqual(sha256_bytes((folder / "release-manifest.json").read_bytes()), pinned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
