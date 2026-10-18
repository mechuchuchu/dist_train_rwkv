"""prepare_g1j_protocol_units.py with `datasets` and the tokenizer stubbed.

The real script downloads a dataset and loads a tokenizer. What is checked here
is the part that ties a shard to a model: the identity it records, the hashes of
the model files it reads, and that the coordinator accepts the shard for exactly
that model and no other.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from g1j_dist_coordinator import read_manifest
from rwkv_dist_protocol import sha256_bytes

OTHER_ID = "RWKV/RWKV7-G1k-2.9B-20260930"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"


class FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, **kwargs):
        ids = [len(m["content"]) % 7 + 1 for m in messages]
        mask = [1 if m["role"] == "assistant" else 0 for m in messages]
        return {"input_ids": ids, "assistant_masks": mask}


class FakeDataset(list):
    def select(self, ids):
        return FakeDataset(self[i] for i in ids)


def fake_rows(count=32):
    return FakeDataset(
        {"messages": [{"role": "user", "content": "q" * (i + 1)},
                      {"role": "assistant", "content": "a" * (i + 2)}]}
        for i in range(count)
    )


def import_script():
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: fake_rows()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
    saved = {name: sys.modules.get(name) for name in ("datasets", "transformers", "prepare_g1j_protocol_units")}
    sys.modules.update({"datasets": datasets, "transformers": transformers})
    sys.modules.pop("prepare_g1j_protocol_units", None)
    try:
        import prepare_g1j_protocol_units as script
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return script


SCRIPT = import_script()


class PrepareUnitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.release = b'{"conversion": {"serialized_parameter_count": 1}}'
        (self.model / "release-manifest.json").write_bytes(self.release)
        (self.model / "tokenizer.json").write_text("{}")
        self.output = self.root / "units.json"

    def run_script(self, *args):
        argv = ["prepare_g1j_protocol_units.py", "--output", str(self.output), *args]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            SCRIPT.main()
        return json.loads(out.getvalue())

    def test_the_shard_records_the_model_it_was_made_for(self):
        self.run_script("--model-id", OTHER_ID, "--model-revision", OTHER_REVISION,
                        "--model-path", str(self.model))
        manifest = json.loads(self.output.read_text())
        self.assertEqual(manifest["model"], {
            "repo": OTHER_ID, "revision": OTHER_REVISION,
            "release_manifest_sha256": sha256_bytes(self.release)})
        self.assertEqual(len(manifest["units"]), 4)

    def test_the_coordinator_accepts_it_for_that_model_only(self):
        self.run_script("--model-id", OTHER_ID, "--model-revision", OTHER_REVISION,
                        "--model-path", str(self.model))
        accepted = read_manifest(self.output, model_id=OTHER_ID, model_revision=OTHER_REVISION)
        self.assertEqual(len(accepted["units"]), 4)
        with self.assertRaisesRegex(ValueError, "tokenized shard was made for"):
            read_manifest(self.output, model_id=OTHER_ID, model_revision="f" * 40)
        with self.assertRaisesRegex(ValueError, "tokenized shard was made for"):
            read_manifest(self.output, model_id=SCRIPT.DEFAULT_MODEL_ID,
                          model_revision=SCRIPT.DEFAULT_MODEL_REVISION)

    def test_without_model_arguments_it_is_the_default_model(self):
        self.run_script("--model-path", str(self.model))
        manifest = json.loads(self.output.read_text())
        self.assertEqual((manifest["model"]["repo"], manifest["model"]["revision"]),
                         (SCRIPT.DEFAULT_MODEL_ID, SCRIPT.DEFAULT_MODEL_REVISION))
        read_manifest(self.output, model_id=SCRIPT.DEFAULT_MODEL_ID,
                      model_revision=SCRIPT.DEFAULT_MODEL_REVISION)

    def test_the_tokenizer_file_hash_is_recorded(self):
        self.run_script("--model-path", str(self.model))
        recorded = json.loads(self.output.read_text())["tokenizer"]["tokenizer_json_sha256"]
        self.assertEqual(recorded, sha256_bytes(b"{}"))

    def test_the_default_model_location_follows_the_model_id(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_script("--model-id", OTHER_ID, "--model-revision", OTHER_REVISION)
        self.assertIn("/workspace/RWKV7-G1k-2.9B-20260930", str(caught.exception))

    def test_a_snapshot_without_a_tokenizer_is_refused_before_anything_is_written(self):
        (self.model / "tokenizer.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_script("--model-path", str(self.model))
        self.assertFalse(self.output.exists())

    def test_unit_labels_cover_only_the_assistant_turn(self):
        self.run_script("--model-path", str(self.model), "--max-length", "8")
        for unit in json.loads(self.output.read_text())["units"]:
            self.assertEqual(unit["labels"][0], -100)
            self.assertEqual(unit["loss_tokens"], sum(label != -100 for label in unit["labels"][1:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
