"""Drive g1j_dist_coordinator.main() from the command line on small fixtures.

The HTTP server loop is replaced by one that returns immediately, so each call
performs the real startup (argument checks, manifest and model checks, resume
or fresh start, checkpoint commit, status line) and then exits. This covers the
wiring between the CLI flags and the pieces that are unit-tested separately.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import g1j_dist_coordinator as coordinator_module
from rwkv_dist_checkpoint import ResumeError, list_checkpoint_versions, load_latest_checkpoint
from rwkv_dist_protocol import canonical_json, sha256_bytes, unit_digest


class ReturningServer:
    def __init__(self, address, handler):
        self.address = address

    def serve_forever(self, poll_interval=0.5):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def write_initial_adapter(directory: Path, layers: int = 24, hidden: int = 2048) -> None:
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "r": 8, "lora_alpha": 16, "base_model_name_or_path": "/local/base",
    }))
    tensors = {}
    for layer in range(layers):
        for module in ("receptance", "key", "value", "output"):
            prefix = f"base_model.model.model.layers.{layer}.att.{module}"
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros((8, hidden))
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros((hidden, 8))
    save_file(tensors, str(directory / "adapter_model.safetensors"))


def write_fixtures(root: Path, *, layers: int = 24, hidden: int = 2048,
                   model_id: str = coordinator_module.DEFAULT_MODEL_ID,
                   revision: str = coordinator_module.DEFAULT_MODEL_REVISION,
                   config: bool = True, adapter_shape: tuple[int, int] | None = None
                   ) -> dict[str, Path]:
    """A model snapshot of `layers` x `hidden`, a shard made for it, and a matching adapter.

    `adapter_shape` builds the starting adapter for a different size, to test the mismatch.
    """
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "model"
    model_dir.mkdir()
    release = json.dumps({"conversion": {"serialized_parameter_count": 123},
                          "model": model_id}).encode()
    (model_dir / "release-manifest.json").write_bytes(release)
    if config:
        (model_dir / "config.json").write_text(json.dumps(
            {"num_hidden_layers": layers, "hidden_size": hidden}))
    units = []
    for index in range(6):
        unit = {"unit_id": f"u{index}", "input_ids": [1, 2], "labels": [-100, 2],
                "loss_tokens": 1, "total_tokens": 2}
        unit["unit_sha256"] = unit_digest(unit)
        units.append(unit)
    content = {
        "model": {
            "repo": model_id,
            "revision": revision,
            "release_manifest_sha256": sha256_bytes(release),
        },
        "tokenizer": {"name": "test-tokenizer"},
        "units": units,
    }
    manifest = {**content, "manifest_sha256": sha256_bytes(canonical_json(content))}
    units_path = root / "units.json"
    units_path.write_text(json.dumps(manifest))
    adapter_dir = root / "initial-adapter"
    write_initial_adapter(adapter_dir, *(adapter_shape or (layers, hidden)))
    return {"units": units_path, "model": model_dir, "adapter": adapter_dir}


class CoordinatorCommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = write_fixtures(self.root)

    def run_main(self, run_dir: Path, *extra: str, defaults: bool = True) -> tuple[dict, str]:
        """Run main() and return (the status line it printed, everything else it printed).

        With defaults=False the model and adapter locations are left to the program.
        """
        located = ["--model-dir", str(self.fixtures["model"]),
                   "--initial-adapter-dir", str(self.fixtures["adapter"])] if defaults else []
        argv = [
            "g1j_dist_coordinator.py", "--units", str(self.fixtures["units"]), *located,
            "--run-dir", str(run_dir), "--token", "protocol-token", "--port", "0",
            "--units-per-step", "2", "--min-units-per-step", "2", *extra,
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch.object(coordinator_module, "ThreadingHTTPServer", ReturningServer), \
                redirect_stdout(output):
            coordinator_module.main()
        text = output.getvalue()
        start = text.index('{\n  "listen"')
        return json.loads(text[start:]), text[:start]

    def refused(self, run_dir: Path, *extra: str, **options) -> str:
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            self.run_main(run_dir, *extra, **options)
        self.assertEqual(caught.exception.code, 2)
        return errors.getvalue()

    def test_fresh_start_commits_version_zero_and_reports_it(self):
        run_dir = self.root / "run"
        status, _ = self.run_main(run_dir)

        self.assertFalse(status["resumed"])
        self.assertEqual(status["round_id"], 1)
        self.assertEqual(status["planned_optimizer_steps"], 3)
        self.assertEqual(list_checkpoint_versions(run_dir / "ckpt"), [0])
        self.assertEqual(load_latest_checkpoint(run_dir / "ckpt").state["run_id"], status["run_id"])

    def test_fresh_start_over_an_existing_run_is_refused(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)
        message = self.refused(run_dir)
        self.assertIn("already holds checkpoints", message)
        self.assertIn("--resume", message)

    def test_a_fresh_start_without_an_initial_adapter_says_how_to_make_one(self):
        (self.fixtures["adapter"] / "adapter_model.safetensors").unlink()
        message = self.refused(self.root / "run")
        self.assertIn("adapter_model.safetensors", message)
        self.assertIn("python make_initial_adapter.py", message)
        self.assertIn(str(self.fixtures["adapter"]), message)
        self.assertFalse((self.root / "run" / "ckpt").exists())

    def test_an_initial_adapter_of_the_wrong_rank_is_a_usage_error(self):
        (self.fixtures["adapter"] / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 16}))
        self.assertIn("rank 8 / alpha 16", self.refused(self.root / "run"))

    def test_the_default_initial_adapter_dir_is_the_one_the_script_writes(self):
        # Neither location is given: the adapter folder is derived from the model name,
        # which is also where make_initial_adapter.py writes by default.
        message = self.refused(self.root / "run", "--model-dir", str(self.fixtures["model"]),
                               defaults=False)
        derived = coordinator_module.default_initial_adapter_dir(coordinator_module.DEFAULT_MODEL_ID)
        self.assertEqual(derived, Path("artifacts/rwkv7-g1j-1.5b-20260831_lora_init"))
        self.assertIn(f"cannot start from {derived}", message)
        self.assertIn(f"--output-dir {derived}", message)

    def test_the_missing_adapter_hint_names_the_model_it_is_for(self):
        shutil.rmtree(self.fixtures["adapter"])
        message = self.refused(self.root / "run")
        self.assertIn(f"--model-id {coordinator_module.DEFAULT_MODEL_ID}", message)
        self.assertIn(f"--model-revision {coordinator_module.DEFAULT_MODEL_REVISION}", message)
        self.assertIn(f"--base-model {self.fixtures['model']}", message)

    # --- other models -------------------------------------------------------------------

    OTHER_ID = "RWKV/RWKV7-G1k-2.9B-20260930"
    OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"

    def use_model(self, name: str, **options) -> list[str]:
        """Switch to fixtures for another model; return the flags that select it."""
        model_id = options.setdefault("model_id", self.OTHER_ID)
        revision = options.setdefault("revision", self.OTHER_REVISION)
        self.fixtures = write_fixtures(self.root / name, **options)
        return ["--model-id", model_id, "--model-revision", revision]

    def test_other_model_sizes_start_from_a_matching_adapter(self):
        sizes = {"2.9B": (32, 2560), "7.2B": (32, 4096), "13.3B": (61, 4096)}
        for label, (layers, hidden) in sizes.items():
            with self.subTest(size=label):
                flags = self.use_model(label, layers=layers, hidden=hidden)
                status, _ = self.run_main(self.root / f"run-{label}", *flags)

                self.assertEqual(status["model_id"], self.OTHER_ID)
                self.assertEqual(status["model_revision"], self.OTHER_REVISION)
                self.assertTrue(status["run_id"].startswith("rwkv7-g1k-2.9b-20260930-"), status["run_id"])
                theta = load_latest_checkpoint(self.root / f"run-{label}" / "ckpt").theta
                self.assertEqual(len(theta), layers * 8)
                self.assertEqual(sum(t.numel() for t in theta.values()), layers * 64 * hidden)
                shapes = {tuple(t.shape) for t in theta.values()}
                self.assertEqual(shapes, {(8, hidden), (hidden, 8)})

    def test_the_run_records_which_model_it_trains(self):
        flags = self.use_model("other", layers=32, hidden=2560)
        self.run_main(self.root / "run", *flags)
        state = load_latest_checkpoint(self.root / "run" / "ckpt").state
        self.assertEqual(state["base"]["repo"], self.OTHER_ID)
        self.assertEqual(state["base"]["revision"], self.OTHER_REVISION)

    def test_an_adapter_made_for_another_size_is_refused_with_both_sizes(self):
        flags = self.use_model("mismatch", layers=32, hidden=2560, adapter_shape=(24, 2048))
        message = self.refused(self.root / "run", *flags)
        self.assertIn("24 layers of width 2048", message)
        self.assertIn("32 layers of width 2560", message)
        self.assertFalse((self.root / "run" / "ckpt").exists())

    def test_an_adapter_with_the_right_layers_but_another_width_is_refused(self):
        flags = self.use_model("mismatch", layers=32, hidden=4096, adapter_shape=(32, 2560))
        self.assertIn("32 layers of width 2560", self.refused(self.root / "run", *flags))

    def test_an_incomplete_adapter_is_refused_whatever_the_model(self):
        flags = self.use_model("partial", layers=32, hidden=2560)
        write_initial_adapter(self.root / "partial-adapter", layers=31, hidden=2560)
        self.fixtures["adapter"] = self.root / "partial-adapter"
        # 31 layers is a valid adapter for a 31-layer model, so it is the config that rejects it.
        self.assertIn("31 layers of width 2560", self.refused(self.root / "run", *flags))

    def test_a_snapshot_without_config_json_still_starts_but_says_it_cannot_check_the_size(self):
        flags = self.use_model("noconfig", layers=32, hidden=2560, config=False)
        status, printed = self.run_main(self.root / "run", *flags)
        self.assertEqual(status["model_id"], self.OTHER_ID)
        self.assertIn("config.json not found", printed)
        self.assertIn("workers verify the exact tensor names and shapes", printed)

    def test_a_config_without_the_model_size_is_refused(self):
        flags = self.use_model("badconfig", layers=32, hidden=2560)
        (self.fixtures["model"] / "config.json").write_text(json.dumps({"hidden_size": 2560}))
        self.assertIn("num_hidden_layers", self.refused(self.root / "run", *flags))

    def test_the_default_model_location_follows_the_model_id(self):
        flags = self.use_model("other", layers=32, hidden=2560)
        seen = []

        def fake_ensure(model_id, revision, model_dir, **kwargs):
            seen.append((model_id, revision, model_dir))
            return self.fixtures["model"]

        with patch.object(coordinator_module, "ensure_hf_checkpoint", fake_ensure):
            self.run_main(self.root / "run", *flags, "--initial-adapter-dir",
                          str(self.fixtures["adapter"]), defaults=False)
        self.assertEqual(seen, [(self.OTHER_ID, self.OTHER_REVISION,
                                 Path("/workspace/RWKV7-G1k-2.9B-20260930"))])

    def test_an_explicit_model_dir_overrides_the_default_location(self):
        seen = []

        def fake_ensure(model_id, revision, model_dir, **kwargs):
            seen.append(model_dir)
            return self.fixtures["model"]

        with patch.object(coordinator_module, "ensure_hf_checkpoint", fake_ensure):
            self.run_main(self.root / "run")
        self.assertEqual(seen, [self.fixtures["model"]])

    def test_a_shard_tokenized_for_another_model_is_refused(self):
        # The shard says RWKV7-G1j; the run asks for G1k, or for another commit of G1j.
        wrong_model = ["--model-id", self.OTHER_ID]
        self.assertIn("tokenized shard was made for", self.refused(self.root / "a", *wrong_model))
        wrong_commit = ["--model-revision", self.OTHER_REVISION]
        message = self.refused(self.root / "b", *wrong_commit)
        self.assertIn("tokenized shard was made for", message)
        self.assertIn(coordinator_module.DEFAULT_MODEL_REVISION, message)
        self.assertIn(self.OTHER_REVISION, message)

    def test_a_branch_or_tag_is_not_accepted_as_a_model_revision(self):
        for revision in ("main", "2c18b29", "2C18B29AB7FBECE25FF6112281EEA0FA41FCB30F"):
            with self.subTest(revision=revision):
                self.assertIn("40-character commit hash",
                              self.refused(self.root / "run", "--model-revision", revision))

    def test_a_model_id_without_a_usable_name_is_a_usage_error(self):
        self.assertIn("cannot derive a name", self.refused(self.root / "run", "--model-id", "/"))

    def test_a_bad_units_file_is_a_usage_error_not_a_traceback(self):
        manifest = json.loads(self.fixtures["units"].read_text())
        without_model = {k: v for k, v in manifest.items() if k not in ("model", "manifest_sha256")}
        without_model["manifest_sha256"] = sha256_bytes(canonical_json(without_model))
        no_units = {k: v for k, v in manifest.items() if k not in ("units", "manifest_sha256")}
        no_units["units"] = []
        no_units["manifest_sha256"] = sha256_bytes(canonical_json(
            {k: v for k, v in no_units.items() if k != "manifest_sha256"}))
        cases = {
            "hash mismatch": {**manifest, "units": manifest["units"][:-1]},
            "no model": without_model,
            "no units": no_units,
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self.fixtures["units"].write_text(json.dumps(content))
                self.assertIn("cannot use --units", self.refused(self.root / f"run-{name}"))
        self.fixtures["units"].write_text("not json")
        self.assertIn("cannot use --units", self.refused(self.root / "run-text"))
        self.fixtures["units"].unlink()
        self.assertIn("cannot use --units", self.refused(self.root / "run-missing"))

    def test_resuming_a_run_with_another_model_is_refused(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)
        flags = self.use_model("other", layers=24, hidden=2048)  # same size, different model
        message = self.refused(run_dir, "--resume", *flags)
        self.assertIn("cannot resume", message)
        self.assertIn("base (checkpoint", message)

    def test_a_local_resume_does_not_say_the_credit_was_lost(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)
        _, printed = self.run_main(run_dir, "--resume")
        self.assertNotIn("contributor credit", printed)

    def test_resume_continues_the_same_run(self):
        run_dir = self.root / "run"
        first, _ = self.run_main(run_dir)

        second, _ = self.run_main(run_dir, "--resume", "--keep-checkpoints", "5")

        self.assertTrue(second["resumed"])
        self.assertEqual(second["resumed_from"], "local")
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["theta_hash"], first["theta_hash"])
        events = [json.loads(line) for line in (run_dir / "round_log.jsonl").read_text().splitlines()]
        self.assertEqual([e["event"] for e in events], ["coordinator_resumed"])

    def test_resume_with_nothing_to_resume_is_an_error(self):
        message = self.refused(self.root / "empty", "--resume")
        self.assertIn("cannot resume", message)
        self.assertIn("no checkpoints", message)

    def test_resume_with_changed_settings_is_refused_with_the_reason(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)
        self.assertIn("hyperparameters", self.refused(run_dir, "--resume", "--learning-rate", "0.5"))
        self.assertIn("round_plan_sha256", self.refused(run_dir, "--resume", "--epochs", "2"))

    def test_resume_from_the_hub_starts_a_fresh_run_directory_from_the_pushed_run(self):
        origin, _ = self.run_main(self.root / "origin")
        pushed = load_latest_checkpoint(self.root / "origin" / "ckpt")
        calls = []

        def fake_load(**kwargs):
            calls.append(kwargs)
            return pushed

        with patch.object(coordinator_module, "load_resume_from_hub", fake_load):
            status, printed = self.run_main(
                self.root / "restored", "--resume-from-hf", "--hf-adapter-repo", "user/adapter",
                "--hf-run-id", origin["run_id"], "--hf-version", "0", "--hf-token", "read-token",
            )

        self.assertEqual(calls[0]["repo_id"], "user/adapter")
        self.assertEqual(calls[0]["run_id"], origin["run_id"])
        self.assertEqual(calls[0]["version"], 0)
        self.assertEqual(calls[0]["token"], "read-token")
        self.assertTrue(status["resumed"])
        self.assertEqual(status["resumed_from"], "hub")
        self.assertEqual(status["run_id"], origin["run_id"])
        self.assertFalse(status["optimizer_restored"])
        self.assertIn("Adam moments start cold", printed)
        self.assertIn("contributor credit is not stored on the Hub", printed)
        # The restored run now has its own local checkpoints, so a later --resume works offline.
        self.assertEqual(list_checkpoint_versions(self.root / "restored" / "ckpt"), [0])

    def test_hub_resume_failure_is_reported_not_raised(self):
        def failing(**kwargs):
            raise ResumeError("repo has no complete checkpoints for run x")

        with patch.object(coordinator_module, "load_resume_from_hub", failing):
            message = self.refused(
                self.root / "restored", "--resume-from-hf", "--hf-adapter-repo", "user/adapter",
                "--hf-run-id", "x",
            )
        self.assertIn("cannot resume from the Hub", message)

    def test_hub_resume_never_mixes_with_an_existing_local_run(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)
        message = self.refused(run_dir, "--resume-from-hf", "--hf-adapter-repo", "user/adapter",
                               "--hf-run-id", "x")
        self.assertIn("already holds checkpoints", message)

    def test_inconsistent_resume_options_are_rejected(self):
        run_dir = self.root / "run"
        cases = {
            "alternatives": ["--resume", "--resume-from-hf", "--hf-adapter-repo", "u/a",
                             "--hf-run-id", "x"],
            "requires --hf-adapter-repo and --hf-run-id": ["--resume-from-hf", "--hf-adapter-repo", "u/a"],
            "only apply with --resume-from-hf": ["--hf-run-id", "x"],
            "cannot be negative": ["--resume-from-hf", "--hf-adapter-repo", "u/a",
                                   "--hf-run-id", "x", "--hf-version", "-1"],
            "--keep-checkpoints must be at least 1": ["--keep-checkpoints", "0"],
            "requires --push-adapter-to-hf or --resume-from-hf": ["--hf-adapter-repo", "u/a"],
        }
        for expected, extra in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, self.refused(run_dir, *extra))
        self.assertFalse(run_dir.exists(), "rejected options must not create a run directory")

    def test_resume_reconciles_hub_pushes_and_sets_aside_the_interrupted_rounds_gradients(self):
        run_dir = self.root / "run"
        push = ["--push-adapter-to-hf", "--hf-adapter-repo", "u/a", "--hf-token", "write"]
        self.run_main(run_dir, *push)
        leftover = run_dir / "gradients" / "round-000001"
        leftover.mkdir(parents=True)
        (leftover / "x.safetensors").write_bytes(b"from before the crash")

        calls = []

        def listing(**kwargs):
            calls.append(kwargs)
            return []

        with patch.object(coordinator_module, "list_hub_checkpoint_versions", listing):
            status, printed = self.run_main(run_dir, "--resume", *push)

        self.assertEqual(calls[0]["repo_id"], "u/a")
        self.assertEqual(calls[0]["run_id"], status["run_id"])
        self.assertIn("hub_push_recovery", printed)
        self.assertIn("Set aside the interrupted round's gradient files", printed)
        self.assertFalse(leftover.exists())
        self.assertEqual(len(list((run_dir / "gradients").glob("round-000001.discarded-*"))), 1)
        self.assertNotIn("WARNING", printed)

    def test_resume_warns_when_the_hub_is_ahead_or_unreachable(self):
        run_dir = self.root / "run"
        push = ["--push-adapter-to-hf", "--hf-adapter-repo", "u/a", "--hf-token", "write"]
        self.run_main(run_dir, *push)

        with patch.object(coordinator_module, "list_hub_checkpoint_versions", lambda **kw: [7]):
            _, printed = self.run_main(run_dir, "--resume", *push)
        self.assertIn("already holds newer versions", printed)
        self.assertIn("[7]", printed)

        def unreachable(**kwargs):
            raise ConnectionError("offline")

        with patch.object(coordinator_module, "list_hub_checkpoint_versions", unreachable):
            _, printed = self.run_main(run_dir, "--resume", *push)
        self.assertIn("could not list the Hub repository", printed)

    def test_resume_without_hub_pushing_never_contacts_the_hub(self):
        run_dir = self.root / "run"
        self.run_main(run_dir)

        def forbidden(**kwargs):
            raise AssertionError("the Hub must not be contacted")

        with patch.object(coordinator_module, "list_hub_checkpoint_versions", forbidden):
            status, printed = self.run_main(run_dir, "--resume")
        self.assertTrue(status["resumed"])
        self.assertNotIn("hub_push_recovery", printed)

    def test_gradient_retention_option_is_validated_and_reported(self):
        self.assertIn("--keep-gradient-rounds", self.refused(self.root / "bad", "--keep-gradient-rounds", "-2"))
        status, _ = self.run_main(self.root / "ok", "--keep-gradient-rounds", "2")
        self.assertEqual(status["keep_gradient_rounds"], 2)
        status, _ = self.run_main(self.root / "default")
        self.assertEqual(status["keep_gradient_rounds"], 1)
        status, _ = self.run_main(self.root / "all", "--keep-gradient-rounds", "-1")
        self.assertEqual(status["keep_gradient_rounds"], -1)

    def test_unknown_optimizer_policy_is_rejected_by_the_parser(self):
        self.assertIn("invalid choice", self.refused(self.root / "run", "--hf-push-optimizer", "maybe"))

    def test_optimizer_policy_is_reported_and_public_upload_is_called_out(self):
        push = ["--push-adapter-to-hf", "--hf-adapter-repo", "u/a", "--hf-token", "write"]
        cases = [
            # (extra flags, policy reported, text that must appear, text that must not)
            (["--hf-push-optimizer", "always", "--hf-adapter-public"], "always",
             "uploads AdamW moments", None),
            (["--hf-adapter-public"], "auto", "withheld from public repositories", "uploads AdamW"),
            ([], "auto", None, "WARNING"),
            (["--hf-push-optimizer", "never"], "never", None, "WARNING"),
        ]
        for index, (flags, policy, expected, forbidden) in enumerate(cases):
            with self.subTest(flags=flags):
                status, printed = self.run_main(self.root / f"run-{index}", *push, *flags)
                self.assertEqual(status["adapter_hub_push"]["optimizer_state"], policy)
                if expected:
                    self.assertIn(expected, printed)
                if forbidden:
                    self.assertNotIn(forbidden, printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
