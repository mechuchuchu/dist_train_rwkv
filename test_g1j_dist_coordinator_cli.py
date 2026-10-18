"""Drive g1j_dist_coordinator.main() from the command line on small fixtures.

The HTTP server loop is replaced by one that returns immediately, so each call
performs the real startup (argument checks, manifest and model checks, resume
or fresh start, checkpoint commit, status line) and then exits. This covers the
wiring between the CLI flags and the pieces that are unit-tested separately.
"""

from __future__ import annotations

import io
import json
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


def write_initial_adapter(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "r": 8, "lora_alpha": 16, "base_model_name_or_path": "/local/base",
    }))
    tensors = {}
    for layer in range(24):
        for module in ("receptance", "key", "value", "output"):
            prefix = f"base_model.model.model.layers.{layer}.att.{module}"
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros((8, 2048))
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros((2048, 8))
    save_file(tensors, str(directory / "adapter_model.safetensors"))


def write_fixtures(root: Path) -> dict[str, Path]:
    model_dir = root / "model"
    model_dir.mkdir()
    release = json.dumps({"conversion": {"serialized_parameter_count": 123}}).encode()
    (model_dir / "release-manifest.json").write_bytes(release)
    units = []
    for index in range(6):
        unit = {"unit_id": f"u{index}", "input_ids": [1, 2], "labels": [-100, 2],
                "loss_tokens": 1, "total_tokens": 2}
        unit["unit_sha256"] = unit_digest(unit)
        units.append(unit)
    content = {
        "model": {
            "repo": coordinator_module.MODEL_ID,
            "revision": coordinator_module.MODEL_REVISION,
            "release_manifest_sha256": sha256_bytes(release),
        },
        "tokenizer": {"name": "test-tokenizer"},
        "units": units,
    }
    manifest = {**content, "manifest_sha256": sha256_bytes(canonical_json(content))}
    units_path = root / "units.json"
    units_path.write_text(json.dumps(manifest))
    adapter_dir = root / "initial-adapter"
    write_initial_adapter(adapter_dir)
    return {"units": units_path, "model": model_dir, "adapter": adapter_dir}


class CoordinatorCommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = write_fixtures(self.root)

    def run_main(self, run_dir: Path, *extra: str) -> tuple[dict, str]:
        """Run main() and return (the status line it printed, everything else it printed)."""
        argv = [
            "g1j_dist_coordinator.py", "--units", str(self.fixtures["units"]),
            "--model-dir", str(self.fixtures["model"]),
            "--initial-adapter-dir", str(self.fixtures["adapter"]),
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

    def refused(self, run_dir: Path, *extra: str) -> str:
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            self.run_main(run_dir, *extra)
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
