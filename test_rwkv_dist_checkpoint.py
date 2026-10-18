from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import rwkv_dist_checkpoint as ckpt
from rwkv_dist_checkpoint import (
    ResumeError,
    check_resume_compatibility,
    checkpoint_name,
    file_sha256,
    list_checkpoint_versions,
    load_checkpoint,
    load_hub_checkpoint,
    load_latest_checkpoint,
    write_checkpoint,
)
from rwkv_dist_protocol import parameter_name_to_peft_saved_name, tensor_map_sha256


THETA_NAMES = ("m.layers.0.att.receptance.lora_A.default.weight",
               "m.layers.0.att.receptance.lora_B.default.weight")


def make_theta(seed: float = 1.0) -> dict[str, torch.Tensor]:
    return {name: torch.full((2, 3), seed + index) for index, name in enumerate(THETA_NAMES)}


def make_optimizer(seed: float = 0.5) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in THETA_NAMES:
        tensors[f"exp_avg/{name}"] = torch.full((2, 3), seed)
        tensors[f"exp_avg_sq/{name}"] = torch.full((2, 3), seed * 2)
        tensors[f"step/{name}"] = torch.tensor(3.0)
    return tensors


def make_state(version: int, theta: dict[str, torch.Tensor], **overrides) -> dict:
    state = {
        "run_id": "run-1",
        "theta_version": version,
        "theta_hash": tensor_map_sha256(theta),
        "planned_rounds": 4,
        "round_plan_sha256": "p" * 64,
        "data_manifest_sha256": "d" * 64,
        "base": {"revision": "pinned"},
        "tokenizer": {"name": "tok"},
        "hyperparameters": {"lr": 0.01},
    }
    state.update(overrides)
    return state


def save_version(root: Path, version: int, *, keep: int = 10, optimizer=None) -> Path:
    theta = make_theta(float(version))
    return write_checkpoint(
        root, state=make_state(version, theta), theta=theta,
        optimizer_tensors=make_optimizer() if optimizer is None else optimizer, keep=keep,
    )


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


class WriteAndLoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "ckpt"

    def test_round_trip_preserves_adapter_optimizer_and_state(self):
        directory = save_version(self.root, 3)
        self.assertEqual(directory.name, "theta-000003")

        loaded = load_latest_checkpoint(self.root)

        self.assertEqual(loaded.state["theta_version"], 3)
        self.assertEqual(loaded.state["run_id"], "run-1")
        self.assertEqual(loaded.skipped, [])
        for name, value in make_theta(3.0).items():
            self.assertTrue(torch.equal(loaded.theta[name], value))
        for name, value in make_optimizer().items():
            self.assertTrue(torch.equal(loaded.optimizer_tensors[name], value), name)
        self.assertEqual(loaded.optimizer_tensors[f"step/{THETA_NAMES[0]}"].shape, torch.Size([]))
        self.assertEqual(set(loaded.state["files"]), {"theta.safetensors", "optimizer.safetensors"})

    def test_a_version_without_optimizer_state_round_trips_as_empty(self):
        save_version(self.root, 0, optimizer={})
        loaded = load_latest_checkpoint(self.root)
        self.assertEqual(loaded.optimizer_tensors, {})

    def test_an_existing_version_is_never_rewritten(self):
        directory = save_version(self.root, 2)
        before = file_sha256(directory / "theta.safetensors")
        theta = make_theta(99.0)
        write_checkpoint(self.root, state=make_state(2, theta), theta=theta,
                         optimizer_tensors=make_optimizer(), keep=10)
        self.assertEqual(file_sha256(directory / "theta.safetensors"), before)

    def test_only_the_newest_versions_are_kept(self):
        for version in range(5):
            save_version(self.root, version, keep=2)
        self.assertEqual(list_checkpoint_versions(self.root), [4, 3])

    def test_keep_must_be_positive(self):
        with self.assertRaises(ValueError):
            save_version(self.root, 1, keep=0)

    def test_no_checkpoints_is_an_error(self):
        with self.assertRaisesRegex(ResumeError, "no checkpoints"):
            load_latest_checkpoint(self.root)


class CrashAndCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "ckpt"
        save_version(self.root, 1)
        save_version(self.root, 2)

    def test_corrupt_newest_falls_back_to_the_previous_version(self):
        flip_last_byte(self.root / checkpoint_name(2) / "theta.safetensors")

        loaded = load_latest_checkpoint(self.root)

        self.assertEqual(loaded.state["theta_version"], 1)
        self.assertEqual(len(loaded.skipped), 1)
        self.assertIn("theta-000002", loaded.skipped[0])
        self.assertIn("recorded hash", loaded.skipped[0])

    def test_corrupt_optimizer_file_is_detected(self):
        flip_last_byte(self.root / checkpoint_name(2) / "optimizer.safetensors")
        self.assertEqual(load_latest_checkpoint(self.root).state["theta_version"], 1)

    def test_missing_state_file_falls_back(self):
        (self.root / checkpoint_name(2) / "state.json").unlink()
        loaded = load_latest_checkpoint(self.root)
        self.assertEqual(loaded.state["theta_version"], 1)
        self.assertIn("state.json", loaded.skipped[0])

    def test_directory_renamed_to_another_version_is_rejected(self):
        (self.root / checkpoint_name(2)).rename(self.root / checkpoint_name(7))
        with self.assertRaisesRegex(ResumeError, "does not match its recorded version"):
            load_checkpoint(self.root / checkpoint_name(7))

    def test_every_version_corrupt_is_an_error_naming_each(self):
        for version in (1, 2):
            flip_last_byte(self.root / checkpoint_name(version) / "theta.safetensors")
        with self.assertRaisesRegex(ResumeError, "no usable checkpoint") as caught:
            load_latest_checkpoint(self.root)
        self.assertIn("theta-000001", str(caught.exception))
        self.assertIn("theta-000002", str(caught.exception))

    def test_crash_at_the_commit_point_leaves_no_partial_version(self):
        with patch.object(ckpt.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                save_version(self.root, 3)

        self.assertEqual(list_checkpoint_versions(self.root), [2, 1])
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".tmp-")], [])
        self.assertEqual(load_latest_checkpoint(self.root).state["theta_version"], 2)

    def test_leftover_staging_from_a_killed_process_is_ignored_then_cleaned(self):
        leftover = self.root / ".tmp-theta-000003-4242"
        leftover.mkdir()
        (leftover / "theta.safetensors").write_bytes(b"half written")

        self.assertEqual(list_checkpoint_versions(self.root), [2, 1])
        self.assertEqual(load_latest_checkpoint(self.root).state["theta_version"], 2)

        save_version(self.root, 3)
        self.assertFalse(leftover.exists())
        self.assertEqual(load_latest_checkpoint(self.root).state["theta_version"], 3)


class CompatibilityTests(unittest.TestCase):
    EXPECTED = {
        "data_manifest_sha256": "d" * 64,
        "round_plan_sha256": "p" * 64,
        "planned_rounds": 4,
        "base": {"revision": "pinned"},
        "tokenizer": {"name": "tok"},
        "hyperparameters": {"lr": 0.01},
    }

    def test_matching_configuration_passes(self):
        check_resume_compatibility(make_state(1, make_theta()), self.EXPECTED)

    def test_every_mismatch_is_named(self):
        state = make_state(1, make_theta(), round_plan_sha256="x" * 64,
                           hyperparameters={"lr": 0.5})
        with self.assertRaises(ResumeError) as caught:
            check_resume_compatibility(state, self.EXPECTED)
        message = str(caught.exception)
        self.assertIn("round_plan_sha256", message)
        self.assertIn("hyperparameters", message)
        self.assertNotIn("data_manifest_sha256", message)


class HubCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "checkpoints" / "run-1" / "theta-000004"
        self.directory.mkdir(parents=True)
        self.theta = make_theta(4.0)

    def publish(self, *, with_optimizer=True, **metadata_overrides) -> dict:
        save_file(
            {parameter_name_to_peft_saved_name(name): tensor for name, tensor in self.theta.items()},
            str(self.directory / "adapter_model.safetensors"),
        )
        metadata = make_state(4, self.theta)
        if with_optimizer:
            save_file(make_optimizer(), str(self.directory / "optimizer.safetensors"))
            metadata["optimizer_sha256"] = file_sha256(self.directory / "optimizer.safetensors")
        metadata.update(metadata_overrides)
        (self.directory / "adapter_metadata.json").write_text(json.dumps(metadata))
        return metadata

    def test_loads_adapter_and_optimizer_under_coordinator_names(self):
        self.publish()
        loaded = load_hub_checkpoint(self.directory)
        self.assertEqual(loaded.state["theta_version"], 4)
        self.assertEqual(loaded.state["run_id"], "run-1")
        self.assertEqual(set(loaded.theta), set(THETA_NAMES))
        for name, value in self.theta.items():
            self.assertTrue(torch.equal(loaded.theta[name], value))
        self.assertEqual(set(loaded.optimizer_tensors), set(make_optimizer()))

    def test_a_push_without_optimizer_still_resumes_the_adapter(self):
        self.publish(with_optimizer=False)
        loaded = load_hub_checkpoint(self.directory)
        self.assertEqual(loaded.optimizer_tensors, {})
        self.assertEqual(loaded.state["theta_version"], 4)

    def test_adapter_that_does_not_match_its_pushed_hash_is_rejected(self):
        self.publish(theta_hash="0" * 64)
        with self.assertRaisesRegex(ResumeError, "theta hash"):
            load_hub_checkpoint(self.directory)

    def test_tampered_optimizer_is_rejected(self):
        self.publish()
        flip_last_byte(self.directory / "optimizer.safetensors")
        with self.assertRaisesRegex(ResumeError, "optimizer file does not match"):
            load_hub_checkpoint(self.directory)

    def test_optimizer_without_a_recorded_hash_is_rejected(self):
        self.publish()
        metadata = json.loads((self.directory / "adapter_metadata.json").read_text())
        del metadata["optimizer_sha256"]
        (self.directory / "adapter_metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ResumeError, "optimizer file does not match"):
            load_hub_checkpoint(self.directory)

    def test_push_made_before_resume_metadata_existed_is_refused(self):
        self.publish()
        metadata = json.loads((self.directory / "adapter_metadata.json").read_text())
        del metadata["round_plan_sha256"]
        (self.directory / "adapter_metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ResumeError, "round_plan_sha256"):
            load_hub_checkpoint(self.directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
