from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from g1j_dist_coordinator import Application
from rwkv_dist_protocol import RoundCoordinator, tensor_map_sha256, unit_digest


def make_coordinator() -> RoundCoordinator:
    unit = {
        "unit_id": "unit-0",
        "input_ids": [1, 2],
        "labels": [-100, 2],
        "loss_tokens": 1,
        "total_tokens": 2,
    }
    unit["unit_sha256"] = unit_digest(unit)
    return RoundCoordinator(
        theta={"layer.att.lora_A.default.weight": torch.ones((1, 1))},
        units=[unit],
        run_id="test-run",
        base={"repo": "RWKV/test", "revision": "a" * 40},
        tokenizer={"name": "test-tokenizer"},
        worker_code_sha256="c" * 64,
        target_loss_tokens=1,
        min_units=1,
        lease_seconds=60,
        learning_rate=1e-4,
    )


class CoordinatorHubPushTests(unittest.TestCase):
    def make_app(self, temp_dir: str) -> Application:
        root = Path(temp_dir)
        config_dir = root / "initial-adapter"
        config_dir.mkdir()
        (config_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "/local/base",
            "inference_mode": True,
        }))
        return Application(
            make_coordinator(),
            token="protocol-token",
            run_dir=root / "run",
            adapter_config_dir=config_dir,
            manifest={"manifest_sha256": "d" * 64},
            auto_training_complete_on_round_close=True,
            hf_adapter_repo="user/adapter",
            hf_token="write-secret",
        )

    def test_pushes_once_per_adapter_version_without_blocking_on_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir)
            with patch("g1j_dist_coordinator.push_adapter_snapshot", return_value="commit-url") as push:
                app.persist_checkpoint()
                status = app.start_hub_push()
                self.assertEqual(status["status"], "queued")
                app.hub_push_thread.join(timeout=5)
                app.start_hub_push()
                self.assertEqual(push.call_count, 1)
                self.assertEqual(push.call_args.kwargs["path_in_repo"],
                                 "checkpoints/test-run/theta-000000")

                name = "layer.att.lora_A.default.weight"
                app.coordinator.theta[name].add_(1)
                app.coordinator.theta_version += 1
                app.coordinator.theta_hash = tensor_map_sha256(app.coordinator.theta)
                app.persist_checkpoint()
                app.start_hub_push()
                app.hub_push_thread.join(timeout=5)

            self.assertEqual(push.call_count, 2)
            self.assertEqual(push.call_args.kwargs["path_in_repo"],
                             "checkpoints/test-run/theta-000001")
            saved = json.loads((app.run_dir / "coordinator_state.json").read_text())
            self.assertEqual(saved["hub_push"]["status"], "uploaded")

    def test_push_failure_is_recorded_without_stopping_local_checkpointing_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir)
            with patch("g1j_dist_coordinator.push_adapter_snapshot",
                       side_effect=[RuntimeError("bad write-secret credential"), "commit-url"]) as push:
                app.persist_checkpoint()
                app.start_hub_push()
                app.hub_push_thread.join(timeout=5)
                self.assertTrue((app.run_dir / "adapter_model.safetensors").is_file())
                saved = json.loads((app.run_dir / "coordinator_state.json").read_text())
                self.assertEqual(saved["hub_push"]["status"], "failed")
                self.assertNotIn("write-secret", json.dumps(saved))
                app.start_hub_push()
                app.hub_push_thread.join(timeout=5)

            self.assertEqual(push.call_count, 2)
            self.assertEqual(app.hub_push_status["status"], "uploaded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
