from __future__ import annotations

import json
import base64
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch

import torch

from g1j_dist_coordinator import Application, build_round_plan, make_handler
from rwkv_dist_protocol import (
    RoundCoordinator,
    encode_tensors,
    sha256_bytes,
    tensor_map_sha256,
    unit_digest,
)


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
            hf_push_every_steps=10,
        )

    def test_pushes_once_per_adapter_version_without_blocking_on_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir)
            with patch("g1j_dist_coordinator.push_adapter_snapshot", return_value="commit-url") as push:
                app.persist_checkpoint()
                app.coordinator.theta_version = 9
                self.assertEqual(app.start_hub_push()["status"], "pending")
                self.assertTrue(app.hub_push_queue.empty())
                app.coordinator.theta_version = 10
                status = app.start_hub_push()
                self.assertEqual(status["status"], "queued")
                app.hub_push_queue.join()
                app.start_hub_push()
                self.assertEqual(push.call_count, 1)
                self.assertEqual(push.call_args.kwargs["path_in_repo"],
                                 "checkpoints/test-run/theta-000010")

                name = "layer.att.lora_A.default.weight"
                app.coordinator.theta[name].add_(1)
                app.coordinator.theta_version = 11
                app.coordinator.theta_hash = tensor_map_sha256(app.coordinator.theta)
                app.persist_checkpoint()
                app.start_hub_push(final=True)
                app.hub_push_queue.join()

            self.assertEqual(push.call_count, 2)
            self.assertEqual(push.call_args.kwargs["path_in_repo"],
                             "checkpoints/test-run/theta-000011")
            saved = json.loads((app.run_dir / "coordinator_state.json").read_text())
            self.assertEqual(saved["hub_push"]["status"], "uploaded")
            self.assertEqual(sorted(saved["hub_push"]["versions"]), ["000010", "000011"])

    def test_push_failure_is_recorded_without_stopping_local_checkpointing_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(temp_dir)
            with patch("g1j_dist_coordinator.push_adapter_snapshot",
                       side_effect=[RuntimeError("bad write-secret credential")] * 3 + ["commit-url"]) as push:
                with patch("g1j_dist_coordinator.time.sleep"):
                    app.coordinator.theta_version = 1
                    app.persist_checkpoint()
                    app.start_hub_push(force=True)
                    app.hub_push_queue.join()
                    self.assertTrue((app.run_dir / "adapter_model.safetensors").is_file())
                    saved = json.loads((app.run_dir / "coordinator_state.json").read_text())
                    self.assertEqual(saved["hub_push"]["status"], "failed")
                    self.assertNotIn("write-secret", json.dumps(saved))
                    app.retry_hub_pushes()
                    app.hub_push_queue.join()

            self.assertEqual(push.call_count, 4)
            self.assertEqual(app.hub_push_status["status"], "uploaded")


class RoundPlanTests(unittest.TestCase):
    def test_epochs_and_unit_chunks_create_unique_hashed_steps(self):
        base_units = []
        for index, tokens in enumerate((2, 3, 5)):
            unit = {
                "unit_id": f"u{index}", "input_ids": [1, 2], "labels": [-100, 2],
                "loss_tokens": tokens, "total_tokens": 2,
            }
            unit["unit_sha256"] = unit_digest(unit)
            base_units.append(unit)

        plan = build_round_plan(
            base_units, epochs=2, units_per_step=2, min_units_per_step=2,
        )

        self.assertEqual(len(plan), 4)
        self.assertEqual([step["target_loss_tokens"] for step in plan], [5, 5, 5, 5])
        ids = [unit["unit_id"] for step in plan for unit in step["units"]]
        self.assertEqual(len(ids), len(set(ids)))
        for step in plan:
            for unit in step["units"]:
                self.assertEqual(unit["unit_sha256"], unit_digest(unit))


class MultiStepHTTPTests(unittest.TestCase):
    def test_late_worker_waits_until_next_step_snapshot_is_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "initial-adapter"
            config_dir.mkdir()
            (config_dir / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "/local/base", "inference_mode": True,
            }))
            first_unit = {
                "unit_id": "unit-0", "input_ids": [1, 2],
                "labels": [-100, 2], "loss_tokens": 1, "total_tokens": 2,
            }
            first_unit["unit_sha256"] = unit_digest(first_unit)
            second_unit = {**first_unit, "unit_id": "unit-1"}
            second_unit["unit_sha256"] = unit_digest(second_unit)
            coordinator = make_coordinator()
            app = Application(
                coordinator,
                token="protocol-token",
                run_dir=root / "run",
                adapter_config_dir=config_dir,
                manifest={"manifest_sha256": "d" * 64},
                auto_training_complete_on_round_close=True,
                round_plan=[
                    {"units": [first_unit], "target_loss_tokens": 1, "min_units": 1},
                    {"units": [second_unit], "target_loss_tokens": 1, "min_units": 1},
                ],
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"

            def get_json(path: str) -> dict:
                request = Request(base_url + path, headers={
                    "Authorization": "Bearer protocol-token",
                })
                with urlopen(request, timeout=5) as response:
                    return json.loads(response.read())

            def submit_result(worker_id: str, lease: dict) -> dict:
                payload = encode_tensors({
                    name: torch.ones_like(value)
                    for name, value in coordinator.theta.items()
                })
                unit = lease["unit"]
                metadata = {
                    "unit_id": unit["unit_id"],
                    "unit_sha256": unit["unit_sha256"],
                    "lease_id": lease["lease_id"],
                    "worker_id": worker_id,
                    "round_id": lease["round_id"],
                    "theta_version": lease["theta_version"],
                    "theta_hash": lease["theta_hash"],
                    "worker_code_sha256": coordinator.worker_code_sha256,
                    "loss_tokens": unit["loss_tokens"],
                    "loss_sum": 1.0,
                    "gradient_sha256": sha256_bytes(payload),
                }
                encoded = base64.urlsafe_b64encode(
                    json.dumps(metadata).encode()
                ).decode()
                request = Request(
                    base_url + "/result",
                    data=payload,
                    headers={
                        "Authorization": "Bearer protocol-token",
                        "Content-Type": "application/octet-stream",
                        "X-Result-Metadata": encoded,
                    },
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    return json.loads(response.read())

            try:
                self.assertEqual(get_json("/join?worker_id=early")["status"], "active")
                lease1 = get_json("/lease?worker_id=early")
                self.assertEqual(lease1["status"], "leased")
                late_join = get_json("/join?worker_id=late")
                self.assertEqual(late_join["status"], "waiting_next_round")

                closed1 = submit_result("early", lease1)
                self.assertTrue(closed1["round_closed"])
                self.assertEqual(closed1["next_round"]["round_id"], 2)
                self.assertEqual(closed1["next_round"]["theta_version"], 1)

                self.assertEqual(get_json("/join?worker_id=late")["status"], "active")
                lease2 = get_json("/lease?worker_id=late")
                self.assertEqual(lease2["status"], "leased")
                self.assertEqual(lease2["round_id"], 2)
                self.assertEqual(lease2["theta_version"], 1)
                closed2 = submit_result("late", lease2)
                self.assertTrue(closed2["round_closed"])
                self.assertTrue(closed2["training_complete"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
