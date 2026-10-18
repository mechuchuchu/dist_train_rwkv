from __future__ import annotations

import json
import base64
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import torch

from g1j_dist_coordinator import (
    Application,
    build_coordinator,
    build_round_plan,
    make_handler,
    round_plan_digest,
)
from rwkv_dist_checkpoint import (
    ResumeError,
    checkpoint_name,
    list_checkpoint_versions,
    load_checkpoint,
    load_latest_checkpoint,
)
from rwkv_dist_protocol import (
    RoundCoordinator,
    decode_tensors,
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
                    self.assertIn("error", saved["hub_push"]["versions"]["000001"])
                    app.retry_hub_pushes()
                    app.hub_push_queue.join()

            self.assertEqual(push.call_count, 4)
            self.assertEqual(app.hub_push_status["status"], "uploaded")
            # The first attempts' error must not follow the version into "uploaded".
            self.assertNotIn("error", app.hub_push_status)
            saved = json.loads((app.run_dir / "coordinator_state.json").read_text())
            self.assertEqual(saved["hub_push"]["status"], "uploaded")
            self.assertNotIn("error", saved["hub_push"])
            self.assertEqual(saved["hub_push"]["versions"]["000001"]["status"], "uploaded")
            self.assertNotIn("error", saved["hub_push"]["versions"]["000001"])


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


class SnapshotEndpointTests(unittest.TestCase):
    def test_headers_describe_the_bytes_that_were_returned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "initial-adapter"
            config_dir.mkdir()
            (config_dir / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "/local/base", "inference_mode": True,
            }))
            coordinator = make_coordinator()
            app = Application(
                coordinator,
                token="protocol-token",
                run_dir=root / "run",
                adapter_config_dir=config_dir,
                manifest={"manifest_sha256": "d" * 64},
                auto_training_complete_on_round_close=True,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"

            def fetch_snapshot():
                request = Request(base_url + "/snapshot", headers={
                    "Authorization": "Bearer protocol-token",
                })
                with urlopen(request, timeout=5) as response:
                    return (response.read(), response.headers["X-Theta-Hash"],
                            response.headers["X-Theta-Version"])

            try:
                body, header_hash, header_version = fetch_snapshot()
                self.assertEqual(header_version, "0")
                self.assertEqual(header_hash, coordinator.theta_hash)
                self.assertEqual(header_hash, tensor_map_sha256(decode_tensors(body)))

                # After the adapter advances, the pair must advance together.
                with coordinator.lock:
                    for tensor in coordinator.theta.values():
                        tensor.add_(1)
                    coordinator.theta_version = 1
                    coordinator.theta_hash = tensor_map_sha256(coordinator.theta)
                body, header_hash, header_version = fetch_snapshot()
                self.assertEqual(header_version, "1")
                self.assertEqual(header_hash, coordinator.theta_hash)
                self.assertEqual(header_hash, tensor_map_sha256(decode_tensors(body)))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


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


THETA_NAMES = ("m.layers.0.att.receptance.lora_A.default.weight",
               "m.layers.0.att.receptance.lora_B.default.weight")
RESUME_MANIFEST = {"manifest_sha256": "d" * 64, "tokenizer": {"name": "test-tokenizer"}}
RESUME_BASE = {"repo": "RWKV/test", "revision": "a" * 40}


def initial_theta() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {name: torch.randn((2, 3), generator=generator) for name in THETA_NAMES}


def make_plan(unit_count: int = 6, units_per_step: int = 2) -> list[dict]:
    units = []
    for index in range(unit_count):
        unit = {"unit_id": f"u{index}", "input_ids": [1, 2], "labels": [-100, 2],
                "loss_tokens": 1, "total_tokens": 2}
        unit["unit_sha256"] = unit_digest(unit)
        units.append(unit)
    return build_round_plan(units, epochs=1, units_per_step=units_per_step,
                            min_units_per_step=units_per_step)


def gradient_for(unit_id: str, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A gradient fixed by the unit alone, so any schedule of workers gives the same sums.

    It grows with the unit's position and varies across elements, so successive
    optimizer steps see different gradients and Adam's moments actually matter.
    """
    position = int(unit_id.rsplit("u", 1)[1])
    return {
        name: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape) * 0.01
        + 0.1 * (position + 1)
        for name, value in theta.items()
    }


def open_run(root: Path, *, loaded=None, learning_rate: float = 0.05, plan=None,
             manifest=None, keep: int = 3) -> Application:
    """Start a coordinator the way main() does: fresh, or resumed from `loaded`."""
    plan = plan or make_plan()
    manifest = manifest or RESUME_MANIFEST
    config_dir = root / "initial-adapter"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "/local/base", "inference_mode": True,
    }))
    plan_sha256 = round_plan_digest(plan)
    coordinator, start = build_coordinator(
        round_plan=plan, plan_sha256=plan_sha256, manifest=manifest, base=RESUME_BASE,
        worker_code_sha256="c" * 64, lease_seconds=60, learning_rate=learning_rate,
        new_run_id="run-resume", initial_theta=None if loaded else initial_theta(),
        loaded=loaded,
    )
    app = Application(
        coordinator, token="protocol-token", run_dir=root / "run",
        adapter_config_dir=config_dir, manifest=manifest,
        auto_training_complete_on_round_close=True, round_plan=plan,
        start_round_index=start, checkpoint_keep=keep, round_plan_sha256=plan_sha256,
    )
    app.persist_checkpoint()
    return app


class Served:
    """An Application behind a real HTTP server, driven the way workers drive it."""

    def __init__(self, app: Application):
        self.app = app
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def get_json(self, path: str) -> dict:
        request = Request(self.base_url + path, headers={"Authorization": "Bearer protocol-token"})
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def post_result(self, worker_id: str, lease: dict) -> tuple[int, dict]:
        coordinator = self.app.coordinator
        unit = lease["unit"]
        payload = encode_tensors(gradient_for(unit["unit_id"], coordinator.theta))
        metadata = {
            "unit_id": unit["unit_id"], "unit_sha256": unit["unit_sha256"],
            "lease_id": lease["lease_id"], "worker_id": worker_id,
            "round_id": lease["round_id"], "theta_version": lease["theta_version"],
            "theta_hash": lease["theta_hash"],
            "worker_code_sha256": coordinator.worker_code_sha256,
            "loss_tokens": unit["loss_tokens"], "loss_sum": 1.0,
            "gradient_sha256": sha256_bytes(payload),
        }
        request = Request(
            self.base_url + "/result", data=payload, method="POST",
            headers={
                "Authorization": "Bearer protocol-token",
                "Content-Type": "application/octet-stream",
                "X-Result-Metadata": base64.urlsafe_b64encode(json.dumps(metadata).encode()).decode(),
            },
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def work(self, worker_id: str, *, max_units: int | None = None) -> list[tuple[int, dict]]:
        """Lease and submit units until the round closes or `max_units` were sent."""
        self.get_json(f"/join?worker_id={worker_id}")
        results = []
        while max_units is None or len(results) < max_units:
            lease = self.get_json(f"/lease?worker_id={worker_id}")
            if lease["status"] != "leased":
                break
            status, body = self.post_result(worker_id, lease)
            results.append((status, body))
            if body.get("round_closed"):
                break
        return results

    def finish(self, worker_id: str = "w") -> None:
        for _ in range(len(self.app.round_plan) + 1):
            if self.app.coordinator.training_complete:
                return
            self.work(worker_id)
        raise AssertionError("run did not complete")


class CoordinatorResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def ckpt_root(self, name: str) -> Path:
        return self.root / name / "run" / "ckpt"

    def run_to_end(self, name: str, **kwargs) -> Application:
        app = open_run(self.root / name, **kwargs)
        with Served(app) as served:
            served.finish()
        return app

    def assert_same_state(self, got: Application, want: Application) -> None:
        got_theta, got_optimizer, got_version, got_hash = got.coordinator.export_state()
        want_theta, want_optimizer, want_version, want_hash = want.coordinator.export_state()
        self.assertEqual(got_version, want_version)
        self.assertEqual(got_hash, want_hash)
        for name, value in want_theta.items():
            self.assertTrue(torch.equal(got_theta[name], value), name)
        self.assertEqual(set(got_optimizer), set(want_optimizer))
        for name, value in want_optimizer.items():
            self.assertTrue(torch.equal(got_optimizer[name], value), name)

    def crash_after_first_round(self, name: str) -> Application:
        app = open_run(self.root / name)
        with Served(app) as served:
            served.work("w")
        self.assertEqual(app.coordinator.theta_version, 1)
        return app

    def test_resume_after_a_closed_round_matches_an_uninterrupted_run_exactly(self):
        reference = self.run_to_end("reference")
        before = self.crash_after_first_round("resumed")

        loaded = load_latest_checkpoint(self.ckpt_root("resumed"))
        self.assertEqual(loaded.state["theta_version"], 1)
        after = open_run(self.root / "resumed", loaded=loaded)

        self.assertEqual(after.coordinator.run_id, before.coordinator.run_id)
        self.assertEqual((after.coordinator.round_id, after.coordinator.theta_version), (2, 1))
        self.assertEqual(after.round_index, 1)
        with Served(after) as served:
            served.finish()
        self.assertTrue(after.coordinator.training_complete)
        self.assert_same_state(after, reference)

    def test_resume_without_optimizer_state_diverges_from_the_uninterrupted_run(self):
        # The moments carry information between steps; dropping them (as when a
        # public repository did not receive optimizer.safetensors) changes the result.
        reference = self.run_to_end("reference")
        self.crash_after_first_round("resumed")
        loaded = load_latest_checkpoint(self.ckpt_root("resumed"))
        loaded.optimizer_tensors = {}

        after = open_run(self.root / "resumed", loaded=loaded)
        with Served(after) as served:
            served.finish()

        self.assertNotEqual(after.coordinator.theta_hash, reference.coordinator.theta_hash)
        self.assertEqual(after.coordinator.theta_version, reference.coordinator.theta_version)

    def test_round_in_flight_at_the_crash_is_discarded_and_reissued(self):
        reference = self.run_to_end("reference")
        crashed = open_run(self.root / "resumed")
        with Served(crashed) as served:
            served.work("w")                                   # round 1 closes -> version 1
            served.get_json("/join?worker_id=w2")
            first = served.work("w1", max_units=1)             # one unit of round 2 accepted
            self.assertEqual(first[0][1]["received_units"], 1)
            held = served.get_json("/lease?worker_id=w2")      # the other unit is leased, unfinished
            self.assertEqual(held["status"], "leased")
        self.assertEqual(crashed.coordinator.theta_version, 1)

        loaded = load_latest_checkpoint(self.ckpt_root("resumed"))
        resumed = open_run(self.root / "resumed", loaded=loaded)

        coordinator = resumed.coordinator
        self.assertEqual((coordinator.round_id, coordinator.theta_version), (2, 1))
        self.assertEqual(coordinator.accepted, {})
        self.assertEqual(coordinator.leases, {})
        self.assertEqual(coordinator.loss_tokens, 0)
        self.assertEqual(set(coordinator.unit_status.values()), {"pending"})
        self.assertEqual(len(coordinator.pending), 2)
        with Served(resumed) as served:
            # The unit finished before the crash is not remembered...
            self.assertEqual(served.get_json("/round")["received_units"], 0)
            # ...and the worker that held a lease from before the restart is refused.
            status, body = served.post_result("w2", held)
            self.assertEqual((status, body["error"]), (409, "unknown_lease"))
            self.assertEqual(coordinator.accepted, {})
            served.finish("w3")
        self.assert_same_state(resumed, reference)

        events = [json.loads(line) for line in
                  (self.root / "resumed" / "run" / "round_log.jsonl").read_text().splitlines()]
        resumes = [event for event in events if event["event"] == "coordinator_resumed"]
        self.assertEqual(len(resumes), 1)
        self.assertEqual(resumes[0]["from_theta_version"], 1)
        self.assertTrue(resumes[0]["optimizer_restored"])
        self.assertTrue(resumes[0]["in_flight_round_discarded"])

    def test_crash_between_closing_a_round_and_opening_the_next_resumes_at_the_next_round(self):
        reference = self.run_to_end("reference")
        crashed = open_run(self.root / "resumed")
        with patch.object(Application, "open_next_round", side_effect=RuntimeError("killed")):
            with Served(crashed) as served:
                served.work("w")
        # The adapter step was committed before the failure, so it survives.
        self.assertEqual(list_checkpoint_versions(self.ckpt_root("resumed")), [1, 0])

        resumed = open_run(self.root / "resumed",
                           loaded=load_latest_checkpoint(self.ckpt_root("resumed")))
        self.assertEqual(resumed.coordinator.round_id, 2)
        with Served(resumed) as served:
            served.finish()
        self.assert_same_state(resumed, reference)

    def test_resume_falls_back_to_an_older_version_when_the_newest_is_damaged(self):
        self.run_to_end("run-a", keep=10)
        newest = self.ckpt_root("run-a") / checkpoint_name(3) / "optimizer.safetensors"
        data = bytearray(newest.read_bytes())
        data[-1] ^= 0xFF
        newest.write_bytes(bytes(data))

        loaded = load_latest_checkpoint(self.ckpt_root("run-a"))
        self.assertEqual(loaded.state["theta_version"], 2)
        resumed = open_run(self.root / "run-a", loaded=loaded)
        self.assertEqual((resumed.coordinator.round_id, resumed.round_index), (3, 2))
        events = [e for e in resumed.coordinator.events if e["event"] == "coordinator_resumed"]
        self.assertEqual(len(events[0]["skipped_checkpoints"]), 1)

    def test_a_checkpoint_from_a_different_configuration_is_refused(self):
        open_run(self.root / "origin")
        loaded = load_latest_checkpoint(self.ckpt_root("origin"))
        cases = {
            "hyperparameters": {"learning_rate": 0.5},
            "data_manifest_sha256": {"manifest": {**RESUME_MANIFEST, "manifest_sha256": "e" * 64}},
            "round_plan_sha256": {"plan": make_plan(units_per_step=3)},
            "tokenizer": {"manifest": {**RESUME_MANIFEST, "tokenizer": {"name": "other"}}},
        }
        for expected, kwargs in cases.items():
            with self.subTest(changed=expected):
                with self.assertRaisesRegex(ResumeError, expected):
                    open_run(self.root / f"refused-{expected}", loaded=loaded, **kwargs)

    def test_a_finished_run_cannot_be_resumed(self):
        finished = self.run_to_end("finished")
        self.assertTrue(finished.coordinator.training_complete)
        loaded = load_latest_checkpoint(self.ckpt_root("finished"))
        self.assertEqual(loaded.state["theta_version"], len(finished.round_plan))

        with self.assertRaisesRegex(ResumeError, "already finished"):
            open_run(self.root / "again", loaded=loaded)

    def test_every_adapter_version_is_checkpointed_and_old_ones_are_pruned(self):
        self.run_to_end("everything", keep=10)
        self.assertEqual(list_checkpoint_versions(self.ckpt_root("everything")), [3, 2, 1, 0])
        for version in range(4):
            loaded = load_checkpoint(self.ckpt_root("everything") / checkpoint_name(version))
            self.assertEqual(loaded.state["run_id"], "run-resume")
            self.assertEqual(loaded.state["planned_rounds"], 3)

        self.run_to_end("pruned", keep=2)
        self.assertEqual(list_checkpoint_versions(self.ckpt_root("pruned")), [3, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
