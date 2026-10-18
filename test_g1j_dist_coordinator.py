from __future__ import annotations

import io
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

import shutil
from safetensors.torch import load_file, save_file

import g1j_dist_coordinator as coordinator_module

from g1j_dist_coordinator import (
    Application,
    build_coordinator,
    build_round_plan,
    load_resume_from_hub,
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
from test_rwkv_hub import FakeHub
from rwkv_dist_protocol import (
    PayloadFile,
    ProtocolError,
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


def logged_events(app: Application, name: str) -> list[dict]:
    """Events of one kind from round_log.jsonl (they leave memory once written)."""
    lines = (app.run_dir / "round_log.jsonl").read_text().splitlines()
    return [event for event in map(json.loads, lines) if event["event"] == name]


def open_run(root: Path, *, loaded=None, learning_rate: float = 0.05, plan=None,
             manifest=None, keep: int = 3, **app_kwargs) -> Application:
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
        **app_kwargs,
    )
    if loaded is not None:  # the same steps main() takes when resuming
        app.set_aside_in_flight_gradients()
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

    def post_result(self, worker_id: str, lease: dict, **metadata_overrides) -> tuple[int, dict]:
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
        } | metadata_overrides
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
        events = logged_events(resumed, "coordinator_resumed")
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


class CoordinatorHubResumeTests(unittest.TestCase):
    """Push checkpoints through the real upload code to a fake Hub, then resume from them."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def reference(self) -> Application:
        app = open_run(self.root / "reference")
        with Served(app) as served:
            served.finish()
        return app

    def push_first_rounds(self, *, rounds: int, repo_private, policy: str,
                          public_requested: bool = False) -> tuple[FakeHub, Application]:
        hub = FakeHub(self.root / "hub", repo_private=repo_private)
        with patch("huggingface_hub.HfApi", return_value=hub.api()):
            app = open_run(
                self.root / "origin", hf_adapter_repo="test/adapter", hf_token="write-token",
                hf_adapter_private=not public_requested, hf_push_every_steps=1,
                hf_push_optimizer=policy,
            )
            with Served(app) as served:
                for _ in range(rounds):
                    served.work("w")
            app.hub_push_queue.join()
        return hub, app

    def load_from(self, hub: FakeHub, **kwargs):
        def fake_list(*, repo_id, run_id, token):
            folder = hub.root / "checkpoints" / run_id
            if not folder.is_dir():
                return []
            return sorted((int(entry.name.split("-")[1]) for entry in folder.glob("theta-*")),
                          reverse=True)

        def fake_download(*, repo_id, run_id, version, token, local_dir):
            return hub.root / "checkpoints" / run_id / f"theta-{version:06d}"

        arguments = dict(repo_id="test/adapter", run_id="run-resume", version=None,
                         token="read-token", work_dir=self.root / "hub-download")
        arguments.update(kwargs)
        with patch("g1j_dist_coordinator.list_hub_checkpoint_versions", fake_list), \
                patch("g1j_dist_coordinator.download_hub_checkpoint", fake_download):
            return load_resume_from_hub(**arguments)

    def finish_from(self, loaded) -> Application:
        app = open_run(self.root / "restored", loaded=loaded)
        with Served(app) as served:
            served.finish()
        return app

    def test_private_repo_resume_through_the_hub_matches_an_uninterrupted_run(self):
        reference = self.reference()
        hub, origin = self.push_first_rounds(rounds=1, repo_private=True, policy="auto")
        folder = hub.root / "checkpoints" / "run-resume" / "theta-000001"
        self.assertTrue((folder / "optimizer.safetensors").is_file())

        loaded = self.load_from(hub)

        self.assertEqual(loaded.state["theta_version"], 1)
        self.assertTrue(loaded.optimizer_tensors)
        local = load_latest_checkpoint(self.root / "origin" / "run" / "ckpt")
        for name, value in local.optimizer_tensors.items():
            self.assertTrue(torch.equal(loaded.optimizer_tensors[name], value), name)
        restored = self.finish_from(loaded)
        self.assertEqual(restored.coordinator.theta_hash, reference.coordinator.theta_hash)
        status = json.loads((self.root / "origin" / "run" / "coordinator_state.json").read_text())
        self.assertTrue(status["hub_push"]["versions"]["000001"]["optimizer_uploaded"])
        self.assertEqual(status["hub_push"]["optimizer_policy"], "auto")

    def test_public_repo_under_auto_restores_the_adapter_but_not_the_moments(self):
        reference = self.reference()
        hub, _ = self.push_first_rounds(rounds=1, repo_private=False, policy="auto",
                                        public_requested=True)
        folder = hub.root / "checkpoints" / "run-resume" / "theta-000001"
        self.assertFalse((folder / "optimizer.safetensors").exists())

        loaded = self.load_from(hub)

        self.assertEqual(loaded.optimizer_tensors, {})
        self.assertEqual(loaded.state["theta_version"], 1)
        restored = self.finish_from(loaded)
        resumed_event = logged_events(restored, "coordinator_resumed")[0]
        self.assertFalse(resumed_event["optimizer_restored"])
        self.assertEqual(restored.coordinator.theta_version, reference.coordinator.theta_version)
        self.assertNotEqual(restored.coordinator.theta_hash, reference.coordinator.theta_hash)

    def test_always_publishes_the_moments_to_a_public_repo_when_asked(self):
        reference = self.reference()
        hub, _ = self.push_first_rounds(rounds=1, repo_private=False, policy="always",
                                        public_requested=True)
        loaded = self.load_from(hub)
        self.assertTrue(loaded.optimizer_tensors)
        self.assertEqual(self.finish_from(loaded).coordinator.theta_hash,
                         reference.coordinator.theta_hash)

    def test_never_keeps_the_moments_off_the_hub_even_when_private(self):
        hub, origin = self.push_first_rounds(rounds=1, repo_private=True, policy="never")
        folder = hub.root / "checkpoints" / "run-resume" / "theta-000001"
        self.assertFalse((folder / "optimizer.safetensors").exists())
        self.assertEqual(self.load_from(hub).optimizer_tensors, {})
        self.assertFalse(origin.hub_push_versions["000001"]["optimizer_uploaded"])

    def test_newest_hub_version_that_fails_verification_is_skipped(self):
        hub, _ = self.push_first_rounds(rounds=2, repo_private=True, policy="auto")
        newest = hub.root / "checkpoints" / "run-resume" / "theta-000002" / "adapter_model.safetensors"
        data = bytearray(newest.read_bytes())
        data[-1] ^= 0xFF
        newest.write_bytes(bytes(data))

        loaded = self.load_from(hub)
        self.assertEqual(loaded.state["theta_version"], 1)
        self.assertEqual(len(loaded.skipped), 1)
        self.assertIn("theta-000002", loaded.skipped[0])

        with self.assertRaisesRegex(ResumeError, "theta-000002"):
            self.load_from(hub, version=2)

    def test_a_pinned_older_version_can_be_chosen(self):
        hub, _ = self.push_first_rounds(rounds=2, repo_private=True, policy="auto")
        self.assertEqual(self.load_from(hub, version=1).state["theta_version"], 1)
        self.assertEqual(self.load_from(hub).state["theta_version"], 2)

    def test_an_unknown_run_has_nothing_to_resume(self):
        hub, _ = self.push_first_rounds(rounds=1, repo_private=True, policy="auto")
        with self.assertRaisesRegex(ResumeError, "no complete checkpoints"):
            self.load_from(hub, run_id="some-other-run")

    def test_a_folder_whose_metadata_names_another_run_or_version_is_not_trusted(self):
        import shutil
        hub, _ = self.push_first_rounds(rounds=1, repo_private=True, policy="auto")
        genuine = hub.root / "checkpoints" / "run-resume" / "theta-000001"
        shutil.copytree(genuine, hub.root / "checkpoints" / "other-run" / "theta-000001")
        shutil.copytree(genuine, hub.root / "checkpoints" / "run-resume" / "theta-000005")

        with self.assertRaisesRegex(ResumeError, "metadata says run 'run-resume' version 1"):
            self.load_from(hub, run_id="other-run")
        with self.assertRaisesRegex(ResumeError, "metadata says run 'run-resume' version 1"):
            self.load_from(hub, version=5)
        # Unpinned, the mislabelled newest folder is skipped and the genuine one is used.
        loaded = self.load_from(hub)
        self.assertEqual(loaded.state["theta_version"], 1)
        self.assertEqual(len(loaded.skipped), 1)

    def test_a_hub_checkpoint_from_a_different_configuration_is_refused_on_resume(self):
        hub, _ = self.push_first_rounds(rounds=1, repo_private=True, policy="auto")
        loaded = self.load_from(hub)
        with self.assertRaisesRegex(ResumeError, "hyperparameters"):
            open_run(self.root / "restored", loaded=loaded, learning_rate=0.5)
        with self.assertRaisesRegex(ResumeError, "round_plan_sha256"):
            open_run(self.root / "restored", loaded=loaded, plan=make_plan(units_per_step=3))

    def test_unknown_optimizer_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hf_push_optimizer"):
            open_run(self.root / "bad", hf_push_optimizer="sometimes")


def gradient_dirs(root: Path) -> list[str]:
    folder = root / "run" / "gradients"
    return sorted(entry.name for entry in folder.iterdir() if entry.is_dir())


class GradientRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def files_in(self, name: str, round_dir: str) -> int:
        return len(list((self.root / name / "run" / "gradients" / round_dir).glob("*.safetensors")))

    def test_gradients_are_stored_per_round_and_everything_is_kept_on_request(self):
        app = open_run(self.root / "all", keep_gradient_rounds=-1)
        with Served(app) as served:
            served.finish()
        self.assertEqual(gradient_dirs(self.root / "all"),
                         ["round-000001", "round-000002", "round-000003"])
        for round_dir in gradient_dirs(self.root / "all"):
            self.assertEqual(self.files_in("all", round_dir), 2)

    def test_by_default_only_the_latest_round_s_gradients_are_kept(self):
        # Every file is as large as the adapter, so keeping all of them would fill
        # a small coordinator's disk; the default is the newest round only.
        app = open_run(self.root / "default")
        with Served(app) as served:
            served.finish()
        self.assertEqual(gradient_dirs(self.root / "default"), ["round-000003"])
        self.assertEqual(self.files_in("default", "round-000003"), 2)

    def test_closed_rounds_beyond_the_retention_count_are_deleted(self):
        app = open_run(self.root / "keep1", keep_gradient_rounds=1)
        with Served(app) as served:
            served.work("w")
            served.work("w")
            # Rounds 1 and 2 are closed; only the newest closed round is kept.
            self.assertEqual(gradient_dirs(self.root / "keep1"), ["round-000002"])
            served.work("w")
        self.assertEqual(gradient_dirs(self.root / "keep1"), ["round-000003"])

    def test_zero_drops_closed_rounds_but_never_the_open_one(self):
        app = open_run(self.root / "keep0", keep_gradient_rounds=0)
        with Served(app) as served:
            served.work("w")
            self.assertEqual(gradient_dirs(self.root / "keep0"), [])
            served.work("w", max_units=1)         # half of round 2 is in
            self.assertEqual(gradient_dirs(self.root / "keep0"), ["round-000002"])
            self.assertEqual(self.files_in("keep0", "round-000002"), 1)

    def test_files_from_the_older_flat_layout_are_never_touched(self):
        app = open_run(self.root / "legacy", keep_gradient_rounds=0)
        legacy = self.root / "legacy" / "run" / "gradients" / ("a" * 64 + ".safetensors")
        legacy.write_bytes(b"old layout")
        with Served(app) as served:
            served.finish()
        self.assertTrue(legacy.is_file())

    def test_a_late_payload_lands_in_the_round_it_was_accepted_in(self):
        app = open_run(self.root / "late")
        with Served(app) as served:
            served.work("w")
        self.assertEqual(app.coordinator.round_id, 2)
        app.persist_result_payload("straggler", b"payload", round_id=1)
        self.assertEqual(self.files_in("late", "round-000001"), 3)
        self.assertEqual(self.files_in("late", "round-000002"), 0)

    def test_interrupted_rounds_files_are_set_aside_on_resume_then_pruned_with_the_rest(self):
        crashed = open_run(self.root / "resumed")
        with Served(crashed) as served:
            served.work("w")
            served.work("w", max_units=1)        # one unit of round 2 accepted, then the crash
        self.assertEqual(self.files_in("resumed", "round-000002"), 1)

        resumed = open_run(self.root / "resumed",
                           loaded=load_latest_checkpoint(self.root / "resumed" / "run" / "ckpt"),
                           keep_gradient_rounds=0)

        names = gradient_dirs(self.root / "resumed")
        self.assertEqual(len([n for n in names if n.startswith("round-000002.discarded-")]), 1)
        self.assertNotIn("round-000002", names)
        with Served(resumed) as served:
            served.work("w")                      # round 2 again, then it closes
        names = gradient_dirs(self.root / "resumed")
        # Round 2 is closed and retention is 0, so both the fresh and the set-aside copies go.
        self.assertEqual(names, [])

    def test_the_http_handler_records_the_round_each_result_was_accepted_in(self):
        app = open_run(self.root / "spy")
        with patch.object(app, "persist_result_payload", wraps=app.persist_result_payload) as spy:
            with Served(app) as served:
                served.work("w", max_units=1)
        self.assertEqual(spy.call_args.kwargs["round_id"], 1)

    def test_any_negative_retention_means_keep_everything(self):
        app = open_run(self.root / "neg", keep_gradient_rounds=-5)
        with Served(app) as served:
            served.finish()
        self.assertEqual(len(gradient_dirs(self.root / "neg")), 3)


class HubPushRecoveryTests(unittest.TestCase):
    """What a restarted coordinator does about Hub pushes the previous process left behind."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hub = FakeHub(self.root / "hub", repo_private=True)
        self.hub_kwargs = dict(hf_adapter_repo="test/adapter", hf_token="write-token",
                               hf_push_every_steps=1)

    # -- helpers -----------------------------------------------------------------
    def hub_versions(self) -> list[int]:
        folder = self.hub.root / "checkpoints" / "run-resume"
        if not folder.is_dir():
            return []
        return sorted(int(entry.name.split("-")[1]) for entry in folder.glob("theta-*")
                      if (entry / "adapter_model.safetensors").is_file()
                      and (entry / "adapter_metadata.json").is_file())

    def run_origin(self, *, rounds: int, fail_uploads: bool = False, with_hub: bool = True,
                   hub_policy: str = "auto") -> Application:
        api = self.hub.api()
        if fail_uploads:
            api.upload_folder.side_effect = RuntimeError("hub down write-token")
        kwargs = dict(self.hub_kwargs, hf_push_optimizer=hub_policy) if with_hub else {}
        with patch("huggingface_hub.HfApi", return_value=api), patch("g1j_dist_coordinator.time.sleep"):
            app = open_run(self.root / "origin", keep=10, **kwargs)
            with Served(app) as served:
                for _ in range(rounds):
                    served.work("w")
            if with_hub:
                app.hub_push_queue.join()
        return app

    def resume(self, *, listing="hub", from_version: int | None = None, **overrides):
        """Resume the origin run in a new Application and run hub recovery to completion."""
        ckpt = self.root / "origin" / "run" / "ckpt"
        loaded = (load_latest_checkpoint(ckpt) if from_version is None
                  else load_checkpoint(ckpt / checkpoint_name(from_version)))

        def list_versions(*, repo_id, run_id, token):
            if listing == "unreachable":
                raise ConnectionError("no route to host (token write-token)")
            return self.hub_versions()[::-1]

        kwargs = dict(self.hub_kwargs, **overrides)
        with patch("huggingface_hub.HfApi", return_value=self.hub.api()), \
                patch("g1j_dist_coordinator.list_hub_checkpoint_versions", list_versions), \
                patch("g1j_dist_coordinator.time.sleep"):
            app = open_run(self.root / "origin", loaded=loaded, **kwargs)
            report = app.recover_hub_pushes()
            app.hub_push_queue.join()
        return app, report

    def queue_dir(self, version: int) -> Path:
        return self.root / "origin" / "run" / "hub_push_queue" / f"theta-{version:06d}"

    def state(self) -> dict:
        return json.loads((self.root / "origin" / "run" / "coordinator_state.json").read_text())

    # -- replay -----------------------------------------------------------------
    def test_a_push_that_never_reached_the_hub_is_replayed_after_a_restart(self):
        origin = self.run_origin(rounds=1, fail_uploads=True)
        self.assertEqual(origin.hub_push_versions["000001"]["status"], "failed")
        self.assertTrue(self.queue_dir(1).is_dir())
        self.assertEqual(self.hub_versions(), [])

        app, report = self.resume()

        self.assertEqual(report["replayed"], [1])
        self.assertEqual(self.hub_versions(), [1])
        self.assertTrue((self.hub.root / "checkpoints/run-resume/theta-000001/optimizer.safetensors").is_file())
        self.assertFalse(self.queue_dir(1).exists(), "uploaded entries are cleaned up")
        entry = app.hub_push_versions["000001"]
        self.assertEqual((entry["status"], entry["source"]), ("uploaded", "replayed_after_restart"))
        self.assertNotIn("error", entry)
        self.assertEqual(app.hub_push_status["status"], "uploaded")
        self.assertEqual(self.state()["hub_push"]["versions"]["000001"]["status"], "uploaded")

    def test_versions_already_on_the_hub_are_recorded_and_never_pushed_again(self):
        self.run_origin(rounds=1, fail_uploads=True)
        # The upload actually landed before the process died.
        landed = self.hub.root / "checkpoints/run-resume/theta-000001"
        landed.mkdir(parents=True)
        shutil.copy2(self.queue_dir(1) / "adapter_model.safetensors", landed)
        shutil.copy2(self.queue_dir(1) / "metadata.json", landed / "adapter_metadata.json")

        app, report = self.resume()

        self.assertEqual(report["found_on_hub"], [1])
        self.assertEqual(report["replayed"], [])
        self.assertFalse(self.queue_dir(1).exists())
        self.assertEqual(app.hub_push_versions["000001"]["source"], "found_on_hub")
        before = self.hub.uploads.copy()
        with patch("huggingface_hub.HfApi", return_value=self.hub.api()):
            app.start_hub_push(force=True)         # what POST /hub-push does
            app.hub_push_queue.join()
        self.assertEqual(self.hub.uploads, before, "nothing is uploaded a second time")

    def test_the_resumed_version_is_pushed_when_it_is_due_and_missing(self):
        self.run_origin(rounds=1, with_hub=False)

        app, report = self.resume()

        self.assertEqual(report["pushed_resumed_version"], 1)
        self.assertEqual(self.hub_versions(), [1])
        self.assertEqual(app.hub_push_versions["000001"]["status"], "uploaded")
        metadata = json.loads((self.hub.root / "checkpoints/run-resume/theta-000001/adapter_metadata.json").read_text())
        self.assertEqual(metadata["theta_version"], 1)
        self.assertEqual(metadata["round_id"], 1, "the round that produced the version, not the open one")

    def test_the_resumed_version_is_left_alone_when_it_is_not_a_push_step(self):
        self.run_origin(rounds=1, with_hub=False)
        app, report = self.resume(hf_push_every_steps=10)
        self.assertIsNone(report["pushed_resumed_version"])
        self.assertEqual(self.hub_versions(), [])

    def test_the_resumed_version_is_not_pushed_again_when_the_hub_has_it(self):
        self.run_origin(rounds=1)
        self.assertEqual(self.hub_versions(), [1])
        uploads_before = len(self.hub.uploads)
        app, report = self.resume()
        self.assertEqual(report["found_on_hub"], [1])
        self.assertIsNone(report["pushed_resumed_version"])
        self.assertEqual(len(self.hub.uploads), uploads_before)

    # -- entries that must not be replayed ------------------------------------------
    def test_entries_from_a_rolled_back_timeline_are_dropped(self):
        self.run_origin(rounds=2, fail_uploads=True)
        self.assertTrue(self.queue_dir(2).is_dir())
        # Version 2's checkpoint is unusable, so resume falls back to version 1.
        shutil.rmtree(self.root / "origin" / "run" / "ckpt" / checkpoint_name(2))

        app, report = self.resume()

        self.assertEqual(app.coordinator.theta_version, 1)
        self.assertEqual(report["dropped"], ["theta-000002"])
        self.assertEqual(report["replayed"], [1])
        self.assertFalse(self.queue_dir(2).exists())
        self.assertEqual(self.hub_versions(), [1])

    def test_a_damaged_queue_entry_is_dropped(self):
        self.run_origin(rounds=1, fail_uploads=True)
        weights = self.queue_dir(1) / "adapter_model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 0xFF
        weights.write_bytes(bytes(data))
        app, report = self.resume(hf_push_every_steps=10)
        self.assertEqual(report["dropped"], ["theta-000001"])
        self.assertEqual(self.hub_versions(), [])

    def test_an_entry_whose_adapter_disagrees_with_the_local_checkpoint_is_dropped(self):
        self.run_origin(rounds=1, fail_uploads=True)
        # Self-consistent but different from what this run actually produced at version 1.
        entry = self.queue_dir(1)
        tensors = load_file(str(entry / "adapter_model.safetensors"))
        tensors = {name: value + 1.0 for name, value in tensors.items()}
        save_file(tensors, str(entry / "adapter_model.safetensors"))
        from rwkv_dist_protocol import peft_saved_name_to_parameter_name
        metadata = json.loads((entry / "metadata.json").read_text())
        metadata["theta_hash"] = tensor_map_sha256(
            {peft_saved_name_to_parameter_name(n): t for n, t in tensors.items()})
        (entry / "metadata.json").write_text(json.dumps(metadata))

        app, report = self.resume(hf_push_every_steps=10)

        self.assertEqual(report["dropped"], ["theta-000001"])
        self.assertEqual(self.hub_versions(), [])

    def test_entries_of_another_run_are_neither_pushed_nor_deleted(self):
        self.run_origin(rounds=1, fail_uploads=True)
        metadata_path = self.queue_dir(1) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["run_id"] = "someone-elses-run"
        metadata_path.write_text(json.dumps(metadata))

        app, report = self.resume(hf_push_every_steps=10)

        self.assertEqual(report["ignored"], ["theta-000001"])
        self.assertTrue(self.queue_dir(1).is_dir())
        self.assertEqual(self.hub_versions(), [])

    # -- the hub itself ---------------------------------------------------------------
    def test_an_unreachable_hub_does_not_stop_the_replay_and_the_error_is_redacted(self):
        self.run_origin(rounds=1, fail_uploads=True)

        app, report = self.resume(listing="unreachable")

        self.assertIn("ConnectionError", report["hub_error"])
        self.assertNotIn("write-token", report["hub_error"])
        self.assertIn("[REDACTED]", report["hub_error"])
        self.assertEqual(report["replayed"], [1])
        self.assertEqual(self.hub_versions(), [1])

    def test_a_hub_that_is_ahead_of_the_resumed_version_is_reported_not_recorded(self):
        self.run_origin(rounds=2)
        self.assertEqual(self.hub_versions(), [1, 2])
        shutil.rmtree(self.root / "origin" / "run" / "ckpt" / checkpoint_name(2))

        app, report = self.resume()

        self.assertEqual(app.coordinator.theta_version, 1)
        self.assertEqual(report["hub_ahead"], [2])
        self.assertNotIn("000002", app.hub_push_versions)
        self.assertEqual(report["found_on_hub"], [1])

    def test_nothing_happens_without_hub_pushing(self):
        self.run_origin(rounds=1, with_hub=False)
        app, report = self.resume(hf_adapter_repo=None, hf_token=None)
        self.assertEqual(report["replayed"], [])
        self.assertIsNone(report["hub_error"])
        self.assertEqual(self.hub_versions(), [])

    # -- the status file ---------------------------------------------------------------
    def test_push_history_survives_later_checkpoint_writes(self):
        app = self.run_origin(rounds=2)
        self.assertEqual(sorted(self.state()["hub_push"]["versions"]), ["000001", "000002"])
        app.persist_checkpoint()                    # rewrites coordinator_state.json
        versions = self.state()["hub_push"]["versions"]
        self.assertEqual(sorted(versions), ["000001", "000002"])
        self.assertEqual(versions["000002"]["status"], "uploaded")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RecordingReader(io.BytesIO):
    """A request body that remembers how much was asked for at a time."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.requests: list[int] = []

    def read(self, size=-1):
        self.requests.append(size)
        return super().read(size)


class LeanCoordinatorTests(unittest.TestCase):
    """Uploads go to disk, not memory, and the event log leaves memory once written."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def incoming(self, app: Application) -> list[str]:
        return sorted(path.name for path in app.incoming_dir.iterdir())

    def test_an_upload_is_copied_in_bounded_chunks_and_hashed_on_the_way(self):
        app = open_run(self.root / "chunks")
        data = bytes(range(256)) * 22  # 5632 bytes
        reader = RecordingReader(data)
        with patch.object(coordinator_module, "UPLOAD_CHUNK_BYTES", 1000):
            upload = app.stream_upload(reader, len(data))
        self.assertLessEqual(max(reader.requests), 1000)
        self.assertGreater(len(reader.requests), 5)
        self.assertEqual(upload.sha256, sha256_bytes(data))
        self.assertEqual(upload.size, len(data))
        self.assertEqual(upload.path.read_bytes(), data)
        self.assertEqual(upload.path.parent, app.incoming_dir)

    def test_a_body_that_ends_early_is_an_error_and_leaves_nothing_behind(self):
        app = open_run(self.root / "short")
        with self.assertRaises(ProtocolError) as caught:
            app.stream_upload(io.BytesIO(b"x" * 100), 5000)
        self.assertEqual(caught.exception.code, "truncated_payload")
        self.assertEqual(self.incoming(app), [])

    def test_a_read_error_mid_upload_leaves_nothing_behind(self):
        app = open_run(self.root / "broken")

        class Breaks:
            calls = 0

            def read(self, size):
                self.calls += 1
                if self.calls > 1:
                    raise ConnectionResetError("client went away")
                return b"x" * size

        with patch.object(coordinator_module, "UPLOAD_CHUNK_BYTES", 10):
            with self.assertRaises(ConnectionResetError):
                app.stream_upload(Breaks(), 100)
        self.assertEqual(self.incoming(app), [])

    def test_an_accepted_upload_is_moved_into_its_round_folder(self):
        app = open_run(self.root / "accepted")
        with Served(app) as served:
            results = served.work("w", max_units=1)
        self.assertEqual(results[0][0], 200)
        self.assertEqual(self.incoming(app), [])
        [stored] = list((app.run_dir / "gradients" / "round-000001").glob("*.safetensors"))
        accepted = logged_events(app, "result_accepted")[-1]
        self.assertEqual(sha256_bytes(stored.read_bytes()), accepted["gradient_sha256"])
        self.assertEqual(list((app.run_dir / "gradients" / "round-000001").glob("*.tmp")), [])

    def test_a_rejected_upload_leaves_no_file_anywhere(self):
        app = open_run(self.root / "rejected")
        with Served(app) as served:
            served.get_json("/join?worker_id=w")
            lease = served.get_json("/lease?worker_id=w")
            status, body = served.post_result("w", lease, gradient_sha256="0" * 64)
        self.assertEqual((status, body["error"]), (409, "gradient_hash_mismatch"))
        self.assertEqual(self.incoming(app), [])
        self.assertEqual(list((app.run_dir / "gradients").rglob("*.safetensors")), [])

    def test_a_lost_lease_upload_is_discarded_without_leaving_a_file(self):
        app = open_run(self.root / "late")
        with Served(app) as served:
            served.get_json("/join?worker_id=w")
            lease = served.get_json("/lease?worker_id=w")
            served.get_json("/lease?worker_id=w")  # a second unit for the same worker
            status, body = served.post_result("w", lease, lease_id="no-such-lease")
        self.assertEqual((status, body["error"]), (409, "unknown_lease"))
        self.assertEqual(self.incoming(app), [])

    def test_uploads_left_by_a_crashed_process_are_cleared_at_startup(self):
        stale = self.root / "restart" / "run" / "incoming"
        stale.mkdir(parents=True)
        (stale / "half-received.part").write_bytes(b"x" * 1000)
        app = open_run(self.root / "restart")
        self.assertEqual(self.incoming(app), [])

    def test_events_leave_memory_once_written_and_none_are_lost_or_repeated(self):
        app = open_run(self.root / "events")
        log = app.run_dir / "round_log.jsonl"
        written = len(log.read_text().splitlines()) if log.exists() else 0
        self.assertEqual(app.coordinator.events, [])
        for batch in range(3):
            for index in range(4):
                app.coordinator.events.append({"event": "probe", "batch": batch, "index": index})
            app.persist_events()
            self.assertEqual(app.coordinator.events, [])
            app.persist_events()  # nothing new: must not repeat anything
        probes = [event for event in map(json.loads, log.read_text().splitlines())
                  if event["event"] == "probe"]
        self.assertEqual([(e["batch"], e["index"]) for e in probes],
                         [(b, i) for b in range(3) for i in range(4)])
        self.assertEqual(len(log.read_text().splitlines()), written + 12)

    def test_an_event_added_while_persisting_is_kept_for_the_next_write(self):
        app = open_run(self.root / "race")
        app.coordinator.events.append({"event": "probe", "n": 1})
        real_dumps = json.dumps
        injected = []

        def dumps_and_append(*args, **kwargs):
            if not injected:  # a heartbeat arrives in the middle of the write
                injected.append(True)
                app.coordinator.events.append({"event": "probe", "n": 2})
            return real_dumps(*args, **kwargs)

        with patch.object(coordinator_module.json, "dumps", dumps_and_append):
            app.persist_events()
        app.persist_events()
        probes = [e["n"] for e in map(json.loads, (app.run_dir / "round_log.jsonl").read_text().splitlines())
                  if e["event"] == "probe"]
        self.assertEqual(probes, [1, 2])
