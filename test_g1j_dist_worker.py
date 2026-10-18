"""Drive the real g1j_dist_worker.main() against a scripted coordinator.

transformers/peft are stubbed and the GPU, model load, snapshot application and
gradient step are replaced, so these tests run on CPU without model weights.
What they exercise is the worker's control flow when a round advances between
its /round, /lease and /snapshot requests.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from rwkv_dist_client import CoordinatorHTTPError


def load_worker_module():
    stubs = {}
    attributes = {
        "peft": {"LoraConfig": object, "get_peft_model": lambda *args, **kwargs: None},
        "transformers": {"AutoModelForCausalLM": object},
    }
    for name, values in attributes.items():
        module = types.ModuleType(name)
        for key, value in values.items():
            setattr(module, key, value)
        stubs[name] = module
    # Swap in only the two stubs and restore only those. patch.dict(sys.modules)
    # would also evict everything imported inside the block (torch included),
    # and re-importing torch's C extension later crashes the interpreter.
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        import g1j_dist_worker
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return g1j_dist_worker


WORKER = load_worker_module()
BUNDLE_SHA = WORKER.bundle_sha256(Path(WORKER.__file__).parent)
MANIFEST_BYTES = b"{}"
MANIFEST_SHA = WORKER.sha256_bytes(MANIFEST_BYTES)
T0 = "a" * 64
T1 = "b" * 64


def round_descriptor(theta_hash: str, theta_version: int) -> dict:
    return {
        "run_id": "run-1",
        "round_id": theta_version + 1,
        "theta_version": theta_version,
        "theta_hash": theta_hash,
        "status": "open",
        "training_complete": False,
        "base": {
            "repo": WORKER.MODEL_ID,
            "revision": WORKER.MODEL_REVISION,
            "release_manifest_sha256": MANIFEST_SHA,
        },
        "worker_code_sha256": BUNDLE_SHA,
        "received_units": 1,
        "received_loss_tokens": 1,
        "close_metrics": None,
    }


def leased(theta_hash: str, theta_version: int, lease_id: str) -> dict:
    return {
        "status": "leased",
        "run_id": "run-1",
        "round_id": theta_version + 1,
        "theta_version": theta_version,
        "theta_hash": theta_hash,
        "lease_id": lease_id,
        "worker_id": "w",
        "lease_seconds": 60.0,
        "lease_expires_at": 0,
        "unit": {"unit_id": "unit-0", "unit_sha256": "u" * 64},
    }


def snapshot_reply(theta_hash: str):
    return 200, {"X-Theta-Hash": theta_hash}, b"adapter-bytes"


class ScriptedCoordinator:
    """Stands in for rwkv_dist_client.Client, answering from per-route queues."""

    def __init__(self, routes: dict, sticky: dict | None = None):
        self.routes = {route: list(replies) for route, replies in routes.items()}
        self.sticky = sticky or {}
        self.calls: list[str] = []

    def request(self, path, *, method="GET", body=None, headers=None, timeout=30.0):
        route = path.split("?", 1)[0]
        self.calls.append(route)
        if route in self.sticky:
            reply = self.sticky[route]
        else:
            queue = self.routes.get(route)
            if not queue:
                raise AssertionError(f"unexpected request to {route}")
            reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            return 200, {}, json.dumps(reply).encode()
        return reply


class FakeModel:
    def named_parameters(self):
        return []


class WorkerStaleLeaseTests(unittest.TestCase):
    def run_worker(self, coordinator: ScriptedCoordinator, *, apply_snapshot=None):
        apply_snapshot = apply_snapshot or MagicMock(return_value={})
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "release-manifest.json").write_bytes(MANIFEST_BYTES)
            argv = ["g1j_dist_worker.py", "--server", "http://coordinator.invalid",
                    "--token", "t", "--worker-id", "w", "--max-units", "1",
                    "--poll-seconds", "0.01", "--model-path", model_dir]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), \
                    patch.object(WORKER, "Client", lambda *args, **kwargs: coordinator), \
                    patch.object(WORKER, "ensure_hf_checkpoint", return_value=Path(model_dir)), \
                    patch.object(WORKER, "load_worker_model", return_value=FakeModel()), \
                    patch.object(WORKER, "apply_snapshot", apply_snapshot), \
                    patch.object(WORKER, "compute_unit",
                                 return_value=(b"gradient-bytes", 1.5, 1, 0.5)), \
                    patch.object(WORKER.torch.cuda, "is_available", return_value=True), \
                    patch.object(WORKER.torch.cuda, "get_device_name", return_value="fake-gpu"), \
                    patch.object(WORKER.signal, "signal"), \
                    redirect_stdout(output):
                WORKER.main()
        return output.getvalue(), apply_snapshot

    def accepted_result(self, theta_hash: str) -> dict:
        return {"status": "accepted", "round_closed": True, "theta_hash": theta_hash}

    def test_snapshot_header_from_a_newer_round_drops_the_lease_and_resyncs(self):
        # The round closes after /lease but before /snapshot: the header already
        # carries the next adapter version. Previously this killed the worker.
        coordinator = ScriptedCoordinator(
            routes={
                "/round": [round_descriptor(T0, 0), round_descriptor(T0, 0),
                           round_descriptor(T1, 1), round_descriptor(T1, 1)],
                "/lease": [leased(T0, 0, "lease-old"), leased(T1, 1, "lease-new")],
                "/snapshot": [snapshot_reply(T1), snapshot_reply(T1)],
                "/release": [CoordinatorHTTPError(409, {"error": "unknown_lease"})],
                "/result": [self.accepted_result(T1)],
            },
            sticky={"/join": {"status": "active"}},
        )

        output, apply_snapshot = self.run_worker(coordinator)

        self.assertIn('"status": "stale_lease_dropped"', output)
        self.assertIn('"lease_id": "lease-old"', output)
        self.assertIn('"release": "already_gone"', output)
        self.assertEqual(coordinator.calls.count("/release"), 1)
        self.assertEqual(coordinator.calls.count("/snapshot"), 2)
        # Only the snapshot that matched the lease was ever applied.
        apply_snapshot.assert_called_once()
        self.assertEqual(apply_snapshot.call_args.args[2], T1)
        self.assertEqual(coordinator.calls.count("/result"), 1)

    def test_lease_newer_than_the_round_descriptor_is_dropped_and_released(self):
        # /round was read just before the round advanced, /lease just after.
        coordinator = ScriptedCoordinator(
            routes={
                "/round": [round_descriptor(T0, 0), round_descriptor(T0, 0),
                           round_descriptor(T1, 1), round_descriptor(T1, 1)],
                "/lease": [leased(T1, 1, "lease-early"), leased(T1, 1, "lease-retry")],
                "/snapshot": [snapshot_reply(T1)],
                "/release": [{"status": "released", "unit_id": "unit-0"}],
                "/result": [self.accepted_result(T1)],
            },
            sticky={"/join": {"status": "active"}},
        )

        output, apply_snapshot = self.run_worker(coordinator)

        self.assertIn('"status": "stale_lease_dropped"', output)
        self.assertIn('"release": "released"', output)
        # The mismatch is caught before any download is attempted.
        self.assertEqual(coordinator.calls.count("/snapshot"), 1)
        apply_snapshot.assert_called_once()
        self.assertEqual(coordinator.calls.count("/result"), 1)

    def test_corrupt_snapshot_is_still_fatal_and_not_mistaken_for_a_race(self):
        # Header matches the lease, so this is not staleness: a payload that
        # fails its own integrity check must keep stopping the worker.
        coordinator = ScriptedCoordinator(
            routes={
                "/round": [round_descriptor(T0, 0), round_descriptor(T0, 0)],
                "/lease": [leased(T0, 0, "lease-1")],
                "/snapshot": [snapshot_reply(T0)],
            },
            sticky={"/join": {"status": "active"}},
        )
        corrupt = MagicMock(side_effect=RuntimeError("snapshot hash mismatch"))

        with self.assertRaisesRegex(RuntimeError, "snapshot hash mismatch"):
            self.run_worker(coordinator, apply_snapshot=corrupt)

        self.assertNotIn("/release", coordinator.calls)
        self.assertNotIn("/result", coordinator.calls)

    def test_up_to_date_lease_is_unaffected(self):
        coordinator = ScriptedCoordinator(
            routes={
                "/round": [round_descriptor(T0, 0), round_descriptor(T0, 0),
                           round_descriptor(T0, 0)],
                "/lease": [leased(T0, 0, "lease-1")],
                "/snapshot": [snapshot_reply(T0)],
                "/result": [self.accepted_result(T0)],
            },
            sticky={"/join": {"status": "active"}},
        )

        output, apply_snapshot = self.run_worker(coordinator)

        self.assertNotIn("stale_lease_dropped", output)
        self.assertNotIn("/release", coordinator.calls)
        apply_snapshot.assert_called_once()
        self.assertEqual(coordinator.calls.count("/result"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
