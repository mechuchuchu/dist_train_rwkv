"""Drive the real g1j_dist_worker.main() against a scripted coordinator.

transformers/peft are stubbed and the GPU, model load, snapshot application and
gradient step are replaced, so these tests run on CPU without model weights.
What they exercise is the worker's control flow when a round advances between
its /round, /lease and /snapshot requests.
"""

from __future__ import annotations

import contextlib
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


def round_descriptor(theta_hash: str, theta_version: int, *, repo: str | None = None,
                     revision: str | None = None) -> dict:
    return {
        "run_id": "run-1",
        "round_id": theta_version + 1,
        "theta_version": theta_version,
        "theta_hash": theta_hash,
        "status": "open",
        "training_complete": False,
        "base": {
            "repo": repo or WORKER.DEFAULT_MODEL_ID,
            "revision": revision or WORKER.DEFAULT_MODEL_REVISION,
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
        self.paths: list[str] = []

    def request(self, path, *, method="GET", body=None, headers=None, timeout=30.0):
        route = path.split("?", 1)[0]
        self.calls.append(route)
        self.paths.append(path)
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


class WorkerHarness:
    """Runs the real main() against a scripted coordinator."""

    def run_worker(self, coordinator: ScriptedCoordinator, *, apply_snapshot=None,
                   flags: tuple[str, ...] = (), model_path: bool = True):
        """Run main(); `flags` are extra arguments, model_path=False leaves --model-path unset."""
        apply_snapshot = apply_snapshot or MagicMock(return_value={})
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "release-manifest.json").write_bytes(MANIFEST_BYTES)
            argv = ["g1j_dist_worker.py", "--server", "http://coordinator.invalid",
                    "--token", "t", "--worker-id", "w", "--max-units", "1",
                    "--poll-seconds", "0.01", *flags]
            if model_path:
                argv += ["--model-path", model_dir]
            output = io.StringIO()
            self.ensure_checkpoint = MagicMock(return_value=Path(model_dir))
            self.load_model = MagicMock(return_value=FakeModel())
            with patch.object(sys, "argv", argv), \
                    patch.object(WORKER, "Client", lambda *args, **kwargs: coordinator), \
                    patch.object(WORKER, "ensure_hf_checkpoint", self.ensure_checkpoint), \
                    patch.object(WORKER, "load_worker_model", self.load_model), \
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


class WorkerStaleLeaseTests(WorkerHarness, unittest.TestCase):
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

    def test_result_for_a_unit_another_worker_already_finished_is_discarded_not_fatal(self):
        # After a coordinator restart the round is reissued from its first unit,
        # so a worker still holding a lease from before the restart can finish a
        # unit that someone else has already been credited for.
        for code in ("duplicate_unit", "unknown_lease", "expired_lease", "round_closed"):
            with self.subTest(code=code):
                coordinator = ScriptedCoordinator(
                    routes={
                        "/round": [round_descriptor(T0, 0)] * 4,
                        "/lease": [leased(T0, 0, "lease-lost"), leased(T0, 0, "lease-next")],
                        "/snapshot": [snapshot_reply(T0)],
                        "/result": [CoordinatorHTTPError(409, {"error": code}),
                                    self.accepted_result(T0)],
                    },
                    sticky={"/join": {"status": "active"}},
                )

                output, _ = self.run_worker(coordinator)

                self.assertIn('"status": "result_discarded_lease_lost"', output)
                self.assertIn(f'"error": "{code}"', output)
                self.assertEqual(coordinator.calls.count("/result"), 2)

    def test_other_rejections_of_a_result_stay_fatal(self):
        coordinator = ScriptedCoordinator(
            routes={
                "/round": [round_descriptor(T0, 0)] * 2,
                "/lease": [leased(T0, 0, "lease-1")],
                "/snapshot": [snapshot_reply(T0)],
                "/result": [CoordinatorHTTPError(403, {"error": "worker_code_mismatch"})],
            },
            sticky={"/join": {"status": "active"}},
        )

        with self.assertRaises(CoordinatorHTTPError):
            self.run_worker(coordinator)

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


OTHER_ID = "RWKV/RWKV7-G1k-2.9B-20260930"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"


class WorkerModelSelectionTests(WorkerHarness, unittest.TestCase):
    """A worker serves exactly the model it was started for; it never takes one from the coordinator."""

    def other_model(self, theta_hash=T0):
        return round_descriptor(theta_hash, 0, repo=OTHER_ID, revision=OTHER_REVISION)

    def a_complete_unit(self, descriptor):
        return ScriptedCoordinator(
            routes={
                "/round": [descriptor] * 3,
                "/lease": [leased(T0, 0, "lease-1")],
                "/snapshot": [snapshot_reply(T0)],
                "/result": [self.accepted_result(T0)],
            },
            sticky={"/join": {"status": "active"}},
        )

    def refuses(self, descriptor, *flags):
        coordinator = ScriptedCoordinator(routes={"/round": [descriptor]})
        with self.assertRaises(RuntimeError) as caught:
            self.run_worker(coordinator, flags=flags)
        # It stopped at the first look at the round: no join, no lease, no model code run.
        self.assertEqual(coordinator.calls, ["/round"])
        self.load_model.assert_not_called()
        return str(caught.exception)

    def test_a_worker_started_for_another_model_trains_it(self):
        coordinator = self.a_complete_unit(self.other_model())
        flags = ("--model-id", OTHER_ID, "--model-revision", OTHER_REVISION)

        output, _ = self.run_worker(coordinator, flags=flags)

        self.assertIn(f"loading {OTHER_ID}", output)
        self.assertEqual(coordinator.calls.count("/result"), 1)
        model_id, revision, _path = self.ensure_checkpoint.call_args.args
        self.assertEqual((model_id, revision), (OTHER_ID, OTHER_REVISION))

    def test_the_default_model_path_follows_the_model_id(self):
        coordinator = self.a_complete_unit(self.other_model())
        flags = ("--model-id", OTHER_ID, "--model-revision", OTHER_REVISION)

        self.run_worker(coordinator, flags=flags, model_path=False)

        self.assertEqual(self.ensure_checkpoint.call_args.args[2],
                         Path("/workspace/RWKV7-G1k-2.9B-20260930"))

    def test_the_default_worker_still_serves_the_default_model(self):
        coordinator = self.a_complete_unit(round_descriptor(T0, 0))

        self.run_worker(coordinator, model_path=False)

        self.assertEqual(self.ensure_checkpoint.call_args.args[:2],
                         (WORKER.DEFAULT_MODEL_ID, WORKER.DEFAULT_MODEL_REVISION))
        self.assertEqual(self.ensure_checkpoint.call_args.args[2],
                         Path("/workspace/RWKV7-G1j-1.5B-20260831"))

    def test_a_default_worker_refuses_a_coordinator_training_another_model(self):
        message = self.refuses(self.other_model())
        self.assertIn(f"coordinator trains {OTHER_ID}@{OTHER_REVISION}", message)
        self.assertIn(f"worker was started for {WORKER.DEFAULT_MODEL_ID}@{WORKER.DEFAULT_MODEL_REVISION}",
                      message)
        self.assertIn("--model-id/--model-revision", message)

    def test_a_worker_for_another_model_refuses_a_default_coordinator(self):
        message = self.refuses(round_descriptor(T0, 0),
                               "--model-id", OTHER_ID, "--model-revision", OTHER_REVISION)
        self.assertIn(f"worker was started for {OTHER_ID}@{OTHER_REVISION}", message)

    def test_the_same_repository_at_another_commit_is_another_model(self):
        descriptor = round_descriptor(T0, 0, revision=OTHER_REVISION)
        self.assertIn(f"{WORKER.DEFAULT_MODEL_ID}@{OTHER_REVISION}", self.refuses(descriptor))

    def test_the_same_commit_under_another_repository_is_another_model(self):
        descriptor = round_descriptor(T0, 0, repo=OTHER_ID)
        self.assertIn(f"{OTHER_ID}@{WORKER.DEFAULT_MODEL_REVISION}", self.refuses(descriptor))

    def test_the_model_to_load_is_never_taken_from_the_coordinator(self):
        # Whatever the coordinator announces, only the worker's own arguments decide what is loaded.
        coordinator = self.a_complete_unit(round_descriptor(T0, 0))
        self.run_worker(coordinator, model_path=False)
        for call in self.ensure_checkpoint.call_args_list:
            self.assertEqual(call.args[:2], (WORKER.DEFAULT_MODEL_ID, WORKER.DEFAULT_MODEL_REVISION))


def contributors_reply(*rows):
    return {"run_id": "run-1", "theta_version": 1, "contributors": [
        {"nickname": name, "units": units, "loss_tokens": tokens, "steps": 1, "compute_seconds": 2.5}
        for name, units, tokens in rows]}


class WorkerContributeTests(WorkerHarness, unittest.TestCase):
    """--contribute NICKNAME: the work is credited to a name the contributor chose."""

    def one_unit(self, extra_routes=None):
        routes = {
            "/round": [round_descriptor(T0, 0)] * 4,
            "/lease": [leased(T0, 0, "lease-1")],
            "/snapshot": [snapshot_reply(T0)],
            "/result": [self.accepted_result(T0)],
        } | (extra_routes or {})
        return ScriptedCoordinator(routes=routes, sticky={"/join": {"status": "active"}})

    def joins(self, coordinator):
        return [path for path in coordinator.paths if path.startswith("/join")]

    def test_the_nickname_is_sent_with_every_join(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(("Alice", 1, 1))]})

        self.run_worker(coordinator, flags=("--contribute", "Alice"))

        joins = self.joins(coordinator)
        self.assertGreaterEqual(len(joins), 2)
        self.assertEqual(set(joins), {"/join?worker_id=w&nickname=Alice"})

    def test_a_nickname_in_any_script_is_sent_percent_encoded(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(("김철수", 1, 1))]})

        output, _ = self.run_worker(coordinator, flags=("--contribute", "김철수"))

        self.assertEqual(set(self.joins(coordinator)),
                         {"/join?worker_id=w&nickname=%EA%B9%80%EC%B2%A0%EC%88%98"})
        self.assertIn('"nickname": "김철수"', output)

    def test_the_nickname_is_tidied_the_way_the_coordinator_will(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(("alice", 1, 1))]})
        self.run_worker(coordinator, flags=("--contribute", "  ａｌｉｃｅ "))
        self.assertEqual(set(self.joins(coordinator)), {"/join?worker_id=w&nickname=alice"})

    def test_the_worker_says_it_is_contributing_once_the_coordinator_has_taken_it(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(("Alice", 1, 1))]})

        output, _ = self.run_worker(coordinator, flags=("--contribute", "Alice"))

        line = next(line for line in output.splitlines() if '"contributing"' in line)
        self.assertEqual(json.loads(line), {"status": "contributing", "nickname": "Alice",
                                            "worker_id": "w", "run_id": "run-1"})
        self.assertLess(output.index("contributing"), output.index("loading "))

    def test_without_the_option_nothing_about_contributors_is_sent_or_printed(self):
        coordinator = self.one_unit()

        output, _ = self.run_worker(coordinator)

        self.assertEqual(set(self.joins(coordinator)), {"/join?worker_id=w"})
        self.assertNotIn("/contributors", coordinator.calls)
        self.assertNotIn("contribut", output)

    def test_an_unusable_nickname_stops_the_worker_before_it_talks_to_anyone(self):
        for nickname in ("bad\nname", "", "x" * 33, "(anonymous)"):
            with self.subTest(nickname=nickname):
                coordinator = self.one_unit()
                with contextlib.redirect_stderr(io.StringIO()) as errors, \
                        self.assertRaises(SystemExit) as caught:
                    self.run_worker(coordinator, flags=("--contribute", nickname))
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--contribute", errors.getvalue())
                self.assertEqual(coordinator.calls, [])

    def test_at_the_end_the_contributor_is_told_what_was_credited_and_where_they_stand(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(
            ("bob", 9, 90), ("Alice", 4, 40), ("(anonymous)", 1, 10))]})

        output, _ = self.run_worker(coordinator, flags=("--contribute", "Alice"))

        line = next(line for line in output.splitlines() if '"status": "contribution"' in line)
        self.assertEqual(json.loads(line), {
            "status": "contribution", "nickname": "Alice", "units": 4, "loss_tokens": 40, "steps": 1,
            "compute_seconds": 2.5, "rank": 2, "contributors": 3})

    def test_equal_work_shares_a_rank(self):
        coordinator = self.one_unit({"/contributors": [contributors_reply(("amy", 2, 20), ("Alice", 2, 20))]})
        output, _ = self.run_worker(coordinator, flags=("--contribute", "Alice"))
        line = next(line for line in output.splitlines() if '"status": "contribution"' in line)
        self.assertEqual(json.loads(line)["rank"], 1)

    def test_a_listing_that_cannot_be_had_does_not_fail_the_worker(self):
        failures = {
            "server error": CoordinatorHTTPError(500, {"error": "boom"}),
            "not json": (200, {}, b"<html>"),
            "someone elses run": contributors_reply(("Alice", 1, 1)) | {"run_id": "another-run"},
            "nickname missing": contributors_reply(("bob", 1, 1)),
            "wrong shape": {"run_id": "run-1"},
        }
        for label, reply in failures.items():
            with self.subTest(case=label):
                coordinator = self.one_unit({"/contributors": [reply]})
                output, _ = self.run_worker(coordinator, flags=("--contribute", "Alice"))
                self.assertEqual(coordinator.calls.count("/result"), 1)
                self.assertNotIn('"status": "contribution"', output)
                if label != "someone elses run":
                    self.assertIn('"contribution_unavailable"', output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
