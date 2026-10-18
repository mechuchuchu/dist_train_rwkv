from __future__ import annotations

import tempfile
import threading
import unittest
import time
from pathlib import Path
from unittest.mock import patch

import torch

import rwkv_dist_protocol
from rwkv_dist_protocol import (
    PayloadFile,
    ProtocolError,
    RoundCoordinator,
    adamw_hyperparameters,
    decode_tensors,
    encode_tensors,
    load_optimizer_state_tensors,
    sha256_bytes,
    tensor_map_sha256,
    unit_digest,
)


def make_unit(unit_id: str, loss_tokens: int) -> dict:
    unit = {
        "unit_id": unit_id,
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "total_tokens": 3,
        "loss_tokens": loss_tokens,
    }
    unit["unit_sha256"] = unit_digest(unit)
    return unit


def make_metadata(lease: dict, unit: dict, grad_payload: bytes, *, worker_code="worker-code"):
    return {
        "unit_id": unit["unit_id"],
        "unit_sha256": unit["unit_sha256"],
        "lease_id": lease["lease_id"],
        "worker_id": lease["worker_id"],
        "round_id": lease["round_id"],
        "theta_version": lease["theta_version"],
        "theta_hash": lease["theta_hash"],
        "worker_code_sha256": worker_code,
        "loss_tokens": unit["loss_tokens"],
        "loss_sum": 2.0,
        "gradient_sha256": sha256_bytes(grad_payload),
    }


class RoundCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.theta = {"lora_A": torch.zeros(2), "lora_B": torch.zeros(1)}
        self.units = [make_unit("unit-0", 2), make_unit("unit-1", 6)]
        self.coordinator = RoundCoordinator(
            theta=self.theta,
            units=self.units,
            run_id="test-run",
            base={"revision": "pinned"},
            tokenizer={"hash": "pinned"},
            worker_code_sha256="worker-code",
            target_loss_tokens=8,
            min_units=2,
            lease_seconds=30,
            learning_rate=1e-2,
        )
        self.assertEqual(self.coordinator.join("worker-0")["status"], "active")
        self.assertEqual(self.coordinator.join("worker-1")["status"], "active")

    def result(self, lease, unit, value):
        payload = encode_tensors({
            "lora_A": torch.full((2,), float(value)),
            "lora_B": torch.full((1,), float(value)),
        })
        return self.coordinator.submit_result(make_metadata(lease, unit, payload), payload)

    def test_token_weighted_round_closes_and_updates_once(self):
        lease0 = self.coordinator.lease("worker-0")
        lease1 = self.coordinator.lease("worker-1")
        self.assertEqual(lease0["unit"]["unit_id"], "unit-0")
        self.assertEqual(lease1["unit"]["unit_id"], "unit-1")
        first = self.result(lease0, lease0["unit"], 1.0)
        self.assertFalse(first["round_closed"])
        final = self.result(lease1, lease1["unit"], 3.0)
        self.assertTrue(final["round_closed"])
        self.assertEqual(final["received_loss_tokens"], 8)
        self.assertEqual(self.coordinator.close_metrics["unit_count"], 2)
        self.assertEqual(self.coordinator.theta_version, 1)
        self.assertNotEqual(self.coordinator.theta_hash, lease0["theta_hash"])

    def test_snapshot_pairs_bytes_with_their_own_hash_and_version(self):
        before_bytes, before_hash, before_version = self.coordinator.snapshot()
        self.assertEqual(before_version, 0)
        self.assertEqual(before_hash, tensor_map_sha256(decode_tensors(before_bytes)))

        lease0 = self.coordinator.lease("worker-0")
        lease1 = self.coordinator.lease("worker-1")
        self.result(lease0, lease0["unit"], 1.0)
        self.assertTrue(self.result(lease1, lease1["unit"], 3.0)["round_closed"])

        after_bytes, after_hash, after_version = self.coordinator.snapshot()
        self.assertEqual(after_version, 1)
        self.assertNotEqual(after_hash, before_hash)
        self.assertEqual(after_hash, tensor_map_sha256(decode_tensors(after_bytes)))
        self.assertEqual(after_hash, self.coordinator.theta_hash)

    def test_snapshot_is_read_under_the_coordinator_lock(self):
        finished = threading.Event()

        def read_snapshot():
            self.coordinator.snapshot()
            finished.set()

        reader = threading.Thread(target=read_snapshot, daemon=True)
        with self.coordinator.lock:
            reader.start()
            # A reader that ignored the lock would finish while we hold it,
            # and could observe a half-applied round close.
            self.assertFalse(finished.wait(0.2))
        self.assertTrue(finished.wait(5))
        reader.join(timeout=5)

    def test_duplicate_unit_is_rejected(self):
        lease = self.coordinator.lease("worker-0")
        payload = encode_tensors({"lora_A": torch.ones(2), "lora_B": torch.ones(1)})
        metadata = make_metadata(lease, lease["unit"], payload)
        self.coordinator.submit_result(metadata, payload)
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.submit_result(metadata, payload)
        self.assertEqual(caught.exception.code, "duplicate_unit")
        self.assertEqual(self.coordinator.loss_tokens, 2)

    def test_stale_theta_result_requeues_unit(self):
        lease = self.coordinator.lease("worker-0")
        payload = encode_tensors({"lora_A": torch.ones(2), "lora_B": torch.ones(1)})
        metadata = make_metadata(lease, lease["unit"], payload)
        metadata["theta_hash"] = "old-hash"
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.submit_result(metadata, payload)
        self.assertEqual(caught.exception.code, "stale_theta")
        self.assertEqual(self.coordinator.unit_status[lease["unit"]["unit_id"]], "pending")
        next_lease = self.coordinator.lease("worker-1")
        self.assertEqual(next_lease["unit"]["unit_id"], "unit-1")
        reissued_lease = self.coordinator.lease("worker-1")
        self.assertEqual(reissued_lease["unit"]["unit_id"], lease["unit"]["unit_id"])

    def test_wrong_gradient_schema_is_rejected_and_requeued(self):
        lease = self.coordinator.lease("worker-0")
        payload = encode_tensors({"wrong_name": torch.ones(2)})
        metadata = make_metadata(lease, lease["unit"], payload)
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.submit_result(metadata, payload)
        self.assertEqual(caught.exception.code, "gradient_names")
        self.assertEqual(self.coordinator.unit_status[lease["unit"]["unit_id"]], "pending")

    def test_heartbeat_extends_only_the_owning_worker_lease(self):
        lease = self.coordinator.lease("worker-0")
        renewed = self.coordinator.heartbeat(lease["lease_id"], "worker-0")
        self.assertGreater(renewed["lease_expires_at"], lease["lease_expires_at"])
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.heartbeat(lease["lease_id"], "worker-other")
        self.assertEqual(caught.exception.code, "worker_lease_mismatch")

    def test_graceful_release_immediately_requeues_unit(self):
        lease = self.coordinator.lease("worker-0")
        result = self.coordinator.release(lease["lease_id"], "worker-0", "operator_stop")
        self.assertEqual(result["status"], "released")
        self.assertEqual(self.coordinator.unit_status[lease["unit"]["unit_id"]], "pending")
        next_lease = self.coordinator.lease("worker-1")
        self.assertEqual(next_lease["unit"]["unit_id"], lease["unit"]["unit_id"])

    def test_expired_lease_requeues_unit_for_another_worker(self):
        lease = self.coordinator.lease("worker-0")
        self.coordinator.leases[lease["lease_id"]].expires_at = time.time() - 1
        other_lease = self.coordinator.lease("worker-1")
        expired_unit_lease = self.coordinator.lease("worker-1")
        self.assertEqual(other_lease["unit"]["unit_id"], "unit-1")
        self.assertEqual(expired_unit_lease["unit"]["unit_id"], lease["unit"]["unit_id"])
        self.assertTrue(any(event["event"] == "lease_expired" for event in self.coordinator.events))

    def test_training_completion_is_explicit_and_only_after_round_close(self):
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.signal_training_complete("too_early")
        self.assertEqual(caught.exception.code, "round_open")

        lease0 = self.coordinator.lease("worker-0")
        lease1 = self.coordinator.lease("worker-1")
        self.result(lease0, lease0["unit"], 1.0)
        self.result(lease1, lease1["unit"], 3.0)
        self.assertFalse(self.coordinator.descriptor()["training_complete"])
        signal = self.coordinator.signal_training_complete("final_round_complete")
        self.assertTrue(signal["training_complete"])
        self.assertTrue(self.coordinator.descriptor()["training_complete"])
        self.assertEqual(self.coordinator.events[-1]["event"], "training_complete")

    def test_late_worker_waits_for_next_round_and_is_promoted_with_new_snapshot(self):
        lease0 = self.coordinator.lease("worker-0")
        initial_hash = lease0["theta_hash"]
        queued = self.coordinator.join("worker-late")
        self.assertEqual(queued["status"], "waiting_next_round")
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.lease("worker-late")
        self.assertEqual(caught.exception.code, "join_pending")

        lease1 = self.coordinator.lease("worker-1")
        self.result(lease0, lease0["unit"], 1.0)
        self.result(lease1, lease1["unit"], 3.0)
        descriptor = self.coordinator.start_next_round(
            [make_unit("epoch-2/unit-0", 2)], target_loss_tokens=2, min_units=1
        )
        self.assertEqual(descriptor["round_id"], 2)
        self.assertEqual(descriptor["theta_version"], 1)
        self.assertNotEqual(descriptor["theta_hash"], initial_hash)
        self.assertEqual(self.coordinator.join("worker-late")["status"], "active")
        late_lease = self.coordinator.lease("worker-late")
        self.assertEqual(late_lease["round_id"], 2)
        self.assertEqual(late_lease["theta_hash"], descriptor["theta_hash"])


class OptimizerResumeTests(unittest.TestCase):
    ROUND1 = [("r1-u0", 2), ("r1-u1", 6)]
    ROUND2 = [("r2-u0", 2), ("r2-u1", 6)]

    def units(self, spec):
        return [make_unit(unit_id, tokens) for unit_id, tokens in spec]

    def make(self, spec, *, theta=None, theta_version=0, optimizer_tensors=None):
        units = self.units(spec)
        coordinator = RoundCoordinator(
            theta=theta or {"lora_A": torch.zeros(2), "lora_B": torch.zeros(1)},
            units=units,
            run_id="resume-test",
            base={"revision": "pinned"},
            tokenizer={"hash": "pinned"},
            worker_code_sha256="worker-code",
            target_loss_tokens=sum(tokens for _, tokens in spec),
            min_units=len(units),
            lease_seconds=30,
            learning_rate=1e-2,
            theta_version=theta_version,
            optimizer_tensors=optimizer_tensors,
        )
        coordinator.join("worker-0")
        return coordinator

    def close_round(self, coordinator, value):
        while True:
            lease = coordinator.lease("worker-0")
            payload = encode_tensors({
                "lora_A": torch.full((2,), float(value)),
                "lora_B": torch.full((1,), float(value)),
            })
            result = coordinator.submit_result(make_metadata(lease, lease["unit"], payload), payload)
            if result["round_closed"]:
                return

    def test_resuming_from_exported_state_matches_an_uninterrupted_run(self):
        continuous = self.make(self.ROUND1)
        self.close_round(continuous, 1.0)
        continuous.start_next_round(self.units(self.ROUND2), target_loss_tokens=8, min_units=2)
        self.close_round(continuous, -0.5)

        interrupted = self.make(self.ROUND1)
        self.close_round(interrupted, 1.0)
        theta, optimizer_tensors, version, theta_hash = interrupted.export_state()
        self.assertEqual(version, 1)

        resumed = self.make(self.ROUND2, theta=theta, theta_version=version,
                            optimizer_tensors=optimizer_tensors)
        self.assertEqual((resumed.round_id, resumed.theta_version), (2, 1))
        self.assertEqual(resumed.theta_hash, theta_hash)
        self.close_round(resumed, -0.5)

        self.assertEqual(resumed.theta_version, continuous.theta_version)
        for name, value in continuous.theta.items():
            self.assertTrue(torch.equal(resumed.theta[name], value), name)
        self.assertEqual(resumed.theta_hash, continuous.theta_hash)

    def test_dropping_the_adam_moments_changes_the_next_update(self):
        continuous = self.make(self.ROUND1)
        self.close_round(continuous, 1.0)
        continuous.start_next_round(self.units(self.ROUND2), target_loss_tokens=8, min_units=2)
        self.close_round(continuous, -0.5)

        interrupted = self.make(self.ROUND1)
        self.close_round(interrupted, 1.0)
        theta, _, version, _ = interrupted.export_state()
        cold = self.make(self.ROUND2, theta=theta, theta_version=version)
        self.close_round(cold, -0.5)

        self.assertNotEqual(cold.theta_hash, continuous.theta_hash)

    def test_export_has_no_optimizer_state_before_the_first_step(self):
        fresh = self.make(self.ROUND1)
        theta, optimizer_tensors, version, theta_hash = fresh.export_state()
        self.assertEqual((optimizer_tensors, version), ({}, 0))
        self.assertEqual(theta_hash, fresh.theta_hash)

    def test_export_names_every_moment_for_every_parameter(self):
        coordinator = self.make(self.ROUND1)
        self.close_round(coordinator, 1.0)
        _, optimizer_tensors, _, _ = coordinator.export_state()
        self.assertEqual(set(optimizer_tensors), {
            f"{key}/{name}"
            for key in ("exp_avg", "exp_avg_sq", "step")
            for name in ("lora_A", "lora_B")
        })
        self.assertEqual(optimizer_tensors["step/lora_A"].shape, torch.Size([]))

    def test_hyperparameters_match_the_helper_used_before_construction(self):
        coordinator = self.make(self.ROUND1)
        self.assertEqual(coordinator.hyperparameters(), adamw_hyperparameters(1e-2))
        self.assertNotEqual(coordinator.hyperparameters(), adamw_hyperparameters(2e-2))

    def test_invalid_optimizer_state_is_rejected(self):
        parameters = {"a": torch.nn.Parameter(torch.zeros(2)), "b": torch.nn.Parameter(torch.zeros(1))}

        def fresh_optimizer():
            return torch.optim.AdamW(list(parameters.values()), lr=1e-2)

        def full(name, shape):
            return {
                f"exp_avg/{name}": torch.zeros(shape),
                f"exp_avg_sq/{name}": torch.zeros(shape),
                f"step/{name}": torch.tensor(1.0),
            }

        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_optimizer_state_tensors(fresh_optimizer(), parameters, {"exp_avg/zzz": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_optimizer_state_tensors(fresh_optimizer(), parameters, {"exp_avg/a": torch.zeros(2)})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_optimizer_state_tensors(fresh_optimizer(), parameters, {**full("a", (3,)), **full("b", (1,))})
        with self.assertRaisesRegex(ValueError, "only some"):
            load_optimizer_state_tensors(fresh_optimizer(), parameters, full("a", (2,)))
        load_optimizer_state_tensors(fresh_optimizer(), parameters, {})  # empty is a no-op

    def test_negative_theta_version_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make(self.ROUND1, theta_version=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class LeanAggregationTests(unittest.TestCase):
    """Gradients are summed as they arrive and read from disk tensor by tensor."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def make_coordinator(self) -> RoundCoordinator:
        coordinator = RoundCoordinator(
            theta={"lora_A": torch.tensor([0.5, -0.25]), "lora_B": torch.tensor([1.0])},
            units=[make_unit("unit-0", 2), make_unit("unit-1", 6)],
            run_id="lean", base={"revision": "pinned"}, tokenizer={"hash": "pinned"},
            worker_code_sha256="worker-code", target_loss_tokens=8, min_units=2,
            lease_seconds=30, learning_rate=1e-2,
        )
        coordinator.join("w0")
        return coordinator

    def payload_file(self, payload: bytes, name: str = "upload.part", *, sha256=None) -> PayloadFile:
        path = self.dir / name
        path.write_bytes(payload)
        return PayloadFile(path=path, sha256=sha256 or sha256_bytes(payload), size=len(payload))

    def gradients(self, a, b) -> dict:
        return {"lora_A": torch.tensor(a), "lora_B": torch.tensor(b)}

    def submit(self, coordinator, tensors_or_bytes, *, as_file=False, name="upload.part", sha256=None):
        lease = coordinator.lease("w0")
        payload = (tensors_or_bytes if isinstance(tensors_or_bytes, bytes)
                   else encode_tensors(tensors_or_bytes))
        metadata = make_metadata(lease, lease["unit"], payload)
        sent = self.payload_file(payload, name, sha256=sha256) if as_file else payload
        return coordinator.submit_result(metadata, sent)

    def run_round(self, coordinator, *, as_file):
        self.submit(coordinator, self.gradients([1.0, 2.0], [3.0]), as_file=as_file, name="a.part")
        return self.submit(coordinator, self.gradients([0.5, -1.0], [2.0]), as_file=as_file, name="b.part")

    def test_a_streamed_file_gives_exactly_the_update_the_same_bytes_give(self):
        from_bytes, from_file = self.make_coordinator(), self.make_coordinator()
        self.assertTrue(self.run_round(from_bytes, as_file=False)["round_closed"])
        self.assertTrue(self.run_round(from_file, as_file=True)["round_closed"])
        self.assertEqual(from_file.theta_hash, from_bytes.theta_hash)
        theta_a, moments_a, _, _ = from_file.export_state()
        theta_b, moments_b, _, _ = from_bytes.export_state()
        for name in theta_b:
            self.assertTrue(torch.equal(theta_a[name], theta_b[name]))
        for name in moments_b:
            self.assertTrue(torch.equal(moments_a[name], moments_b[name]))

    def test_the_update_is_adamw_on_the_token_normalised_sum(self):
        coordinator = self.make_coordinator()
        self.run_round(coordinator, as_file=True)
        parameters = {"lora_A": torch.nn.Parameter(torch.tensor([0.5, -0.25])),
                      "lora_B": torch.nn.Parameter(torch.tensor([1.0]))}
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-2, weight_decay=0.0)
        parameters["lora_A"].grad = torch.tensor([1.5, 1.0]) / 8
        parameters["lora_B"].grad = torch.tensor([5.0]) / 8
        optimizer.step()
        for name, parameter in parameters.items():
            self.assertTrue(torch.equal(coordinator.theta[name], parameter.detach()), name)

    def test_individual_gradients_are_not_kept_only_their_running_sum(self):
        coordinator = self.make_coordinator()
        first = self.submit(coordinator, self.gradients([1.0, 2.0], [3.0]), as_file=True, name="a.part")
        self.assertFalse(first["round_closed"])
        for result in coordinator.accepted.values():
            self.assertFalse(any(isinstance(value, (torch.Tensor, dict)) for value in result.values()))
        self.assertTrue(torch.equal(coordinator.grad_sums["lora_A"], torch.tensor([1.0, 2.0])))
        self.submit(coordinator, self.gradients([0.5, -1.0], [2.0]), as_file=True, name="b.part")
        self.assertIsNone(coordinator.grad_sums)  # applied and released when the round closed

    def test_a_new_round_starts_from_an_empty_sum(self):
        coordinator = self.make_coordinator()
        self.run_round(coordinator, as_file=True)
        coordinator.start_next_round([make_unit("next-0", 2)], target_loss_tokens=2, min_units=1)
        self.assertIsNone(coordinator.grad_sums)
        coordinator.join("w0")
        self.submit(coordinator, self.gradients([4.0, 4.0], [4.0]), as_file=True, name="c.part")
        self.assertEqual(coordinator.theta_version, 2)

        # The second step must see only its own gradient, with Adam's moments carried over.
        parameters = {"lora_A": torch.nn.Parameter(torch.tensor([0.5, -0.25])),
                      "lora_B": torch.nn.Parameter(torch.tensor([1.0]))}
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-2, weight_decay=0.0)
        for a, b, tokens in (([1.5, 1.0], [5.0], 8), ([4.0, 4.0], [4.0], 2)):
            parameters["lora_A"].grad = torch.tensor(a) / tokens
            parameters["lora_B"].grad = torch.tensor(b) / tokens
            optimizer.step()
        for name, parameter in parameters.items():
            self.assertTrue(torch.equal(coordinator.theta[name], parameter.detach()), name)

    def test_a_file_payload_is_never_decoded_into_memory_as_a_whole(self):
        coordinator = self.make_coordinator()
        with patch.object(rwkv_dist_protocol, "decode_tensors", side_effect=AssertionError("read whole")):
            result = self.submit(coordinator, self.gradients([1.0, 2.0], [3.0]), as_file=True)
        self.assertEqual(result["status"], "accepted")

    def test_the_hash_computed_while_receiving_is_the_one_compared(self):
        coordinator = self.make_coordinator()
        with self.assertRaises(ProtocolError) as caught:
            self.submit(coordinator, self.gradients([1.0, 2.0], [3.0]), as_file=True, sha256="0" * 64)
        self.assertEqual(caught.exception.code, "gradient_hash_mismatch")
        self.assertEqual(coordinator.unit_status["unit-0"], "pending")

    def rejected(self, coordinator, payload: bytes) -> str:
        with self.assertRaises(ProtocolError) as caught:
            self.submit(coordinator, payload, as_file=True, name="bad.part")
        return caught.exception.code

    def test_every_kind_of_bad_file_is_rejected_requeued_and_leaves_the_sum_alone(self):
        for label, payload, code in [
            ("not safetensors", b"definitely not a safetensors file", "invalid_payload"),
            ("truncated", encode_tensors(self.gradients([1.0, 2.0], [3.0]))[:-3], "invalid_payload"),
            ("missing tensor", encode_tensors({"lora_A": torch.ones(2)}), "gradient_names"),
            ("extra tensor", encode_tensors(self.gradients([1.0, 2.0], [3.0]) | {"extra": torch.ones(1)}),
             "gradient_names"),
            ("wrong shape", encode_tensors({"lora_A": torch.ones(3), "lora_B": torch.ones(1)}),
             "gradient_shape"),
            ("nan in the second tensor", encode_tensors(self.gradients([1.0, 2.0], [float("nan")])),
             "nonfinite_gradient"),
            ("inf in the first tensor", encode_tensors(self.gradients([float("inf"), 2.0], [3.0])),
             "nonfinite_gradient"),
        ]:
            with self.subTest(label):
                coordinator = self.make_coordinator()
                self.submit(coordinator, self.gradients([10.0, 20.0], [30.0]), as_file=True, name="good.part")
                before = {k: v.clone() for k, v in coordinator.grad_sums.items()}
                self.assertEqual(self.rejected(coordinator, payload), code)
                # A tensor that fails late must not leave an earlier one half-added.
                for name, value in before.items():
                    self.assertTrue(torch.equal(coordinator.grad_sums[name], value), name)
                self.assertEqual(coordinator.loss_tokens, 2)
                self.assertEqual(len(coordinator.accepted), 1)
                self.assertEqual(coordinator.unit_status["unit-1"], "pending")
                # and the unit can be done properly afterwards
                final = self.submit(coordinator, self.gradients([0.5, -1.0], [2.0]),
                                    as_file=True, name="retry.part")
                self.assertTrue(final["round_closed"])

    def test_a_low_precision_gradient_file_is_summed_in_float32(self):
        coordinator = self.make_coordinator()
        half = {"lora_A": torch.tensor([1.5, 2.5], dtype=torch.bfloat16),
                "lora_B": torch.tensor([3.0], dtype=torch.bfloat16)}
        self.submit(coordinator, half, as_file=True)
        self.assertEqual(coordinator.grad_sums["lora_A"].dtype, torch.float32)
        self.assertTrue(torch.equal(coordinator.grad_sums["lora_A"], torch.tensor([1.5, 2.5])))

    def test_the_snapshot_is_encoded_once_per_adapter_version(self):
        coordinator = self.make_coordinator()
        real = rwkv_dist_protocol.encode_tensors
        with patch.object(rwkv_dist_protocol, "encode_tensors", wraps=real) as encode:
            first = coordinator.snapshot()
            again = coordinator.snapshot()
            self.assertIs(again[0], first[0])
            self.assertIs(coordinator.snapshot_bytes(), first[0])
            self.assertEqual(encode.call_count, 1)
            self.run_round(coordinator, as_file=True)  # the adapter advances
            encode.reset_mock()
            after = coordinator.snapshot()
            coordinator.snapshot()
            self.assertEqual(encode.call_count, 1)
        self.assertIsNot(after[0], first[0])
        self.assertEqual(after[2], 1)
        self.assertEqual(after[1], tensor_map_sha256(decode_tensors(after[0])))
        self.assertEqual(first[1], tensor_map_sha256(decode_tensors(first[0])))
