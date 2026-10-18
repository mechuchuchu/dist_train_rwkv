from __future__ import annotations

import unittest
import time

import torch

from rwkv_dist_protocol import (
    ProtocolError,
    RoundCoordinator,
    encode_tensors,
    sha256_bytes,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
