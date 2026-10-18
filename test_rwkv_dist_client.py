from __future__ import annotations

import json
import unittest

from rwkv_dist_client import (
    CoordinatorHTTPError,
    CoordinatorUnavailable,
    CoordinatorWaitExpired,
    CoordinatorWaiter,
    StaleLease,
    check_lease_current,
    fetch_leased_snapshot,
    release_stale_lease,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.timeouts = []
        self.calls = []

    def request(self, path, *, method="GET", body=None, headers=None, timeout=30.0):
        self.timeouts.append(timeout)
        self.calls.append((path, method, body))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CoordinatorWaiterTests(unittest.TestCase):
    def test_retries_then_resumes_and_clears_outage(self):
        clock = FakeClock()
        client = FakeClient([
            CoordinatorUnavailable("connection refused"),
            CoordinatorUnavailable("connection refused"),
            (200, {}, b"ok"),
        ])
        statuses = []
        waiter = CoordinatorWaiter(
            client, max_wait_seconds=20, request_timeout_seconds=5,
            sleep=clock.sleep, monotonic=clock.monotonic,
            on_status=lambda *args: statuses.append(args),
        )
        self.assertEqual(waiter.request("/health")[0], 200)
        self.assertIsNone(waiter.outage_started_at)
        self.assertEqual(clock.now, 3.0)
        self.assertEqual([item[0] for item in statuses], [
            "coordinator_unavailable", "coordinator_reconnected"
        ])

    def test_stops_at_shared_three_hour_window(self):
        clock = FakeClock()
        client = FakeClient([CoordinatorUnavailable("offline")] * 10)
        waiter = CoordinatorWaiter(
            client, max_wait_seconds=5, request_timeout_seconds=2,
            sleep=clock.sleep, monotonic=clock.monotonic,
        )
        waiter.note_unavailable("heartbeat failed")
        with self.assertRaises(CoordinatorWaitExpired):
            waiter.request("/round")
        self.assertEqual(clock.now, 5.0)
        self.assertEqual(len(client.timeouts), 3)

    def test_wait_window_cannot_exceed_three_hours(self):
        with self.assertRaises(ValueError):
            CoordinatorWaiter(FakeClient([]), max_wait_seconds=10801)


LEASE = {"lease_id": "lease-1", "theta_hash": "hash-0", "theta_version": 0}


class StaleLeaseTests(unittest.TestCase):
    def test_matching_lease_and_descriptor_pass(self):
        check_lease_current(LEASE, {"theta_hash": "hash-0", "theta_version": 0})

    def test_hash_or_version_mismatch_is_stale_not_fatal(self):
        with self.assertRaises(StaleLease):
            check_lease_current(LEASE, {"theta_hash": "hash-1", "theta_version": 0})
        with self.assertRaises(StaleLease):
            check_lease_current(LEASE, {"theta_hash": "hash-0", "theta_version": 1})

    def test_snapshot_is_returned_only_when_header_matches_the_lease(self):
        ok = lambda path: (200, {"X-Theta-Hash": "hash-0"}, b"adapter-bytes")
        self.assertEqual(fetch_leased_snapshot(ok, LEASE), b"adapter-bytes")

    def test_snapshot_with_other_or_missing_header_is_stale(self):
        newer = lambda path: (200, {"X-Theta-Hash": "hash-1"}, b"newer")
        headerless = lambda path: (200, {}, b"unknown")
        with self.assertRaises(StaleLease):
            fetch_leased_snapshot(newer, LEASE)
        with self.assertRaises(StaleLease):
            fetch_leased_snapshot(headerless, LEASE)

    def test_release_posts_the_lease_and_reports_success(self):
        client = FakeClient([(200, {}, b'{"status": "released"}')])
        outcome = release_stale_lease(client, LEASE, "worker-0", "stale")
        self.assertEqual(outcome, "released")
        path, method, body = client.calls[0]
        self.assertEqual((path, method), ("/release", "POST"))
        self.assertEqual(json.loads(body), {
            "lease_id": "lease-1", "worker_id": "worker-0", "reason": "stale",
        })

    def test_release_never_raises_on_coordinator_replies(self):
        gone = FakeClient([CoordinatorHTTPError(409, {"error": "unknown_lease"})])
        self.assertEqual(release_stale_lease(gone, LEASE, "worker-0"), "already_gone")
        denied = FakeClient([CoordinatorHTTPError(403, {"error": "worker_lease_mismatch"})])
        self.assertEqual(release_stale_lease(denied, LEASE, "worker-0"), "rejected_http_403")
        down = FakeClient([CoordinatorUnavailable("offline")])
        self.assertEqual(release_stale_lease(down, LEASE, "worker-0"), "release_unreachable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
