from __future__ import annotations

import unittest

from rwkv_dist_client import (
    CoordinatorUnavailable,
    CoordinatorWaitExpired,
    CoordinatorWaiter,
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

    def request(self, path, *, method="GET", body=None, headers=None, timeout=30.0):
        self.timeouts.append(timeout)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
