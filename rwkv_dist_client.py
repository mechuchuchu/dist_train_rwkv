"""HTTP client and bounded coordinator-outage wait used by RWKV workers."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Callable


WORKER_BUNDLE_FILES = (
    "g1j_dist_worker.py",
    "rwkv_dist_protocol.py",
    "rwkv_dist_client.py",
    "rwkv_hub.py",
)


class CoordinatorUnavailable(RuntimeError):
    """A transport or server failure that may be temporary."""


class CoordinatorHTTPError(RuntimeError):
    def __init__(self, status: int, detail: object):
        super().__init__(f"coordinator returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class CoordinatorWaitExpired(RuntimeError):
    def __init__(self, waited_seconds: float):
        super().__init__(f"coordinator unavailable for {waited_seconds:.1f}s")
        self.waited_seconds = waited_seconds


class Client:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, path: str, *, method="GET", body: bytes | None = None,
                headers: dict[str, str] | None = None, timeout: float = 30.0):
        request_headers = {"Authorization": f"Bearer {self.token}"}
        request_headers.update(headers or {})
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                detail = json.loads(payload)
            except Exception:
                detail = payload[:1000].decode("utf-8", "replace")
            if exc.code in (408, 425, 429) or exc.code >= 500:
                raise CoordinatorUnavailable(f"HTTP {exc.code}: {detail}") from exc
            raise CoordinatorHTTPError(exc.code, detail) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CoordinatorUnavailable(str(exc)) from exc

    def get_json(self, path: str, *, timeout: float = 30.0):
        _, _, body = self.request(path, timeout=timeout)
        return json.loads(body)


class CoordinatorWaiter:
    """Retry transient failures for one shared, bounded outage window."""

    def __init__(self, client: Client, *, max_wait_seconds: float = 10800.0,
                 request_timeout_seconds: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic,
                 on_status: Callable[[str, float, str], None] | None = None):
        if max_wait_seconds <= 0 or max_wait_seconds > 10800:
            raise ValueError("max_wait_seconds must be in (0, 10800]")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.client = client
        self.max_wait_seconds = float(max_wait_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.sleep = sleep
        self.monotonic = monotonic
        self.on_status = on_status
        self.lock = threading.RLock()
        self.outage_started_at: float | None = None
        self.last_error = ""

    def note_unavailable(self, error: object, *, started_at: float | None = None) -> None:
        now = self.monotonic()
        with self.lock:
            first = self.outage_started_at is None
            if first:
                self.outage_started_at = now if started_at is None else started_at
            self.last_error = str(error)
        if first and self.on_status:
            self.on_status("coordinator_unavailable", 0.0, str(error))

    def note_recovered(self) -> None:
        with self.lock:
            if self.outage_started_at is None:
                return
            elapsed = self.monotonic() - self.outage_started_at
            last_error = self.last_error
            self.outage_started_at = None
            self.last_error = ""
        if self.on_status:
            self.on_status("coordinator_reconnected", elapsed, last_error)

    def remaining_wait(self) -> float:
        with self.lock:
            started = self.outage_started_at
        if started is None:
            return self.max_wait_seconds
        return self.max_wait_seconds - (self.monotonic() - started)

    def request(self, path: str, *, method="GET", body: bytes | None = None,
                headers: dict[str, str] | None = None):
        delay = 1.0
        while True:
            remaining = self.remaining_wait()
            if remaining <= 0:
                waited = self.max_wait_seconds
                with self.lock:
                    last_error = self.last_error
                if self.on_status:
                    self.on_status("coordinator_wait_expired", waited, last_error)
                raise CoordinatorWaitExpired(waited)
            timeout = min(self.request_timeout_seconds, remaining)
            attempt_started_at = self.monotonic()
            try:
                response = self.client.request(
                    path, method=method, body=body, headers=headers, timeout=timeout
                )
            except CoordinatorUnavailable as exc:
                self.note_unavailable(exc, started_at=attempt_started_at)
                remaining = self.remaining_wait()
                if remaining <= 0:
                    if self.on_status:
                        self.on_status("coordinator_wait_expired", self.max_wait_seconds, str(exc))
                    raise CoordinatorWaitExpired(self.max_wait_seconds) from exc
                self.sleep(min(delay, remaining))
                delay = min(delay * 2.0, 30.0)
            else:
                self.note_recovered()
                return response

    def get_json(self, path: str):
        _, _, body = self.request(path)
        return json.loads(body)
