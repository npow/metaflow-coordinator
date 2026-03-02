"""
SessionService ABC and concrete implementations.

FastAPIService  — in-process uvicorn serving a FastAPI app
ProcessService  — external subprocess (Redis, Postgres, any daemon)
SessionServiceGroup — run multiple services concurrently in one step
"""
from __future__ import annotations

import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting at {start}")


def _get_local_ip() -> str:
    """
    Returns 127.0.0.1 for local runs (all tasks on the same host),
    or the VPC private IP for remote runs (AWS Batch / Kubernetes).
    """
    from .rendezvous import _is_remote
    if not _is_remote():
        return "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Readiness strategies
# ---------------------------------------------------------------------------

class SocketReady:
    """Ready when a TCP connection to host:port succeeds."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port

    def is_ready(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False


class HttpReady:
    """Ready when a GET to base_url+path returns HTTP < 500."""

    def __init__(self, path: str = "/health"):
        self.path = path
        self._base_url: str = ""

    def is_ready(self) -> bool:
        if not self._base_url:
            return False
        try:
            import httpx
            r = httpx.get(f"{self._base_url}{self.path}", timeout=2)
            return r.status_code < 500
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SessionService(ABC):
    """
    A service scoped to one Metaflow run.

    Implementations must block inside ``run()`` until they are done,
    then tear down cleanly.
    """

    @abstractmethod
    def run(self, service_id: str, namespace: str = "metaflow-svc") -> None:
        """Start the service, register its URL, block until done, then stop."""

    @property
    def url(self) -> str | None:
        """The registered URL; set after ``run()`` begins, None before."""
        return None


# ---------------------------------------------------------------------------
# FastAPIService
# ---------------------------------------------------------------------------

class FastAPIService(SessionService):
    """
    Run a FastAPI app inside the current process using uvicorn.

    Args:
        app:     FastAPI application instance.
        done:    threading.Event that the app sets when it is done.
                 ``run()`` blocks until this event fires (or timeout).
        port:    Preferred port; a free port near this value is auto-discovered.
        timeout: Maximum seconds to wait for ``done`` before giving up.
    """

    def __init__(
        self,
        app: Any,
        done: threading.Event,
        port: int | None = None,
        timeout: int = 7200,
        drain_delay: float = 0.0,
    ):
        self._app = app
        self._done = done
        self._port = port
        self._timeout = timeout
        self._drain_delay = drain_delay
        self._url: str | None = None

    @property
    def url(self) -> str | None:
        return self._url

    def run(self, service_id: str, namespace: str = "metaflow-svc") -> None:
        import uvicorn
        from .rendezvous import register_service

        actual_port = _find_free_port(self._port or 8765)
        ip = _get_local_ip()
        url = f"http://{ip}:{actual_port}"
        self._url = url

        register_service(namespace=namespace, service_id=service_id, url=url)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=actual_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Give uvicorn a moment to bind
        time.sleep(1.5)

        print(
            f"[metaflow-session-service] {service_id!r} listening on {url}"
        )

        fired = self._done.wait(timeout=self._timeout)
        if not fired:
            print(
                f"[metaflow-session-service] {service_id!r} timed out after {self._timeout}s"
            )

        if self._drain_delay > 0:
            time.sleep(self._drain_delay)
        server.should_exit = True


# ---------------------------------------------------------------------------
# ProcessService
# ---------------------------------------------------------------------------

class ProcessService(SessionService):
    """
    Run an external subprocess as a session service.

    Use ``{port}`` in ``command`` — it is substituted with the auto-discovered
    free port at runtime.  The default readiness check waits for the port to
    accept TCP connections.

    Args:
        command:    Command list; ``{port}`` is substituted before execution.
        done:       threading.Event that signals when the process should stop.
        url_scheme: URL scheme for the registered URL (e.g. ``"redis"``).
        timeout:    Maximum seconds to wait for ``done``.
        ready:      Readiness strategy; defaults to SocketReady on the chosen port.
        start_timeout: Seconds to wait for the process to become ready.
    """

    def __init__(
        self,
        command: list[str],
        done: threading.Event,
        url_scheme: str = "tcp",
        port: int | None = None,
        timeout: int = 7200,
        ready: SocketReady | HttpReady | None = None,
        start_timeout: int = 30,
    ):
        self._command = command
        self._done = done
        self._url_scheme = url_scheme
        self._port = port
        self._timeout = timeout
        self._ready_override = ready
        self._start_timeout = start_timeout
        self._url: str | None = None

    @property
    def url(self) -> str | None:
        return self._url

    def run(self, service_id: str, namespace: str = "metaflow-svc") -> None:
        from .rendezvous import register_service
        from .exceptions import ServiceNotReadyError

        actual_port = _find_free_port(self._port or 8765)
        ip = _get_local_ip()

        cmd = [arg.format(port=actual_port) for arg in self._command]

        ready = self._ready_override
        if ready is None:
            ready = SocketReady(host="127.0.0.1", port=actual_port)
        elif isinstance(ready, HttpReady):
            ready._base_url = f"http://127.0.0.1:{actual_port}"

        proc = subprocess.Popen(cmd)

        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if ready.is_ready():
                break
            time.sleep(0.2)
        else:
            proc.terminate()
            raise ServiceNotReadyError(
                f"Process {cmd[0]!r} did not become ready within {self._start_timeout}s"
            )

        url = f"{self._url_scheme}://{ip}:{actual_port}"
        self._url = url
        register_service(namespace=namespace, service_id=service_id, url=url)
        print(f"[metaflow-session-service] process {service_id!r} at {url}")

        fired = self._done.wait(timeout=self._timeout)
        if not fired:
            print(
                f"[metaflow-session-service] process {service_id!r} timed out after {self._timeout}s"
            )

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# SessionServiceGroup
# ---------------------------------------------------------------------------

class SessionServiceGroup:
    """
    Run multiple named services concurrently inside one step.

    Each service is registered under ``{service_id}/{role}`` in the rendezvous.
    ``run()`` blocks until ALL services have completed.

    Example::

        tracker = CompletionTracker(n_workers=20)
        redis = ProcessService(
            command=["redis-server", "--port", "{port}"],
            done=tracker.done,
            url_scheme="redis",
        )
        group = SessionServiceGroup({"redis": redis, "tracker": tracker})
        group.run(service_id=self.run_id, namespace="my-flow")
    """

    def __init__(self, services: dict[str, SessionService]):
        self._services = services

    def run(self, service_id: str, namespace: str = "metaflow-svc") -> None:
        errors: list[Exception] = []
        threads: list[threading.Thread] = []

        # Pre-allocate a distinct port for each service so that concurrent threads
        # cannot both find the same free port.  Do this sequentially in the calling
        # thread before spawning anything.  Both FastAPIService and ProcessService
        # expose self._port which run() will use as the starting point for the search.
        next_start = 8765
        for svc in self._services.values():
            if hasattr(svc, "_port") and svc._port is None:
                svc._port = _find_free_port(next_start)
                next_start = svc._port + 1  # search above this port for the next service

        def run_one(role: str, svc: SessionService) -> None:
            try:
                svc.run(service_id=f"{service_id}/{role}", namespace=namespace)
            except Exception as exc:
                errors.append(exc)

        for role, svc in self._services.items():
            t = threading.Thread(target=run_one, args=(role, svc), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if errors:
            raise errors[0]
