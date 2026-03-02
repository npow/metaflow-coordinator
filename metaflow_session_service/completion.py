"""
Built-in coordination services.

CompletionTracker  — fires when N workers call POST /complete.
ResultCollector    — fires when N workers call POST /submit; stores their results.
BarrierService     — N-of-N synchronization barrier for multi-round coordination.
WorkQueue          — pull-based work distribution; fires when all items are submitted.
SemaphoreService   — concurrency limiter; fires when N workers call POST /done.
"""
from __future__ import annotations

import threading

from .service import FastAPIService


class CompletionTracker(FastAPIService):
    """
    A lightweight coordination service that counts N worker completions.

    Args:
        n_workers:          Number of workers expected to call POST /complete.
        port:               Preferred port (auto-discovered if None).
        timeout:            Maximum seconds to wait before giving up.
        per_worker_timeout: If set, the done event is fired (and ``timed_out``
                            is set to True) if not all workers complete within
                            this many seconds of the *first* completion.

    The ``done`` property is a threading.Event that fires when all N workers
    have reported (or when the per-worker timeout fires).  Pass it to other
    services' ``done`` parameter to tie their lifetime to the tracker::

        tracker = CompletionTracker(n_workers=20, per_worker_timeout=300)
        redis = ProcessService(
            command=["redis-server", "--port", "{port}"],
            done=tracker.done,
            url_scheme="redis",
        )
    """

    def __init__(
        self,
        n_workers: int,
        port: int | None = None,
        timeout: int = 7200,
        per_worker_timeout: int | None = None,
    ):
        from fastapi import FastAPI

        n_workers_ = n_workers
        state = {"completed": 0}
        timed_out = [False]
        watchdog: list = [None]
        lock = threading.Lock()
        done_event = threading.Event()

        def _on_timeout() -> None:
            with lock:
                if not done_event.is_set():
                    timed_out[0] = True
                    done_event.set()

        app = FastAPI(title="completion-tracker")

        @app.post("/complete")
        async def complete():
            with lock:
                state["completed"] += 1
                if per_worker_timeout is not None and state["completed"] == 1:
                    t = threading.Timer(per_worker_timeout, _on_timeout)
                    t.daemon = True
                    watchdog[0] = t
                    t.start()
                if state["completed"] >= n_workers_:
                    if watchdog[0] is not None:
                        watchdog[0].cancel()
                    done_event.set()
            return {"completed": state["completed"], "total": n_workers_}

        @app.get("/health")
        async def health():
            return {
                "ready": True,
                "completed": state["completed"],
                "total": n_workers_,
                "timed_out": timed_out[0],
            }

        super().__init__(app=app, done=done_event, port=port, timeout=timeout)

        self._state = state
        self._timed_out = timed_out
        self._n_workers = n_workers_
        self._done_event = done_event

    @property
    def done(self) -> threading.Event:
        """Event that fires when all workers complete (or timeout fires)."""
        return self._done_event

    @property
    def completed(self) -> int:
        """Number of workers that have called POST /complete so far."""
        return self._state["completed"]

    @property
    def timed_out(self) -> bool:
        """True if the per_worker_timeout fired before all workers completed."""
        return self._timed_out[0]

    @property
    def n_workers(self) -> int:
        return self._n_workers


class ResultCollector(FastAPIService):
    """
    Collects one result per worker.  Fires done_event when all N workers
    have submitted.

    Workers POST to /submit::

        httpx.post(f"{url}/submit", json={"worker_id": self.input, "result": my_result})

    Coordinator reads results after ``run()`` returns::

        collector = ResultCollector(n_workers=20)
        group = SessionServiceGroup({"collector": collector, ...})
        group.run(service_id=self.run_id, namespace="my-flow")
        print(collector.results)           # list in submission order
        print(collector.results_by_worker) # dict keyed by worker_id

    Args:
        n_workers: Number of workers expected to POST /submit.
        port:      Preferred port (auto-discovered if None).
        timeout:   Maximum seconds to wait before giving up.
    """

    def __init__(
        self,
        n_workers: int,
        port: int | None = None,
        timeout: int = 7200,
    ):
        from fastapi import FastAPI

        n_workers_ = n_workers
        results_by_worker: dict = {}
        results_list: list = []
        lock = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title="result-collector")

        @app.post("/submit")
        async def submit(body: dict):
            worker_id = body.get("worker_id")
            result = body.get("result")
            with lock:
                results_by_worker[worker_id] = result
                results_list.append(result)
                if len(results_by_worker) >= n_workers_:
                    done_event.set()
            return {"collected": len(results_by_worker), "total": n_workers_}

        @app.get("/health")
        async def health():
            return {
                "ready": True,
                "collected": len(results_by_worker),
                "total": n_workers_,
            }

        super().__init__(app=app, done=done_event, port=port, timeout=timeout)

        self._results_by_worker = results_by_worker
        self._results_list = results_list
        self._n_workers = n_workers_
        self._done_event = done_event

    @property
    def done(self) -> threading.Event:
        return self._done_event

    @property
    def results(self) -> list:
        """Results in submission order."""
        return list(self._results_list)

    @property
    def results_by_worker(self) -> dict:
        """Results keyed by worker_id."""
        return dict(self._results_by_worker)

    @property
    def n_workers(self) -> int:
        return self._n_workers


class BarrierService(FastAPIService):
    """
    N-of-N synchronization barrier for multi-round coordination.

    All N workers must arrive at round R before any worker is released
    to proceed to round R+1.  Fires done_event after all rounds complete.

    Workers::

        for round_num in range(n_rounds):
            # ... do work for this round ...

            # signal arrival and wait for release
            httpx.post(f"{barrier_url}/arrive/{round_num}",
                       json={"worker_id": self.input})
            while True:
                r = httpx.get(f"{barrier_url}/released/{round_num}").json()
                if r["released"]:
                    break
                time.sleep(0.1)

    Args:
        n_workers: Number of workers that must arrive before each release.
        n_rounds:  Number of rounds; done_event fires after the last round.
        port:      Preferred port (auto-discovered if None).
        timeout:   Maximum seconds to wait before giving up.
    """

    def __init__(
        self,
        n_workers: int,
        n_rounds: int = 1,
        port: int | None = None,
        timeout: int = 7200,
    ):
        from fastapi import FastAPI

        n_workers_ = n_workers
        n_rounds_ = n_rounds
        arrivals: dict[int, set] = {}
        released: set[int] = set()
        rounds_complete = [0]
        lock = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title="barrier")

        @app.post("/arrive/{round_num}")
        async def arrive(round_num: int, body: dict):
            """Worker signals arrival at barrier for round_num."""
            worker_id = body.get("worker_id")
            with lock:
                if round_num not in arrivals:
                    arrivals[round_num] = set()
                arrivals[round_num].add(worker_id)
                if len(arrivals[round_num]) >= n_workers_:
                    released.add(round_num)
                    rounds_complete[0] += 1
                    if rounds_complete[0] >= n_rounds_:
                        done_event.set()
            return {
                "arrived": len(arrivals.get(round_num, set())),
                "total": n_workers_,
            }

        @app.get("/released/{round_num}")
        async def is_released(round_num: int):
            """Returns {"released": true} once all workers have arrived."""
            return {"released": round_num in released}

        @app.get("/health")
        async def health():
            with lock:
                return {
                    "rounds_complete": rounds_complete[0],
                    "total_rounds": n_rounds_,
                }

        super().__init__(app=app, done=done_event, port=port, timeout=timeout)

        self._done_event = done_event
        self._rounds_complete = rounds_complete
        self._n_workers = n_workers_
        self._n_rounds = n_rounds_

    @property
    def done(self) -> threading.Event:
        return self._done_event

    @property
    def rounds_complete(self) -> int:
        """Number of rounds that have been fully released."""
        return self._rounds_complete[0]

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def n_rounds(self) -> int:
        return self._n_rounds


class WorkQueue(FastAPIService):
    """
    Pull-based work distribution service.

    The coordinator pre-loads N items; workers call POST /pull to get the next
    available item, process it, then POST /submit to return the result.
    ``done_event`` fires when all N items have been submitted.

    Workers::

        while True:
            try:
                item = httpx.post(f"{url}/pull", json={}).json()
            except HTTPX_ERRORS:
                break
            if item.get("done"):
                break
            result = process(item["payload"])
            httpx.post(f"{url}/submit",
                       json={"item_id": item["item_id"], "result": result})

    Coordinator reads results after ``run()`` returns::

        wq = WorkQueue(items=records, drain_delay=2.0)
        wq.run(service_id=self.coordinator_id, namespace="my-flow")
        print(wq.results_by_item)   # {0: result, 1: result, ...}
        print(wq.results)           # [result, result, ...] in item order

    Args:
        items:       List of payloads to distribute (any JSON-serializable values).
        port:        Preferred port (auto-discovered if None).
        timeout:     Maximum seconds to wait before giving up.
        drain_delay: Seconds after ``done_event`` before stopping (default 2.0).
    """

    def __init__(
        self,
        items: list,
        port: int | None = None,
        timeout: int = 7200,
        drain_delay: float = 2.0,
    ):
        from collections import deque
        from fastapi import FastAPI

        n_items = len(items)
        queue: deque = deque(enumerate(items))
        results_by_item: dict = {}
        lock = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title="work-queue")

        @app.post("/pull")
        async def pull(body: dict):
            """Return the next item, or {\"done\": true} if the queue is empty."""
            with lock:
                if queue:
                    item_id, payload = queue.popleft()
                    return {"done": False, "item_id": item_id, "payload": payload}
                return {"done": True}

        @app.post("/submit")
        async def submit(body: dict):
            """Submit the result for a completed item."""
            item_id = body["item_id"]
            result = body.get("result")
            with lock:
                results_by_item[item_id] = result
                if len(results_by_item) >= n_items:
                    done_event.set()
            return {"ok": True, "completed": len(results_by_item), "total": n_items}

        @app.get("/health")
        async def health():
            with lock:
                return {
                    "queued": len(queue),
                    "completed": len(results_by_item),
                    "total": n_items,
                }

        super().__init__(
            app=app, done=done_event, port=port, timeout=timeout,
            drain_delay=drain_delay,
        )

        self._items = list(items)
        self._results_by_item = results_by_item
        self._done_event = done_event
        self._n_items = n_items

    @property
    def done(self) -> threading.Event:
        """Event that fires when all items have been submitted."""
        return self._done_event

    @property
    def results_by_item(self) -> dict:
        """Dict mapping item_id (int index) to the submitted result."""
        return dict(self._results_by_item)

    @property
    def results(self) -> list:
        """Results in item_id order; ``None`` for items not yet submitted."""
        return [self._results_by_item.get(i) for i in range(self._n_items)]

    @property
    def n_items(self) -> int:
        return self._n_items


class SemaphoreService(FastAPIService):
    """
    Concurrency-limiting semaphore service.

    Workers call POST /acquire (async-blocking until a slot is free), do their
    work, then POST /release.  Call POST /done when finished.  ``done_event``
    fires when ``n_workers`` have called POST /done.

    Example::

        # Coordinator step:
        sem = SemaphoreService(max_concurrent=3, n_workers=self.n_workers_int)
        sem.run(service_id=self.coordinator_id, namespace="my-flow")
        self.semaphore_stats = sem.stats

        # Worker step:
        for item in my_items:
            httpx.post(f"{url}/acquire", json={"worker_id": self.input}, timeout=120)
            result = call_api(item)
            httpx.post(f"{url}/release", json={})
        httpx.post(f"{url}/done", json={})

    Args:
        max_concurrent: Maximum concurrent workers inside the critical section.
        n_workers:      Number of workers that must call POST /done before shutdown.
        port:           Preferred port (auto-discovered if None).
        timeout:        Maximum seconds to wait before giving up.
        drain_delay:    Seconds after ``done_event`` before stopping (default 2.0).
    """

    def __init__(
        self,
        max_concurrent: int,
        n_workers: int,
        port: int | None = None,
        timeout: int = 7200,
        drain_delay: float = 2.0,
    ):
        import asyncio
        from fastapi import FastAPI

        n_workers_ = n_workers
        max_concurrent_ = max_concurrent
        stats: dict = {"calls": 0, "total_wait_ms": 0.0, "peak_wait_ms": 0.0}
        done_workers = [0]
        lock = threading.Lock()
        done_event = threading.Event()

        _sem: list[asyncio.Semaphore] = []

        app = FastAPI(title="semaphore")

        @app.on_event("startup")
        async def _startup():
            _sem.append(asyncio.Semaphore(max_concurrent_))

        @app.post("/acquire")
        async def acquire(body: dict):
            """Async-block until a concurrency slot is free."""
            t0 = asyncio.get_running_loop().time()
            await _sem[0].acquire()
            wait_ms = (asyncio.get_running_loop().time() - t0) * 1000
            with lock:
                stats["calls"] += 1
                stats["total_wait_ms"] += wait_ms
                if wait_ms > stats["peak_wait_ms"]:
                    stats["peak_wait_ms"] = wait_ms
            return {"ok": True, "wait_ms": round(wait_ms, 1)}

        @app.post("/release")
        async def release(body: dict):
            """Release a concurrency slot."""
            _sem[0].release()
            return {"ok": True}

        @app.post("/done")
        async def worker_done(body: dict):
            """Signal that this worker is finished; fires done_event when all done."""
            with lock:
                done_workers[0] += 1
                if done_workers[0] >= n_workers_:
                    done_event.set()
            return {"ok": True, "workers_done": done_workers[0]}

        @app.get("/health")
        async def health():
            slots_used = (max_concurrent_ - _sem[0]._value) if _sem else 0
            return {
                "slots_used": slots_used,
                "max_concurrent": max_concurrent_,
                "workers_done": done_workers[0],
                "total_workers": n_workers_,
            }

        super().__init__(
            app=app, done=done_event, port=port, timeout=timeout,
            drain_delay=drain_delay,
        )

        self._stats = stats
        self._done_event = done_event
        self._n_workers = n_workers_
        self._max_concurrent = max_concurrent_

    @property
    def done(self) -> threading.Event:
        """Event that fires when all n_workers have called POST /done."""
        return self._done_event

    @property
    def stats(self) -> dict:
        """Semaphore usage stats: calls, total_wait_ms, peak_wait_ms."""
        return dict(self._stats)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent
