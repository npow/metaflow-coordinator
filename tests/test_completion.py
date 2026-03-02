"""Unit tests for built-in coordination services using FastAPI TestClient."""
import threading

import pytest
from fastapi.testclient import TestClient


def test_completion_tracker_fires_when_all_workers_complete():
    from metaflow_coordinator import CompletionTracker

    tracker = CompletionTracker(n_workers=3)
    client = TestClient(tracker._app)

    assert not tracker.done.is_set()
    client.post("/complete")
    client.post("/complete")
    assert not tracker.done.is_set()
    client.post("/complete")
    assert tracker.done.is_set()
    assert tracker.completed == 3


def test_completion_tracker_health():
    from metaflow_coordinator import CompletionTracker

    tracker = CompletionTracker(n_workers=2)
    client = TestClient(tracker._app)

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is True
    assert data["completed"] == 0
    assert data["total"] == 2


def test_result_collector_fires_and_stores_results():
    from metaflow_coordinator import ResultCollector

    collector = ResultCollector(n_workers=2)
    client = TestClient(collector._app)

    assert not collector.done.is_set()
    client.post("/submit", json={"worker_id": "a", "result": 10})
    assert not collector.done.is_set()
    client.post("/submit", json={"worker_id": "b", "result": 20})
    assert collector.done.is_set()
    assert collector.results_by_worker == {"a": 10, "b": 20}
    assert set(collector.results) == {10, 20}


def test_work_queue_distributes_and_collects():
    from metaflow_coordinator import WorkQueue

    wq = WorkQueue(items=["x", "y", "z"])
    client = TestClient(wq._app)

    # Pull all items
    pulled = []
    for _ in range(3):
        r = client.post("/pull", json={}).json()
        assert not r["done"]
        pulled.append((r["item_id"], r["payload"]))

    # Queue exhausted
    r = client.post("/pull", json={}).json()
    assert r["done"]

    # Submit results
    for item_id, payload in pulled:
        client.post("/submit", json={"item_id": item_id, "result": payload.upper()})

    assert wq.done.is_set()
    assert wq.results == ["X", "Y", "Z"]


def test_barrier_releases_after_all_arrive():
    from metaflow_coordinator import BarrierService

    barrier = BarrierService(n_workers=2, n_rounds=1)
    client = TestClient(barrier._app)

    # First worker arrives — not released yet
    client.post("/arrive/0", json={"worker_id": "w0"})
    r = client.get("/released/0").json()
    assert not r["released"]

    # Second worker arrives — now released
    client.post("/arrive/0", json={"worker_id": "w1"})
    r = client.get("/released/0").json()
    assert r["released"]
    assert barrier.done.is_set()


def test_semaphore_service_lifecycle():
    from metaflow_coordinator import SemaphoreService

    sem = SemaphoreService(max_concurrent=2, n_workers=2)
    # Use context manager so the startup event fires (initializes asyncio.Semaphore)
    with TestClient(sem._app) as client:
        # Two workers acquire and release
        r = client.post("/acquire", json={})
        assert r.json()["ok"]
        client.post("/release", json={})

        r = client.post("/acquire", json={})
        assert r.json()["ok"]
        client.post("/release", json={})

        # Both workers signal done
        client.post("/done", json={})
        assert not sem.done.is_set()
        client.post("/done", json={})
        assert sem.done.is_set()
