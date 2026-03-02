"""
Staged Pipeline — items flow through ordered processing stages.

All previous examples use a star topology where workers do the same job.
This example adds a second dimension: workers handle any stage, and the
coordinator owns the routing.  When a worker completes stage S for item I,
the coordinator places I into the stage-S+1 queue.  Workers never know the
pipeline structure — they just call POST /next, do the work, POST /complete.

Use case: log record enrichment.  Raw log strings flow through three stages:
  parse    → extract IP, level, message from raw text
  enrich   → resolve service name from IP, add numeric severity
  classify → categorise event type, recommend action (page-oncall / ticket)

Each stage transforms the item; the output of stage S is the input to S+1.
Workers are interchangeable across stages.  The coordinator's single asyncio
queue holds all in-flight work regardless of stage.

Architecture:
   start ─── run_coordinator (pipeline queue + stage router)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Protocol:
   Worker → POST /next         (blocks; returns next available (item, stage) pair)
   Worker → POST /complete     (submits stage result; coordinator routes to next stage)

Run:
    python examples/staged_pipeline.py run --n_workers 3
"""
import threading
import time

from metaflow import FlowSpec, Parameter, current, step

from metaflow_session_service import FastAPIService, await_service, coordinator_join, worker_join

# ── Dataset ───────────────────────────────────────────────────────────────────

RAW_LOGS = [
    "10.0.0.1 ERROR Connection refused to db.internal:5432 after 30s",
    "10.0.0.2 INFO  GET /api/users 200 145ms",
    "10.0.0.3 WARN  Memory usage at 87% on node-03",
    "10.0.0.1 ERROR Deadlock detected in transaction tx-8824",
    "10.0.0.4 INFO  Scheduled job cleanup completed in 1.2s",
    "10.0.0.2 DEBUG Cache miss for key user:4421",
    "10.0.0.5 ERROR Failed to publish to Kafka topic events after 3 retries",
    "10.0.0.3 WARN  Disk usage at 92% — GC triggered",
    "10.0.0.1 ERROR Auth token expired for service account admin",
    "10.0.0.4 INFO  Replica sync completed lag=0ms",
    "10.0.0.2 INFO  POST /api/checkout 201 890ms",
    "10.0.0.5 WARN  Circuit breaker OPEN for payment-service",
]

_IP_TO_SERVICE = {
    "10.0.0.1": "auth-service",
    "10.0.0.2": "api-gateway",
    "10.0.0.3": "node-manager",
    "10.0.0.4": "data-sync",
    "10.0.0.5": "event-bus",
}

_LEVEL_SEVERITY = {"ERROR": 3, "WARN": 2, "INFO": 1, "DEBUG": 0}

_CATEGORIES = {
    "connectivity": ["refused", "timeout", "connect", "circuit breaker"],
    "auth":         ["token", "auth", "expired", "permission"],
    "resource":     ["memory", "disk", "cpu", "gc"],
    "data":         ["deadlock", "transaction", "sync", "replica", "cache"],
    "messaging":    ["kafka", "publish", "event", "retry"],
}

PIPELINE_STAGES = ["parse", "enrich", "classify"]
N_LOGS = len(RAW_LOGS)


def _process_stage(stage: str, data: dict) -> dict:
    """Each stage transforms the item; 10 ms simulated latency per stage."""
    time.sleep(0.01)

    if stage == "parse":
        parts = data["raw"].split(maxsplit=2)
        return {
            "ip":      parts[0],
            "level":   parts[1],
            "message": parts[2] if len(parts) > 2 else "",
        }

    if stage == "enrich":
        return {
            **data,
            "service":  _IP_TO_SERVICE.get(data["ip"], "unknown"),
            "severity": _LEVEL_SEVERITY.get(data["level"], 0),
        }

    if stage == "classify":
        msg = data["message"].lower()
        category = "other"
        for cat, keywords in _CATEGORIES.items():
            if any(kw in msg for kw in keywords):
                category = cat
                break
        action = (
            "page-oncall"   if data["severity"] == 3 else
            "create-ticket" if data["severity"] == 2 else
            "log-only"
        )
        return {**data, "category": category, "action": action}

    return data


# ── Flow ──────────────────────────────────────────────────────────────────────

class StagedPipelineFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=3, type=int,
                          help="Parallel pipeline workers")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ──────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        A single asyncio.Queue holds all in-flight work across all stages.

        When a worker completes stage S, /complete puts the item back in the
        queue tagged with stage S+1.  Workers only call /next and /complete —
        they never know what stage comes next.  The coordinator is the sole
        owner of pipeline topology.
        """
        import asyncio
        from fastapi import FastAPI

        n_workers = self.n_workers_int
        results   = {i: {} for i in range(N_LOGS)}
        completed: set[int] = set()
        lock      = threading.Lock()

        work_done  = threading.Event()
        done_event = threading.Event()

        _loop:  list[asyncio.AbstractEventLoop] = []
        _queue: list[asyncio.Queue]             = []

        def _drain():
            work_done.wait()
            # One poison-pill per worker so every blocked /next returns cleanly.
            for _ in range(n_workers):
                _loop[0].call_soon_threadsafe(_queue[0].put_nowait, None)
            done_event.set()

        threading.Thread(target=_drain, daemon=True).start()

        app = FastAPI(title="staged-pipeline")

        @app.on_event("startup")
        async def _startup():
            _loop.append(asyncio.get_running_loop())
            q = asyncio.Queue()
            for item_id, raw in enumerate(RAW_LOGS):
                await q.put({"item_id": item_id, "stage": "parse",
                             "data": {"raw": raw}})
            _queue.append(q)

        @app.post("/next")
        async def next_item(body: dict):
            """Block until the next (item, stage) pair is available."""
            item = await _queue[0].get()
            if item is None:
                return {"done": True}   # consume pill — do not re-enqueue
            return item

        @app.post("/complete")
        async def complete(body: dict):
            item_id = body["item_id"]
            stage   = body["stage"]
            result  = body["result"]

            with lock:
                results[item_id][stage] = result

            stage_idx = PIPELINE_STAGES.index(stage)
            if stage_idx < len(PIPELINE_STAGES) - 1:
                next_stage = PIPELINE_STAGES[stage_idx + 1]
                await _queue[0].put({
                    "item_id": item_id,
                    "stage":   next_stage,
                    "data":    result,
                })
            else:
                with lock:
                    completed.add(item_id)
                    if len(completed) >= N_LOGS:
                        work_done.set()

            return {"ok": True}

        @app.get("/health")
        async def health():
            return {"queue_size": _queue[0].qsize() if _queue else 0,
                    "completed":  len(completed), "total": N_LOGS}

        svc = FastAPIService(app=app, done=done_event, drain_delay=2.0)
        svc.run(service_id=self.coordinator_id, namespace="staged-pipeline")

        self.pipeline_results = results
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.svc_url      = await_service(self.coordinator_id,
                                          namespace="staged-pipeline", timeout=120)
        self.worker_range = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_range")

    @step
    def run_worker(self):
        import httpx

        processed_stages = 0

        while True:
            try:
                resp = httpx.post(
                    f"{self.svc_url}/next",
                    json={"worker_id": self.input},
                    timeout=60,
                )
                resp.raise_for_status()
                item = resp.json()
            except Exception:
                break

            if item.get("done"):
                break

            result = _process_stage(item["stage"], item["data"])

            try:
                httpx.post(
                    f"{self.svc_url}/complete",
                    json={
                        "item_id": item["item_id"],
                        "stage":   item["stage"],
                        "result":  result,
                    },
                    timeout=10,
                ).raise_for_status()
            except Exception:
                break

            processed_stages += 1

        self.processed_stages = processed_stages
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.total_stages = sum(inp.processed_stages for inp in inputs)
        self.next(self.join)

    # ── Final join ────────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        from collections import Counter

        n_complete = sum(
            1 for stages in self.pipeline_results.values()
            if len(stages) == len(PIPELINE_STAGES)
        )
        expected = N_LOGS * len(PIPELINE_STAGES)

        print(f"\nStaged Pipeline — {self.n_workers_int} workers, "
              f"{len(PIPELINE_STAGES)} stages, {N_LOGS} log records")
        print(f"  Records fully processed: {n_complete}/{N_LOGS}")
        print(f"  Total stage executions:  {self.total_stages}  "
              f"(expected {expected} = {N_LOGS} × {len(PIPELINE_STAGES)})")

        final = [
            self.pipeline_results[i]["classify"]
            for i in range(N_LOGS)
            if "classify" in self.pipeline_results[i]
        ]
        actions    = Counter(r["action"]   for r in final)
        categories = Counter(r["category"] for r in final)
        services   = Counter(r["service"]  for r in final)

        print(f"\n  Recommended actions:")
        for action, count in actions.most_common():
            print(f"    {'█' * count} {count}  {action}")

        print(f"\n  Event categories:")
        for cat, count in categories.most_common():
            print(f"    {'█' * count} {count}  {cat}")

        print(f"\n  Events by service:")
        for svc, count in services.most_common():
            print(f"    {'█' * count} {count}  {svc}")

        print(f"\n  Each record passed through: "
              f"{' → '.join(PIPELINE_STAGES)}")
        print(f"  Workers handled all stages; the coordinator owned the routing.")


if __name__ == "__main__":
    StagedPipelineFlow()
