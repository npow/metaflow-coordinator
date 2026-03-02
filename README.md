# metaflow-session-service

> Ephemeral coordination services for parallel Metaflow steps — no Redis, no databases, no open ports.

When you fan out to N parallel worker steps in Metaflow, those workers share nothing.
This library lets you spin up a lightweight HTTP service **inside the coordinator step**,
have all workers discover and call it, then tear it down automatically when work is done.

```
coordinator step  ←──── registers URL
    worker 0      ────► calls URL  (work queue item, semaphore, barrier, …)
    worker 1      ────► calls URL
    worker N      ────► calls URL
coordinator step  ←──── receives results, service tears down
```

Everything is scoped to one run. Locally the URL is shared via `/tmp`; on AWS Batch or
Kubernetes it goes through S3. No infrastructure to provision. No services to maintain.

---

## Is this for me?

**Yes** if you write Metaflow flows with `foreach` parallelism and need workers to:
- pull tasks from a shared queue rather than having tasks pre-assigned
- throttle calls to a rate-limited API (at most N concurrent across all workers)
- synchronize at a barrier between rounds (e.g. federated learning)
- write to a shared cache to avoid redundant computation
- report results to a single aggregator step

**No** if your workers are fully independent (embarrassingly parallel with no shared state)
— you don't need this library, plain Metaflow `foreach` is sufficient.

---

## Install

```bash
pip install metaflow-session-service
# S3 rendezvous for remote execution (AWS Batch / Kubernetes):
pip install "metaflow-session-service[s3]"
```

---

## Minimal example

```python
from metaflow import FlowSpec, current, step
from metaflow_session_service import CompletionTracker, FastAPIService, await_service

class MyFlow(FlowSpec):
    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.worker_ids = list(range(4))
        self.next(self.run_coordinator, self.launch_workers)

    @step
    def run_coordinator(self):
        from fastapi import FastAPI
        results = {}
        tracker = CompletionTracker(n_workers=4)
        app = FastAPI()

        @app.post("/submit")
        async def submit(body: dict):
            results[body["worker_id"]] = body["value"]
            return {"ok": True}

        FastAPIService(app=app, done=tracker.done).run(
            service_id=self.coordinator_id, namespace="my-flow"
        )
        self.results = results
        self.next(self.join)

    @step
    def launch_workers(self):
        self.url = await_service(self.coordinator_id, namespace="my-flow", timeout=60)
        self.next(self.run_worker, foreach="worker_ids")

    @step
    def run_worker(self):
        import httpx
        httpx.post(f"{self.url}/submit",   json={"worker_id": self.input, "value": self.input * 2})
        httpx.post(f"{self.url}/complete")
        self.next(self.join_workers)

    @step
    def join_workers(self, inputs):
        self.merge_artifacts(inputs, include=["coordinator_id"])
        self.next(self.join)

    @step
    def join(self, inputs):
        for inp in inputs:
            if hasattr(inp, "results"):
                self.results = inp.results
        self.next(self.end)

    @step
    def end(self):
        print(self.results)

if __name__ == "__main__":
    MyFlow()
```

The coordinator step blocks inside `FastAPIService.run()` until `tracker.done` fires
(when all 4 workers call `POST /complete`). No threads or event loops to manage.

---

## Coordination patterns

Ten patterns cover the full landscape of worker↔coordinator interactions.
Each has a runnable example in `examples/`.

| # | Pattern | What the coordinator does | Example |
|---|---------|--------------------------|---------|
| 1 | **Work Queue** | Holds a queue; workers pull the next item | `work_queue.py` |
| 2 | **Broadcast + Consensus** | Fans out same input; aggregates N independent results | `agent_ensemble.py` |
| 3 | **Shared Mutable State** | Key-value cache shared across workers | `shared_cache.py` |
| 4 | **Synchronization Barrier** | Holds all workers at a round boundary until all arrive | `gradient_aggregator.py` |
| 5 | **Adaptive Search** | Eliminates low-performing candidates between rounds | `tournament.py` |
| 6 | **Rate Limiter** | `asyncio.Semaphore` — at most N concurrent API calls | `rate_limiter.py` |
| 7 | **External Process** | Wraps Redis, nginx, or any binary as a session service | `shard_server.py` |
| 8 | **Priority Queue** | Max-heap dispatch; workers can re-prioritise pending items | `priority_queue.py` |
| 9 | **Staged Pipeline** | Routes items through ordered stages; workers handle any stage | `staged_pipeline.py` |
| 10 | **Nested Coordinators** | Tree topology: group coordinators each manage sub-workers | `nested_coordinators.py` |

<details>
<summary>Pattern descriptions</summary>

### Work Queue
Workers call `POST /next` to claim the next task.  Work distributes itself to whichever
worker is free — no pre-assignment needed.

### Broadcast + Consensus
Every worker receives the same input and produces an independent result.  The coordinator
reduces them: majority vote, averaging, best-of-N.

### Shared Mutable State
Coordinator owns a key-value store.  Workers read and write it to avoid redundant
computation on overlapping sub-problems.

### Synchronization Barrier
Coordinator blocks all workers at round R until every worker has arrived, then releases
them simultaneously to round R+1.  Built-in: `BarrierService(n_workers, n_rounds)`.

### Adaptive Search
After each evaluation round, the coordinator keeps only the top-performing candidates and
eliminates the rest — successive halving / tournament bracket.

### Rate Limiter
Coordinator holds an `asyncio.Semaphore(max_concurrent)`.  Workers call `POST /acquire`
before hitting a rate-limited API and `POST /release` after.  Enforces the limit across
the entire worker fleet regardless of how many tasks run in parallel.

### External Process
`ProcessService` launches any binary with `{port}` substituted at runtime, polls a
readiness check, registers its URL, and terminates it on shutdown.  Swap
`python -m http.server` for `redis-server`, `nginx`, `duckdb`, or any daemon.

### Priority Queue
Coordinator maintains a max-priority heap backed by an `asyncio.Condition`.  Workers
receive the highest-priority pending item.  A worker can call `POST /boost` to raise
the priority of related items it discovers mid-run — without touching in-flight items.

### Staged Pipeline
A single queue holds work regardless of stage.  When a worker completes stage S, the
coordinator advances the item to stage S+1.  Workers are homogeneous and stateless;
the coordinator owns the pipeline topology.

### Nested Coordinators
Top coordinator manages group coordinators; each group coordinator manages its own
workers.  Service IDs are hierarchical: `{run_id}` (top), `{run_id}/group-{id}` (group).
Implemented as nested Metaflow splits — a `foreach` branch that itself contains a
sub-coordinator step and a nested `foreach` of workers.

</details>

---

## Core API

### Service types

```python
from metaflow_session_service import (
    FastAPIService,     # in-process FastAPI + uvicorn
    ProcessService,     # external subprocess (Redis, nginx, …)
    SessionServiceGroup # run multiple services concurrently
)

# In-process FastAPI service
svc = FastAPIService(app=my_fastapi_app, done=done_event)
svc.run(service_id=self.coordinator_id, namespace="my-flow")

# External process — {port} is substituted at runtime
svc = ProcessService(
    command=["redis-server", "--port", "{port}"],
    done=tracker.done,
    url_scheme="redis",
    ready=HttpReady(path="/health"),   # or SocketReady() (default)
)

# Multiple services in one coordinator step
group = SessionServiceGroup({"redis": redis_svc, "tracker": tracker})
group.run(service_id=self.coordinator_id, namespace="my-flow")
```

### Built-in coordination services

```python
from metaflow_session_service import CompletionTracker, ResultCollector, BarrierService

# Fires done when N workers call POST /complete
tracker = CompletionTracker(n_workers=20)

# Fires done when N workers call POST /submit; stores their results
collector = ResultCollector(n_workers=20)
# ... after run() returns:
print(collector.results)            # list in submission order
print(collector.results_by_worker)  # dict keyed by worker_id

# Synchronization barrier across R rounds
barrier = BarrierService(n_workers=20, n_rounds=5)
```

### Service discovery

```python
from metaflow_session_service import await_service, discover_services

# Single service
url = await_service(self.coordinator_id, namespace="my-flow", timeout=120)

# Multiple services from a SessionServiceGroup (parallel discovery)
urls = discover_services(
    self.coordinator_id,
    roles=["redis", "tracker"],
    namespace="my-flow",
    timeout=120,
)
redis_url   = urls["redis"]
tracker_url = urls["tracker"]
```

### Join decorators (boilerplate reduction)

```python
from metaflow_session_service import coordinator_join, worker_join

# Replaces manual merge_artifacts in the final coordinator+workers join
@step
@coordinator_join
def join(self, inputs):
    self.next(self.end)

# Replaces manual merge_artifacts in foreach-reduce steps
@step
@worker_join
def join_workers(self, inputs):
    self.results = [inp.result for inp in inputs]   # custom aggregation only
    self.next(self.join)
```

### Checkpointing

```python
from metaflow_session_service import save_checkpoint, load_checkpoint

# Save state periodically so a restarted coordinator can resume
save_checkpoint(namespace="my-flow", service_id=run_id, data={"completed": n})

# On restart
state = load_checkpoint(namespace="my-flow", service_id=run_id)
if state:
    n_completed = state["completed"]
```

---

## Remote execution

Add `@batch` or `@kubernetes` to coordinator and worker steps.  Service URLs are
exchanged via S3 automatically — no VPC changes or security-group rules needed as long
as coordinator and workers share a VPC and the coordinator's port is reachable.

For external processes that need native binaries, use `@conda` (installs binaries) in
addition to `@pypi` (installs Python clients):

```python
@conda(packages={"redis": "7.2"})   # redis-server binary
@pypi(packages={"redis": "5.0"})    # Python client
@step
def run_coordinator(self):
    tracker  = CompletionTracker(n_workers=self.n_workers)
    redis    = ProcessService(["redis-server", "--port", "{port}"],
                              done=tracker.done, url_scheme="redis")
    SessionServiceGroup({"redis": redis, "tracker": tracker}).run(
        service_id=self.coordinator_id, namespace="my-flow"
    )
```
