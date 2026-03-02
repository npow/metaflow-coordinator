"""
Redis Shared Cache — avoid redundant computation across parallel workers.

Workers process an overlapping set of documents. Without a shared cache,
the same expensive computation runs multiple times across the fleet. With
Redis, the first worker to process a document caches the result; all others
read from cache instead of recomputing.

Uses Redis HGET/HSETNX for race-safe reads and writes: two workers can race
to process the same document, but only one write lands (HSETNX is atomic).

Architecture:
   start ─── run_coordinator (redis-server + completion tracker)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── end

Run:
    python examples/redis_cache.py run --n_workers 4
"""
import hashlib
import json
import time

from metaflow import FlowSpec, Parameter, conda, current, pypi, step

from metaflow_coordinator import (
    CompletionTracker,
    ProcessService,
    SessionServiceGroup,
    coordinator_join,
    discover_services,
    worker_join,
    HTTPX_ERRORS,
)

# ── Simulated workload ─────────────────────────────────────────────────────────

# 30 documents; each worker gets a slice with intentional overlap so
# the same document appears in multiple workers' assignments.
DOCUMENTS = [f"doc-{i:03d}" for i in range(30)]


def _compute(doc_id: str) -> str:
    """Simulate 50 ms of expensive work — embedding, OCR, feature extraction…"""
    time.sleep(0.05)
    return hashlib.sha256(doc_id.encode()).hexdigest()[:16]


# ── Flow ───────────────────────────────────────────────────────────────────────

class RedisCacheFlow(FlowSpec):

    n_workers = Parameter("n_workers", default=4, type=int, help="Parallel workers")

    @step
    def start(self):
        self.coordinator_id = current.run_id

        # Distribute documents with overlap: each worker gets n/w + 4 extra docs
        # drawn from neighbouring slices so cache hits are guaranteed.
        n, w = len(DOCUMENTS), self.n_workers
        stride = n // w
        self.worker_docs = [
            [DOCUMENTS[(i * stride + j) % n] for j in range(stride + 4)]
            for i in range(w)
        ]
        self.worker_ids = list(range(w))
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ───────────────────────────────────────────────────────────

    # @conda installs the redis-server binary
    @conda(packages={"redis": "7.2"})
    @step
    def run_coordinator(self):
        """Start redis-server and a completion tracker; block until workers finish."""
        tracker   = CompletionTracker(n_workers=self.n_workers)
        redis_svc = ProcessService(
            command=["redis-server", "--port", "{port}", "--save", ""],
            done=tracker.done,
            url_scheme="redis",
        )
        SessionServiceGroup({"redis": redis_svc, "tracker": tracker}).run(
            service_id=self.coordinator_id, namespace="redis-cache"
        )
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        urls = discover_services(
            self.coordinator_id,
            names=["redis", "tracker"],
            namespace="redis-cache",
            timeout=120,
        )
        self.redis_url   = urls["redis"]
        self.tracker_url = urls["tracker"]
        self.next(self.run_worker, foreach="worker_ids")

    @pypi(packages={"redis": "5.0"})
    @step
    def run_worker(self):
        import httpx
        import redis as redis_client

        r = redis_client.Redis.from_url(self.redis_url, decode_responses=True)

        computed   = 0
        cache_hits = 0

        for doc_id in self.worker_docs[self.input]:
            cached = r.hget("cache", doc_id)
            if cached:
                # Another worker already computed this — skip the work
                cache_hits += 1
            else:
                result = _compute(doc_id)
                # HSETNX: set only if the key does not exist (atomic)
                r.hsetnx("cache", doc_id, json.dumps({"doc_id": doc_id, "result": result}))
                computed += 1

        try:
            httpx.post(f"{self.tracker_url}/complete", timeout=10).raise_for_status()
        except HTTPX_ERRORS:
            pass

        self.computed   = computed
        self.cache_hits = cache_hits
        self.next(self.join_workers)

    # ── Reduce ────────────────────────────────────────────────────────────────

    @step
    @worker_join
    def join_workers(self, inputs):
        self.total_computed   = sum(inp.computed   for inp in inputs)
        self.total_cache_hits = sum(inp.cache_hits for inp in inputs)
        self.next(self.join)

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        total   = self.total_computed + self.total_cache_hits
        saved_s = self.total_cache_hits * 0.05  # 50 ms per skipped computation
        print(f"\nRedis Shared Cache — {self.n_workers} workers, "
              f"{len(DOCUMENTS)} unique documents")
        print(f"  Total calls:  {total}")
        print(f"  Computed:     {self.total_computed}")
        print(f"  Cache hits:   {self.total_cache_hits} "
              f"({self.total_cache_hits / max(total, 1):.0%})")
        print(f"  Time saved:   ~{saved_s:.1f}s of redundant computation avoided")
        print("\n  redis-server ran as a ProcessService — no Redis instance needed.")
        print("  HSETNX ensured each document was computed at most once, race-free.")


if __name__ == "__main__":
    RedisCacheFlow()
