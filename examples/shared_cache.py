"""
Shared Feature Store — entity embedding hydration with a shared cache.

In ML pipelines, parallel workers often need the same feature vectors:
user embeddings, product embeddings, entity metadata.  Without a shared
cache, the same entity is fetched N times from a slow source (database,
embedding API, feature store).  With a shared cache, each entity is
fetched exactly once, no matter how many workers need it.

This flow uses SessionServiceGroup to run two services together:
  • "store"   → FastAPIService  (GET/PUT entity feature vectors)
  • "tracker" → CompletionTracker (shuts everything down when workers finish)

Realistic scenario: recommendation pipeline hydrating user and item
embeddings from a feature store before running batch inference.

Swapping to Redis (no code changes on the worker side needed):
    Replace the FastAPIService store with a ProcessService backed by redis-server.
    See comments in run_coordinator() below.

Run:
    python examples/shared_cache.py run --n_workers 6 --n_entities 80 --overlap 20
"""
import hashlib
import threading
import time

from metaflow import FlowSpec, Parameter, current, step

from metaflow_session_service import (
    CompletionTracker,
    FastAPIService,
    SessionServiceGroup,
    discover_services,
    coordinator_join,
    worker_join,
)

EMBEDDING_DIM = 32   # dimensionality of our mock feature vectors


def _fake_embedding(entity_id: str) -> list[float]:
    """
    Mock feature vector for an entity.

    In production, call your embedding API / feature store here:
        resp = requests.get(f"https://feature-store/v1/entity/{entity_id}")
        return resp.json()["embedding"]
    """
    h = hashlib.sha256(entity_id.encode()).digest()
    # Deterministic float vector in [-1, 1] from the hash bytes
    return [(b / 127.5) - 1.0 for b in h[:EMBEDDING_DIM]]


def _fetch_latency_ms() -> float:
    """Simulate a slow upstream feature store call (30 ms)."""
    time.sleep(0.03)
    return 30.0


class SharedFeatureStoreFlow(FlowSpec):
    n_workers  = Parameter("n_workers",  default=6,  type=int, help="Parallel workers")
    n_entities = Parameter("n_entities", default=80, type=int, help="Total unique entity IDs")
    overlap    = Parameter("overlap",    default=20, type=int,
                           help="Entities appearing in multiple workers' batches")

    @step
    def start(self):
        import random
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.n_entities_int = int(self.n_entities)
        self.overlap_int    = int(self.overlap)

        # Assign batches: each worker gets some unique entities + shared "hot" entities.
        # Hot entities simulate popular items (e.g., bestseller products, power users).
        rng          = random.Random(42)
        hot_entities = [f"entity:hot:{k}" for k in range(self.overlap_int)]
        n_unique     = self.n_entities_int - self.overlap_int
        unique_per   = max(1, n_unique // self.n_workers_int)

        self.worker_batches = []
        for i in range(self.n_workers_int):
            unique = [f"entity:worker{i}:{k}" for k in range(unique_per)]
            batch  = hot_entities + unique
            rng.shuffle(batch)
            self.worker_batches.append(batch)

        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        Run the feature store + completion tracker as a SessionServiceGroup.

        When all workers call POST /{tracker_url}/complete, both services
        shut down and this step returns.

        ── Redis swap-in ──────────────────────────────────────────────────
        To back the store with a real Redis server instead of in-memory Python:

            from metaflow_session_service import ProcessService

            cache_svc = ProcessService(
                command=["redis-server", "--port", "{port}"],
                done=tracker.done,
                url_scheme="redis",
            )
            group = SessionServiceGroup({"store": cache_svc, "tracker": tracker})

        Workers would then use redis.Redis.from_url(store_url) instead of httpx.
        No other changes needed.  Install: conda install redis=7.2 (for the binary)
        and pip install redis (for the Python client).
        ───────────────────────────────────────────────────────────────────
        """
        from fastapi import FastAPI

        store: dict[str, list] = {}
        stats = {"hits": 0, "misses": 0, "total_latency_saved_ms": 0.0}
        lock  = threading.Lock()

        store_app = FastAPI(title="feature-store")

        @store_app.get("/entity/{entity_id}")
        async def get_entity(entity_id: str):
            with lock:
                if entity_id in store:
                    stats["hits"] += 1
                    stats["total_latency_saved_ms"] += 30.0
                    return {"hit": True, "embedding": store[entity_id]}
                stats["misses"] += 1
                return {"hit": False, "embedding": None}

        @store_app.put("/entity/{entity_id}")
        async def put_entity(entity_id: str, body: dict):
            with lock:
                store[entity_id] = body["embedding"]
            return {"ok": True}

        @store_app.get("/stats")
        async def get_stats():
            with lock:
                return dict(stats)

        tracker   = CompletionTracker(n_workers=self.n_workers_int)
        store_svc = FastAPIService(app=store_app, done=tracker.done)

        group = SessionServiceGroup({"store": store_svc, "tracker": tracker})
        group.run(service_id=self.coordinator_id, namespace="feature-store")

        total = stats["hits"] + stats["misses"]
        self.cache_stats = {
            "cache_hits":   stats["hits"],
            "cache_misses": stats["misses"],
            "hit_rate":     round(stats["hits"] / total, 3) if total else 0,
            "entities_cached": len(store),
            "latency_saved_ms": round(stats["total_latency_saved_ms"], 1),
        }
        self.next(self.join)

    # ── Workers ────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        urls = discover_services(
            self.coordinator_id,
            roles=["store", "tracker"],
            namespace="feature-store",
            timeout=120,
        )
        self.store_url   = urls["store"]
        self.tracker_url = urls["tracker"]
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """
        Hydrate entity embeddings for my batch.
        Check the shared cache first; compute and store on miss.
        """
        import httpx

        my_entities    = self.worker_batches[self.input]
        embeddings     = {}
        local_hits     = 0
        local_misses   = 0

        for entity_id in my_entities:
            # Cache read
            resp = httpx.get(f"{self.store_url}/entity/{entity_id}", timeout=10).json()

            if resp["hit"]:
                embeddings[entity_id] = resp["embedding"]
                local_hits += 1
            else:
                # Cache miss: fetch from upstream (slow) and write back
                embedding = _fake_embedding(entity_id)
                _fetch_latency_ms()   # simulate 30ms upstream call
                httpx.put(
                    f"{self.store_url}/entity/{entity_id}",
                    json={"embedding": embedding},
                    timeout=10,
                ).raise_for_status()
                embeddings[entity_id] = embedding
                local_misses += 1

        # Signal completion to the tracker (this eventually shuts down the coordinator)
        httpx.post(f"{self.tracker_url}/complete", timeout=10).raise_for_status()

        self.embeddings_fetched  = len(embeddings)
        self.local_hits          = local_hits
        self.local_misses        = local_misses
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.worker_hit_counts   = [inp.local_hits   for inp in inputs]
        self.worker_miss_counts  = [inp.local_misses for inp in inputs]
        self.total_fetched       = sum(inp.embeddings_fetched for inp in inputs)
        self.next(self.join)

    # ── Final join ─────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        s = self.cache_stats
        total_calls = sum(self.worker_hit_counts) + sum(self.worker_miss_counts)

        print(f"\nFeature Store Hydration — {self.n_workers_int} workers, "
              f"{self.n_entities_int} entities ({self.overlap_int} shared / hot)")
        print(f"\n  Coordinator-side cache:")
        print(f"    Entities stored:    {s['entities_cached']}")
        print(f"    Read hits:          {s['cache_hits']}")
        print(f"    Read misses:        {s['cache_misses']}")
        print(f"    Hit rate:           {s['hit_rate']:.1%}")
        print(f"    Upstream calls saved: {s['cache_hits']}  (×30ms = {s['latency_saved_ms']} ms saved)")

        print(f"\n  Worker-side breakdown:")
        for i, (h, m) in enumerate(zip(self.worker_hit_counts, self.worker_miss_counts)):
            total = h + m
            pct   = h / total if total else 0
            print(f"    Worker {i}: {h:3d} hits / {m:3d} misses  ({pct:.0%} hit rate)")

        time_without = total_calls * 30
        time_with    = sum(self.worker_miss_counts) * 30
        print(f"\n  Without cache: ~{time_with + s['cache_hits'] * 30:,} ms total upstream work")
        print(f"  With cache:    ~{time_with:,} ms total upstream work")
        print(f"  Saved:          {s['latency_saved_ms']:,.0f} ms  ({s['hit_rate']:.0%} reduction)")


if __name__ == "__main__":
    SharedFeatureStoreFlow()
