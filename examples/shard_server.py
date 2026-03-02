"""
Data Shard Server — serve pre-computed data to workers via ProcessService.

In production ML pipelines, the coordinator often holds reference data that
every worker needs: a vocabulary file, a pre-built index, a lookup table of
entity embeddings.  Loading this data separately into each worker is wasteful
(N × memory, N × IO).  Instead, serve it once from the coordinator step
using a lightweight HTTP file server.

This example uses Python's built-in ``python -m http.server`` wrapped in a
ProcessService — no extra binaries required.  Workers download their assigned
shard over plain HTTP using httpx.

The pattern generalises to any binary that serves data:

    nginx (production file serving, TLS):
        ProcessService(["nginx", "-c", "/tmp/nginx.conf"], url_scheme="http", ...)

    Redis (key-value lookups, deduplication sets):
        ProcessService(["redis-server", "--port", "{port}"], url_scheme="redis", ...)
        # pip install redis;  conda install redis (for the redis-server binary)

    DuckDB HTTP extension (SQL analytics over shared tables):
        ProcessService(["duckdb", "-cmd", ".open :memory:", ...], url_scheme="http", ...)

ProcessService handles port discovery, readiness polling, and clean shutdown
automatically — you only specify the command, the done event, and the URL scheme.

Architecture:
   start ─── run_coordinator (file server + result collector)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Services in the coordinator (SessionServiceGroup):
   "files"     → ProcessService  (python -m http.server — serves JSON shards)
   "collector" → FastAPIService  (POST /result — aggregates worker results)
   "tracker"   → CompletionTracker (POST /complete — shuts everything down)

Run:
    python examples/shard_server.py run --n_workers 4 --n_shards 8
"""
import hashlib
import json
import os
import tempfile
import threading

from metaflow import FlowSpec, Parameter, current, step

from metaflow_session_service import (
    CompletionTracker,
    FastAPIService,
    HttpReady,
    HTTPX_ERRORS,
    ProcessService,
    SessionServiceGroup,
    discover_services,
    coordinator_join,
    worker_join,
)

# ── Data generation ───────────────────────────────────────────────────────────

# A small fixed vocabulary so the example is self-contained.
_VOCAB_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "machine", "learning", "neural", "network", "gradient", "descent",
    "python", "data", "model", "train", "feature", "vector", "embedding",
    "cluster", "batch", "epoch", "loss", "accuracy", "precision", "recall",
    "token", "sentence", "document", "query", "index", "search", "rank",
]

# Build a word → id vocabulary (this is what workers will download)
VOCABULARY = {word: idx for idx, word in enumerate(_VOCAB_WORDS)}

# Sample sentences for workers to tokenize
_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning model train on data features",
    "neural network gradient descent batch epoch loss",
    "document search and query index rank results",
    "python vector embedding cluster precision recall",
    "the brown fox and the lazy dog over train",
    "feature vector machine learning model accuracy",
    "token sentence document query search index rank",
    "gradient descent neural network epoch loss train",
    "embedding cluster batch precision recall vector",
    "the quick fox jumps over and the dog runs",
    "data model feature learning precision accuracy rank",
]


def _make_shard(shard_id: int, n_sentences: int = 10) -> dict:
    """Return a shard dict: list of sentences assigned to this shard."""
    rng_seed = hashlib.md5(f"shard-{shard_id}".encode()).digest()
    n = len(_SENTENCES)
    # Deterministic round-robin with offset so shards differ
    sentences = [_SENTENCES[(shard_id * 3 + i) % n] for i in range(n_sentences)]
    return {"shard_id": shard_id, "sentences": sentences}


# ── Flow ──────────────────────────────────────────────────────────────────────

class ShardServerFlow(FlowSpec):
    n_workers  = Parameter("n_workers",  default=4, type=int, help="Parallel workers")
    n_shards   = Parameter("n_shards",   default=8, type=int, help="Number of data shards")
    sentences_per_shard = Parameter("sentences_per_shard", default=10, type=int,
                                    help="Sentences per shard")

    @step
    def start(self):
        self.coordinator_id       = current.run_id
        self.n_workers_int        = int(self.n_workers)
        self.n_shards_int         = int(self.n_shards)
        self.sentences_per_shard_int = int(self.sentences_per_shard)

        # Assign shards round-robin to workers
        self.worker_shards = [
            [s for s in range(self.n_shards_int) if s % self.n_workers_int == w]
            for w in range(self.n_workers_int)
        ]
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ───────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        1. Write vocabulary.json + shard-NNNN.json to a temp directory.
        2. Start python -m http.server (ProcessService) to serve them.
        3. Start a FastAPIService result collector for worker results.
        4. Start a CompletionTracker that shuts everything down when all
           workers call POST /complete.

        The three services share a SessionServiceGroup so they all start
        and stop together cleanly.
        """
        from fastapi import FastAPI

        n_workers = self.n_workers_int
        n_shards  = self.n_shards_int

        # ── Write data files ─────────────────────────────────────────────────
        shard_dir = tempfile.mkdtemp(prefix=f"mf-shards-{self.coordinator_id}-")

        with open(os.path.join(shard_dir, "vocabulary.json"), "w") as f:
            json.dump(VOCABULARY, f)

        for shard_id in range(n_shards):
            shard = _make_shard(shard_id, self.sentences_per_shard_int)
            fname = f"shard-{shard_id:04d}.json"
            with open(os.path.join(shard_dir, fname), "w") as f:
                json.dump(shard, f)

        print(f"[coordinator] wrote vocabulary + {n_shards} shards to {shard_dir}")

        # ── Result collector (FastAPIService) ────────────────────────────────
        results = {}
        lock    = threading.Lock()

        tracker       = CompletionTracker(n_workers=n_workers)
        collector_app = FastAPI(title="shard-collector")

        @collector_app.post("/result")
        async def submit_result(body: dict):
            with lock:
                results[body["shard_id"]] = body["stats"]
            return {"ok": True, "received": len(results)}

        @collector_app.get("/health")
        async def health():
            return {"results": len(results), "total": n_shards}

        collector_svc = FastAPIService(app=collector_app, done=tracker.done)

        # ── File server (ProcessService) ─────────────────────────────────────
        # python -m http.server {port} --directory <dir>
        # Readiness: HttpReady polls GET / until it returns a directory listing
        file_svc = ProcessService(
            command=["python", "-m", "http.server", "{port}",
                     "--directory", shard_dir],
            done=tracker.done,
            url_scheme="http",
            ready=HttpReady(path="/"),
        )

        # ── Run all three services together ──────────────────────────────────
        group = SessionServiceGroup({
            "files":     file_svc,
            "collector": collector_svc,
            "tracker":   tracker,
        })
        group.run(service_id=self.coordinator_id, namespace="shard-server")

        self.shard_results = results
        self.shard_dir     = shard_dir
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        urls = discover_services(
            self.coordinator_id,
            roles=["files", "collector", "tracker"],
            namespace="shard-server",
            timeout=120,
        )
        self.files_url     = urls["files"]
        self.collector_url = urls["collector"]
        self.tracker_url   = urls["tracker"]
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """
        1. Download vocabulary.json from the file server (once).
        2. For each assigned shard: download shard JSON, tokenize sentences,
           compute per-word frequency stats.
        3. POST stats to the result collector.
        4. Signal completion to the tracker.
        """
        import httpx
        from collections import Counter

        # Download the shared vocabulary (same for all workers; one HTTP call)
        vocab = httpx.get(f"{self.files_url}/vocabulary.json", timeout=30).json()

        my_shards    = self.worker_shards[self.input]
        total_tokens = 0
        total_oov    = 0   # out-of-vocabulary tokens
        word_counts  = Counter()

        for shard_id in my_shards:
            fname = f"shard-{shard_id:04d}.json"
            try:
                shard = httpx.get(f"{self.files_url}/{fname}", timeout=30).json()
            except HTTPX_ERRORS:
                continue

            for sentence in shard["sentences"]:
                for token in sentence.split():
                    word_counts[token] += 1
                    if token in vocab:
                        total_tokens += 1
                    else:
                        total_oov += 1

            stats = {
                "shard_id":     shard_id,
                "n_sentences":  len(shard["sentences"]),
                "tokens":       total_tokens,
                "oov":          total_oov,
                "top_words":    word_counts.most_common(3),
            }
            try:
                httpx.post(
                    f"{self.collector_url}/result",
                    json={"shard_id": shard_id, "stats": stats},
                    timeout=10,
                ).raise_for_status()
            except HTTPX_ERRORS:
                continue

        # Signal to the tracker that this worker is done
        try:
            httpx.post(f"{self.tracker_url}/complete", timeout=10).raise_for_status()
        except HTTPX_ERRORS:
            pass

        self.shards_done  = len(my_shards)
        self.total_tokens = total_tokens
        self.total_oov    = total_oov
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.worker_token_counts = [inp.total_tokens for inp in inputs]
        self.worker_oov_counts   = [inp.total_oov   for inp in inputs]
        self.worker_shard_counts = [inp.shards_done  for inp in inputs]
        self.next(self.join)

    # ── Final join ────────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        total_tokens = sum(self.worker_token_counts)
        total_oov    = sum(self.worker_oov_counts)
        total_shards = len(self.shard_results)
        vocab_size   = len(VOCABULARY)

        print(f"\nShard Server Pipeline — {self.n_workers_int} workers, "
              f"{self.n_shards_int} shards, vocabulary size {vocab_size}")
        print(f"  Shards processed:   {total_shards}/{self.n_shards_int}")
        print(f"  Total tokens:       {total_tokens}")
        print(f"  OOV tokens:         {total_oov}  "
              f"({total_oov / max(total_tokens + total_oov, 1):.1%})")

        print("\n  Worker breakdown (shards / in-vocabulary tokens):")
        for i, (shards, tok) in enumerate(
                zip(self.worker_shard_counts, self.worker_token_counts)):
            bar = "█" * shards
            print(f"    Worker {i}: {bar} {shards} shards, {tok} tokens")

        print(f"\n  Files were served from:  {self.shard_dir}")
        print("  The coordinator ran python -m http.server as a ProcessService.")
        print("  In production swap it for: nginx, Redis, DuckDB, or any binary.")


if __name__ == "__main__":
    ShardServerFlow()
