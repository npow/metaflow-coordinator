"""
Work Queue — pull-based ETL pipeline for log records.

Metaflow's `foreach` assigns exactly one batch to each worker up front.
With a pull queue, workers grab records as fast as they finish them — slow
workers don't block fast ones, and no worker sits idle at the end.

Real-world scenario: nightly ETL job parsing Nginx / Apache access logs.
Replace the generate_log_lines() function with a real S3 listing, database
query, or Kafka consumer to adapt this to your own pipeline.

Architecture:
   start ─── run_coordinator (hosts the queue, blocks until all records done)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Run:
    python examples/work_queue.py run --n_workers 4 --n_records 40
"""
import re
from collections import defaultdict

from metaflow import FlowSpec, Parameter, current, step

from metaflow_coordinator import WorkQueue, await_service, coordinator_join, worker_join, HTTPX_ERRORS

# ── Log generator (simulates records coming from S3 / a DB) ───────────────

_LOG_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
_LOG_PATHS   = [
    "/api/v1/users", "/api/v1/orders", "/api/v1/products",
    "/api/v2/events", "/health", "/login", "/logout",
    "/static/app.js", "/static/app.css",
]
_LOG_STATUSES = [200, 200, 200, 200, 301, 304, 400, 404, 404, 500]

def generate_log_lines(n: int):
    import random
    rng = random.Random(0)
    records = []
    for i in range(n):
        ip     = f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
        method = rng.choice(_LOG_METHODS)
        path   = rng.choice(_LOG_PATHS)
        status = rng.choice(_LOG_STATUSES)
        size   = rng.randint(200, 65536) if status == 200 else rng.randint(50, 512)
        ms     = rng.randint(5, 2000)    # response time in ms
        records.append({
            "id":   i,
            "line": f'{ip} - - [01/Mar/2026:12:00:{i % 60:02d} +0000] '
                    f'"{method} {path} HTTP/1.1" {status} {size} {ms}ms',
        })
    return records

# ── Log parser ─────────────────────────────────────────────────────────────

_LOG_PATTERN = re.compile(
    r'(\S+) \S+ \S+ \[(.+?)\] "(\w+) (\S+) \S+" (\d+) (\d+) (\d+)ms'
)

def parse_log_line(line: str) -> dict | None:
    m = _LOG_PATTERN.match(line)
    if not m:
        return None
    ip, ts, method, path, status, size, ms = m.groups()
    endpoint = path.rsplit("/", 1)[0] or path   # strip trailing ID
    return {
        "ip":       ip,
        "ts":       ts,
        "method":   method,
        "endpoint": endpoint,
        "status":   int(status),
        "size_kb":  round(int(size) / 1024, 2),
        "latency_ms": int(ms),
        "error":    int(status) >= 400,
    }


class WorkQueueFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=4,  type=int, help="Parallel workers")
    n_records = Parameter("n_records", default=40, type=int, help="Log records to process")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.n_records_int  = int(self.n_records)
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """Serve a queue of log records; shut down when all are parsed and submitted."""
        records = generate_log_lines(self.n_records_int)
        wq = WorkQueue(items=records, drain_delay=2.0)
        wq.run(service_id=self.coordinator_id, namespace="log-etl")

        # Aggregate stats across all parsed records
        results = wq.results_by_item         # {item_id: parsed_record | None}
        parsed  = [r for r in results.values() if r is not None]
        by_status   = defaultdict(int)
        by_endpoint = defaultdict(int)
        total_latency = 0
        for r in parsed:
            by_status[r["status"]] += 1
            by_endpoint[r["endpoint"]] += 1
            total_latency += r["latency_ms"]

        self.etl_stats = {
            "total":          self.n_records_int,
            "parsed_ok":      len(parsed),
            "parse_errors":   self.n_records_int - len(parsed),
            "avg_latency_ms": round(total_latency / len(parsed), 1) if parsed else 0,
            "status_counts":  dict(by_status),
            "top_endpoints":  dict(sorted(by_endpoint.items(), key=lambda kv: -kv[1])[:5]),
            "error_rate":     round(sum(r["error"] for r in parsed) / len(parsed), 3) if parsed else 0,
        }
        self.next(self.join)

    # ── Workers ────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.queue_url      = await_service(self.coordinator_id, namespace="log-etl", timeout=120)
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """Pull log lines from the queue, parse them, and submit parsed records."""
        import httpx

        records_processed = 0
        while True:
            try:
                item = httpx.post(f"{self.queue_url}/pull", json={}, timeout=30).json()
            except HTTPX_ERRORS:
                break   # coordinator drained and shut down

            if item.get("done"):
                break

            parsed = parse_log_line(item["payload"]["line"])
            try:
                httpx.post(
                    f"{self.queue_url}/submit",
                    json={"item_id": item["item_id"], "result": parsed},
                    timeout=30,
                ).raise_for_status()
            except HTTPX_ERRORS:
                break

            records_processed += 1

        self.records_processed = records_processed
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.records_per_worker = {inp.input: inp.records_processed for inp in inputs}
        self.next(self.join)

    # ── Final join ─────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        s = self.etl_stats
        print(f"\nETL Pipeline — {s['total']} log records, {self.n_workers_int} workers")
        print(f"  Parsed OK:      {s['parsed_ok']}")
        print(f"  Parse errors:   {s['parse_errors']}")
        print(f"  Avg latency:    {s['avg_latency_ms']} ms")
        print(f"  Error rate:     {s['error_rate']:.1%}")

        print("\n  HTTP status distribution:")
        for status, count in sorted(s["status_counts"].items()):
            bar = "█" * (count * 20 // s["total"])
            print(f"    {status}: {bar} {count}")

        print("\n  Top endpoints by hit count:")
        for ep, count in s["top_endpoints"].items():
            print(f"    {ep:<35s} {count}")

        print("\n  Records processed per worker:")
        for worker, count in sorted(self.records_per_worker.items()):
            bar = "█" * count
            print(f"    Worker {worker}: {bar} ({count})")


if __name__ == "__main__":
    WorkQueueFlow()
