"""
Postgres Results Store — write experiment results to a shared database.

Workers run hyperparameter search trials in parallel, writing metrics
directly to a shared Postgres table. The coordinator queries for the
top configurations at the end using SQL — no manual result aggregation
in Metaflow artifacts needed.

Unlike a FastAPIService result collector, Postgres gives you full SQL:
GROUP BY, window functions, percentiles, joins against reference tables, etc.

The coordinator runs postgres via ProcessService. After all workers call
POST /complete, a background thread queries the results table, then fires
the Postgres stop event so the service tears down cleanly.

Architecture:
   start ─── run_coordinator (postgres + completion tracker)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── end

Run:
    python examples/postgres_results.py run --n_workers 4 --n_trials 20
"""
import math
import random
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

from metaflow import FlowSpec, Parameter, conda, current, step

from metaflow_coordinator import (
    CompletionTracker,
    ProcessService,
    SessionServiceGroup,
    coordinator_join,
    discover_services,
    worker_join,
    HTTPX_ERRORS,
)


def _simulate_trial(lr: float, dropout: float) -> float:
    """Return a deterministic validation accuracy for a hyperparameter config."""
    import time
    time.sleep(0.05)
    return round(0.6 + 0.3 * math.exp(-abs(lr - 0.01) * 100) * (1 - dropout), 4)


# ── Flow ───────────────────────────────────────────────────────────────────────

class PostgresResultsFlow(FlowSpec):

    n_workers = Parameter("n_workers", default=4,  type=int, help="Parallel workers")
    n_trials  = Parameter("n_trials",  default=20, type=int, help="Total trials")
    top_k     = Parameter("top_k",     default=5,  type=int, help="Top-k results to show")

    @step
    def start(self):
        random.seed(42)
        self.coordinator_id = current.run_id

        # Generate hyperparameter grid
        trials = [
            {
                "trial_id": f"trial-{i:03d}",
                "lr":       round(random.choice([1e-4, 1e-3, 1e-2, 1e-1]), 4),
                "dropout":  round(random.uniform(0.1, 0.5), 2),
            }
            for i in range(self.n_trials)
        ]
        # Distribute round-robin across workers
        self.worker_trials = [trials[i::self.n_workers] for i in range(self.n_workers)]
        self.worker_ids    = list(range(self.n_workers))
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ───────────────────────────────────────────────────────────

    # @conda installs the postgres binary (initdb + postgres) and psycopg2
    @conda(libraries={"postgresql": ">=16.1", "psycopg2": ">=2.9.9"})
    @step
    def run_coordinator(self):
        import os
        import psycopg2

        coordinator_id = self.coordinator_id
        top_k          = self.top_k

        # Initialise a fresh Postgres data directory (ephemeral; no persistent state).
        # PostgreSQL refuses to run as root, so when root is detected we delegate
        # initdb / postgres to the 'daemon' system user via runuser(1).
        data_dir = tempfile.mkdtemp(prefix=f"mf-pg-{coordinator_id}-")
        is_root  = os.getuid() == 0
        if is_root:
            import grp, pwd
            daemon_uid = pwd.getpwnam("daemon").pw_uid
            daemon_gid = grp.getgrnam("daemon").gr_gid
            os.chown(data_dir, daemon_uid, daemon_gid)
            _wrap = ["runuser", "-u", "daemon", "--"]
        else:
            _wrap = []

        subprocess.run(
            _wrap + ["initdb", "-D", data_dir, "--auth", "trust", "--username", "postgres",
                     "--locale=C"],
            check=True, capture_output=True,
        )

        tracker = CompletionTracker(n_workers=self.n_workers)
        pg_stop = threading.Event()
        top_rows: list = []

        def _query_then_stop():
            """After workers finish, run SQL aggregation, then shut down Postgres."""
            tracker.done.wait()
            urls   = discover_services(
                coordinator_id, names=["pg"], namespace="pg-results", timeout=10,
            )
            parsed = urlparse(urls["pg"])
            conn   = psycopg2.connect(
                host=parsed.hostname, port=parsed.port,
                user="postgres", dbname="postgres",
            )
            cur = conn.cursor()
            cur.execute(
                """
                SELECT trial_id, lr, dropout, accuracy
                FROM   results
                ORDER  BY accuracy DESC
                LIMIT  %s
                """,
                (top_k,),
            )
            top_rows.extend(cur.fetchall())
            conn.close()
            pg_stop.set()

        threading.Thread(target=_query_then_stop, daemon=True).start()

        pg_svc = ProcessService(
            command=_wrap + [
                "postgres", "-D", data_dir, "-p", "{port}",
                "-h", "0.0.0.0",
                "-c", "log_min_messages=FATAL",
            ],
            done=pg_stop,
            url_scheme="postgresql",
            start_timeout=30,
        )
        SessionServiceGroup({"pg": pg_svc, "tracker": tracker}).run(
            service_id=coordinator_id, namespace="pg-results"
        )

        self.top_results = top_rows
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        urls = discover_services(
            self.coordinator_id,
            names=["pg", "tracker"],
            namespace="pg-results",
            timeout=120,
        )
        self.pg_url      = urls["pg"]
        self.tracker_url = urls["tracker"]
        self.next(self.run_worker, foreach="worker_ids")

    @conda(libraries={"psycopg2": ">=2.9.9"})
    @step
    def run_worker(self):
        import httpx
        import psycopg2

        parsed = urlparse(self.pg_url)
        conn   = psycopg2.connect(
            host=parsed.hostname, port=parsed.port,
            user="postgres", dbname="postgres",
        )
        conn.autocommit = True
        cur = conn.cursor()

        # CREATE TABLE IF NOT EXISTS is safe for concurrent workers in Postgres
        cur.execute("""
            CREATE TABLE IF NOT EXISTS results (
                trial_id TEXT PRIMARY KEY,
                worker_id INT,
                lr        FLOAT,
                dropout   FLOAT,
                accuracy  FLOAT
            )
        """)

        for trial in self.worker_trials[self.input]:
            acc = _simulate_trial(trial["lr"], trial["dropout"])
            cur.execute(
                """
                INSERT INTO results (trial_id, worker_id, lr, dropout, accuracy)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (trial["trial_id"], self.input, trial["lr"], trial["dropout"], acc),
            )

        conn.close()

        try:
            httpx.post(f"{self.tracker_url}/complete", timeout=10).raise_for_status()
        except HTTPX_ERRORS:
            pass

        self.n_trials_run = len(self.worker_trials[self.input])
        self.next(self.join_workers)

    # ── Reduce ────────────────────────────────────────────────────────────────

    @step
    @worker_join
    def join_workers(self, inputs):
        self.total_trials = sum(inp.n_trials_run for inp in inputs)
        self.next(self.join)

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print(f"\nPostgres Results Store — {self.n_workers} workers, "
              f"{self.total_trials} trials")
        print(f"\n  Top {self.top_k} configurations (queried via SQL):")
        print(f"  {'trial_id':<12} {'lr':>8} {'dropout':>8} {'accuracy':>10}")
        print("  " + "-" * 44)
        for trial_id, lr, dropout, acc in self.top_results:
            print(f"  {trial_id:<12} {lr:>8.4f} {dropout:>8.2f} {acc:>10.4f}")
        print("\n  postgres ran as a ProcessService — no database to provision.")
        print("  Workers wrote to a shared table; coordinator queried top-k via SQL.")


if __name__ == "__main__":
    PostgresResultsFlow()
