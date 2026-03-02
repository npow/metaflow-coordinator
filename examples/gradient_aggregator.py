"""
Gradient Aggregator — federated learning over data shards.

Each worker trains a local model on its data shard and sends gradients to the
coordinator.  The coordinator averages the gradients (FedAvg) and broadcasts
the global update back.  Workers apply the update and start the next round.

This is the core communication pattern in federated/distributed training —
no shared storage, no data movement, just gradient exchange.

Architecture:
                     ┌──────────────────────────────────────────────┐
                     │  run_coordinator                              │
   start ────────────│  (FedAvg service: collects, averages, serves) │──── join ── end
            ╲        └──────────────────────────────────────────────┘     /
             ╲─── launch_workers ──── run_worker × n_workers ─────────────

Per-round protocol:
   Worker                         Coordinator
     │  POST /gradient/{round}  → │  buffer until n_workers arrive
     │                            │  compute mean gradient
     │  GET  /average/{round}   ← │  return {"ready": true, "gradient": [...]}
     │  apply update, next round  │

Run:
    python examples/gradient_aggregator.py run --n_workers 4 --n_rounds 5
"""
import random
import threading

from metaflow import FlowSpec, Parameter, current, step

from metaflow_session_service import FastAPIService, await_service, coordinator_join, worker_join


# Dimension of the toy model parameter vector
PARAM_DIM = 16


class GradientAggregatorFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=4,  type=int, help="Number of workers / data shards")
    n_rounds  = Parameter("n_rounds",  default=5,  type=int, help="Training rounds")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.n_rounds_int   = int(self.n_rounds)
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator branch ─────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """Collect per-round gradients, compute FedAvg, serve back to workers."""
        from fastapi import FastAPI

        n_workers   = self.n_workers_int
        n_rounds    = self.n_rounds_int
        # round_num → {worker_id: gradient}
        gradients: dict[int, dict] = {}
        averages:  dict[int, list] = {}   # round_num → averaged gradient
        done_workers = [0]
        lock        = threading.Lock()
        done_event  = threading.Event()

        app = FastAPI(title="gradient-aggregator")

        @app.post("/gradient/{round_num}")
        async def receive_gradient(round_num: int, body: dict):
            """Accept a gradient from one worker for the given round."""
            worker_id = body["worker_id"]
            gradient  = body["gradient"]
            with lock:
                if round_num not in gradients:
                    gradients[round_num] = {}
                gradients[round_num][worker_id] = gradient

                if len(gradients[round_num]) == n_workers:
                    # All workers submitted for this round — compute FedAvg
                    all_grads = list(gradients[round_num].values())
                    averages[round_num] = [
                        sum(g[i] for g in all_grads) / n_workers
                        for i in range(PARAM_DIM)
                    ]
            return {"ok": True}

        @app.get("/average/{round_num}")
        async def get_average(round_num: int):
            """Return the averaged gradient once all workers have submitted."""
            with lock:
                if round_num in averages:
                    return {"ready": True, "gradient": averages[round_num]}
            return {"ready": False}

        @app.post("/done")
        async def worker_done():
            """A worker finished all rounds; shut down when all workers report."""
            with lock:
                done_workers[0] += 1
                if done_workers[0] >= n_workers:
                    done_event.set()
            return {"ok": True}

        @app.get("/health")
        async def health():
            with lock:
                return {
                    "rounds_complete": len(averages),
                    "total_rounds":    n_rounds,
                    "workers_done":    done_workers[0],
                }

        svc = FastAPIService(app=app, done=done_event)
        svc.run(service_id=self.coordinator_id, namespace="gradient-aggregator")

        self.loss_curve = [
            # L2 norm of average gradient per round — proxy for convergence
            sum(g ** 2 for g in avg) ** 0.5
            for _, avg in sorted(averages.items())
        ]
        self.next(self.join)

    # ── Worker branch ──────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.svc_url        = await_service(self.coordinator_id, namespace="gradient-aggregator", timeout=120)
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """
        Simulate a worker training on a local data shard.

        Each worker maintains a parameter vector and sends gradients every
        round.  The "gradient" here is a noisy direction toward a fixed target
        (simulating convergence without any ML framework dependency).
        """
        import time
        import httpx

        worker_id = self.input
        rng       = random.Random(worker_id * 7919)  # reproducible per-worker

        # Each worker's data shard points toward a slightly different "target"
        target = [rng.gauss(0.5, 0.2) for _ in range(PARAM_DIM)]
        params = [rng.gauss(0.0, 1.0) for _ in range(PARAM_DIM)]

        for round_num in range(self.n_rounds_int):
            # Compute gradient: direction toward local target + data noise
            gradient = [
                (p - t) + rng.gauss(0, 0.05)
                for p, t in zip(params, target)
            ]

            # Send gradient to coordinator
            httpx.post(
                f"{self.svc_url}/gradient/{round_num}",
                json={"worker_id": worker_id, "gradient": gradient},
                timeout=30,
            ).raise_for_status()

            # Poll for the global averaged gradient
            while True:
                resp = httpx.get(f"{self.svc_url}/average/{round_num}", timeout=10).json()
                if resp["ready"]:
                    avg_grad = resp["gradient"]
                    break
                time.sleep(0.2)

            # Apply FedAvg update (SGD step on global gradient)
            lr = 0.1
            params = [p - lr * g for p, g in zip(params, avg_grad)]

        # Signal completion
        httpx.post(f"{self.svc_url}/done", timeout=30).raise_for_status()
        self.final_params = params
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.all_final_params = [inp.final_params for inp in inputs]
        # Average final params across workers (final global model)
        n = len(self.all_final_params)
        self.global_params = [
            sum(w[i] for w in self.all_final_params) / n
            for i in range(PARAM_DIM)
        ]
        self.next(self.join)

    # ── Final join ─────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print(f"\nFedAvg over {self.n_rounds_int} rounds with {self.n_workers_int} workers")
        print("\nGradient norm per round (↓ = converging):")
        for i, loss in enumerate(self.loss_curve):
            bar = "█" * int(loss * 20)
            print(f"  Round {i + 1}: {bar} {loss:.4f}")
        l2 = sum(p ** 2 for p in self.global_params) ** 0.5
        print(f"\nGlobal model L2 norm: {l2:.4f}")


if __name__ == "__main__":
    GradientAggregatorFlow()
