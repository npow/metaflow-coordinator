"""
Echo service — minimal demonstration of metaflow-coordinator.

Two services run in the coordinator step:
  - echo  : a FastAPIService with a POST /echo endpoint
  - tracker: a CompletionTracker that knows when all workers are done

Workers discover each service URL via await_service(), call the echo
endpoint, then report completion to the tracker.  When all workers
have reported, both services shut down and the coordinator step exits.

Run locally:
    python examples/echo_service.py run --n_workers 4
"""
from metaflow import FlowSpec, Parameter, current, step

from metaflow_coordinator import (
    CompletionTracker,
    FastAPIService,
    SessionServiceGroup,
    discover_services,
    coordinator_join,
    worker_join,
)


class EchoServiceFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=3, type=int, help="Number of workers")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int = int(self.n_workers)
        # Parallel branches: coordinator blocks while workers run
        self.next(self.run_coordinator, self.launch_workers)

    # ------------------------------------------------------------------
    # Coordinator branch — hosts both services, blocks until workers done
    # ------------------------------------------------------------------

    @step
    def run_coordinator(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        class EchoRequest(BaseModel):
            worker: int
            message: str

        tracker = CompletionTracker(n_workers=self.n_workers_int)

        echo_app = FastAPI(title="echo")

        @echo_app.post("/echo")
        async def echo(req: EchoRequest):
            return {"echo": req.message, "from_worker": req.worker}

        @echo_app.get("/health")
        async def health():
            return {"ready": True}

        echo_svc = FastAPIService(app=echo_app, done=tracker.done)

        group = SessionServiceGroup({"echo": echo_svc, "tracker": tracker})
        group.run(service_id=self.coordinator_id, namespace="echo-example")

        self.next(self.join)

    # ------------------------------------------------------------------
    # Worker branch — discovers services, calls echo, reports done
    # ------------------------------------------------------------------

    @step
    def launch_workers(self):
        urls = discover_services(
            self.coordinator_id,
            names=["echo", "tracker"],
            namespace="echo-example",
            timeout=120,
        )
        self.echo_url    = urls["echo"]
        self.tracker_url = urls["tracker"]
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        import httpx

        resp = httpx.post(
            f"{self.echo_url}/echo",
            json={"worker": self.input, "message": f"hello from worker {self.input}"},
            timeout=30,
        )
        resp.raise_for_status()
        self.echo_response = resp.json()
        print(f"[worker {self.input}] got: {self.echo_response}")

        # Report completion to tracker — this eventually shuts down the coordinator
        httpx.post(f"{self.tracker_url}/complete", timeout=30).raise_for_status()
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.echo_responses = [inp.echo_response for inp in inputs]
        self.next(self.join)

    # ------------------------------------------------------------------
    # Final join
    # ------------------------------------------------------------------

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print(f"All {self.n_workers_int} workers done.")
        for r in getattr(self, "echo_responses", []):
            print(" ", r)


if __name__ == "__main__":
    EchoServiceFlow()
