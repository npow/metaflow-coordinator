"""
Priority Queue — best-first processing with dynamic re-prioritisation.

A coordinator maintains a max-priority heap backed by an asyncio.Condition.
Workers always receive the highest-priority pending item.  When a worker
confirms that a post IS spam it calls POST /boost, which raises the priority
of the same author's other pending posts so they surface to the front.

Without re-prioritisation: spam from prolific authors gets reviewed in
arbitrary order, losing the investigative context.
With re-prioritisation: once one post is confirmed spam, the author's
remaining posts jump to the front — reviewers see the full picture at once.

This is fundamentally different from a plain work queue (unordered FIFO) or
the tournament pattern (items are eliminated).  Here items are reordered
dynamically as workers discover new information; nothing is discarded.

Architecture:
   start ─── run_coordinator (priority dict + asyncio.Condition)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Protocol:
   Worker → POST /next         (blocks until highest-priority item is free)
   Worker → POST /complete     (marks item done, submits verdict)
   Worker → POST /boost        (escalate all pending posts from same author)

Run:
    python examples/priority_queue.py run --n_workers 4
"""
import threading
import time

from metaflow import FlowSpec, Parameter, current, step

from metaflow_coordinator import FastAPIService, await_service, coordinator_join, worker_join

# ── Dataset ───────────────────────────────────────────────────────────────────

# (post_id, author, text, initial_severity, true_label)
POSTS = [
    (0,  "alice",   "Just ordered the new model — excited!",                      5,  "ok"),
    (1,  "bob99",   "CLICK HERE for free iPhone — limited time!!!",               85, "spam"),
    (2,  "carol",   "Anyone else having issues with checkout?",                   15, "ok"),
    (3,  "bob99",   "Win $1000 cash — survey takes 30 seconds",                  82, "spam"),
    (4,  "dave",    "Great customer service, resolved my issue in minutes.",       8,  "ok"),
    (5,  "eve42",   "Buy cheap meds no prescription needed",                      90, "spam"),
    (6,  "alice",   "The packaging was a bit damaged but product is fine.",       12, "ok"),
    (7,  "bob99",   "Make money from home — no experience required",             78, "spam"),
    (8,  "frank",   "Returned it — didn't match the description.",                18, "ok"),
    (9,  "eve42",   "Cheap Rolex replica ships from warehouse",                   88, "spam"),
    (10, "grace",   "Five stars, will buy again.",                                 4,  "ok"),
    (11, "bob99",   "Lose 10 kg in 10 days — doctor approved",                   80, "spam"),
    (12, "henry",   "Took a week to arrive but worth the wait.",                  10, "ok"),
    (13, "eve42",   "Earn $500/day working from home — I made $2000 last week",  87, "spam"),
    (14, "irene",   "Product stopped working after two months.",                  22, "ok"),
    (15, "bob99",   "Free gift with every order — click to claim",               76, "spam"),
    (16, "jack",    "Good quality, fast shipping.",                                6,  "ok"),
    (17, "eve42",   "Online casino — 200% welcome bonus no deposit",             91, "spam"),
    (18, "karen",   "Instructions were unclear but figured it out.",              14, "ok"),
    (19, "bob99",   "MLM opportunity — be your own boss",                        83, "spam"),
]

N_POSTS = len(POSTS)


def _moderate(text: str) -> str:
    """Keyword-based deep classifier with 20 ms simulated latency."""
    time.sleep(0.02)
    SPAM_WORDS = {
        "free", "click", "win", "cash", "cheap", "meds", "money",
        "prescription", "replica", "casino", "bonus", "mlm", "opportunity",
        "lose", "earn", "survey", "limited", "doctor approved",
    }
    hits = sum(1 for kw in SPAM_WORDS if kw in text.lower())
    return "spam" if hits >= 2 else "ok"


# ── Flow ──────────────────────────────────────────────────────────────────────

class PriorityQueueFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=4, type=int,
                          help="Parallel moderator workers")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ──────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        Best-first dispatcher backed by asyncio.Condition + a priority dict.

        The priority dict allows O(1) updates for /boost.  asyncio.Condition
        provides efficient blocking so workers never spin-poll: a worker
        waiting in /next is suspended until /boost or /complete notifies it
        that a new highest-priority item is available.
        """
        import asyncio
        from fastapi import FastAPI

        n_workers = self.n_workers_int

        priorities: dict[int, int] = {pid: score for pid, _, _, score, _ in POSTS}
        issued:     set[int]       = set()
        completed:  set[int]       = set()
        verdicts:   dict[int, str] = {}

        author_posts: dict[str, list[int]] = {}
        for pid, author, _, _, _ in POSTS:
            author_posts.setdefault(author, []).append(pid)

        done_event = threading.Event()

        app  = FastAPI(title="priority-queue")
        _cond: list[asyncio.Condition] = []

        @app.on_event("startup")
        async def _startup():
            _cond.append(asyncio.Condition())

        @app.post("/next")
        async def next_item(body: dict):
            """Return the highest-priority pending item; block if none are free."""
            async with _cond[0]:
                while True:
                    if len(issued) + len(completed) >= N_POSTS:
                        return {"done": True}
                    pending = [
                        (priorities[pid], pid)
                        for pid in priorities
                        if pid not in issued and pid not in completed
                    ]
                    if pending:
                        prio, pid     = max(pending)
                        issued.add(pid)
                        _, author, text, _, _ = POSTS[pid]
                        return {"post_id": pid, "author": author,
                                "text": text, "priority": prio}
                    await _cond[0].wait()

        @app.post("/complete")
        async def complete(body: dict):
            pid     = body["post_id"]
            verdict = body["verdict"]
            async with _cond[0]:
                issued.discard(pid)
                completed.add(pid)
                verdicts[pid] = verdict
                if len(completed) >= N_POSTS:
                    done_event.set()
                _cond[0].notify_all()
            return {"ok": True}

        @app.post("/boost")
        async def boost(body: dict):
            """Raise priority of all pending posts from the confirmed-spam author."""
            author    = body["author"]
            boost_amt = body.get("amount", 25)
            boosted   = []
            async with _cond[0]:
                for pid in author_posts.get(author, []):
                    if pid not in completed and pid not in issued:
                        new_p = min(100, priorities[pid] + boost_amt)
                        if new_p > priorities[pid]:
                            priorities[pid] = new_p
                            boosted.append(pid)
                if boosted:
                    _cond[0].notify_all()
            return {"ok": True, "boosted": boosted}

        @app.get("/health")
        async def health():
            return {"completed": len(completed), "in_flight": len(issued),
                    "total": N_POSTS}

        svc = FastAPIService(app=app, done=done_event, drain_delay=2.0)
        svc.run(service_id=self.coordinator_id, namespace="priority-queue")

        self.verdicts         = verdicts
        self.final_priorities = dict(priorities)
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.svc_url      = await_service(self.coordinator_id,
                                          namespace="priority-queue", timeout=120)
        self.worker_range = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_range")

    @step
    def run_worker(self):
        import httpx

        processed = 0

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

            verdict = _moderate(item["text"])

            if verdict == "spam":
                try:
                    httpx.post(
                        f"{self.svc_url}/boost",
                        json={"author": item["author"], "amount": 25},
                        timeout=10,
                    )
                except Exception:
                    pass

            try:
                httpx.post(
                    f"{self.svc_url}/complete",
                    json={"post_id": item["post_id"], "verdict": verdict},
                    timeout=10,
                ).raise_for_status()
            except Exception:
                break

            processed += 1

        self.processed = processed
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.total_processed = sum(inp.processed for inp in inputs)
        self.next(self.join)

    # ── Final join ────────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        spam    = sum(1 for v in self.verdicts.values() if v == "spam")
        ok      = sum(1 for v in self.verdicts.values() if v == "ok")
        correct = sum(
            1 for pid, _, _, _, true in POSTS
            if pid in self.verdicts and self.verdicts[pid] == true
        )

        print(f"\nContent Moderation — {self.n_workers_int} workers, best-first")
        print(f"  Posts reviewed:  {len(self.verdicts)}/{N_POSTS}")
        print(f"  Spam found:      {spam}")
        print(f"  OK posts:        {ok}")
        print(f"  Accuracy:        {correct}/{N_POSTS} ({correct/N_POSTS:.0%})")

        initial = {pid: score for pid, _, _, score, _ in POSTS}
        boosted = {
            pid: (initial[pid], self.final_priorities[pid])
            for pid in initial
            if self.final_priorities[pid] > initial[pid]
        }
        if boosted:
            print(f"\n  Dynamic priority boosts (initial → final):")
            for pid in sorted(boosted, key=lambda p: boosted[p][1], reverse=True):
                _, author, text, _, _ = POSTS[pid]
                init, final = boosted[pid]
                print(f"    [{author}] post {pid}: {init} → {final}"
                      f"  \"{text[:55]}\"")

        print(f"\n  Best-first ordering ensured that posts from confirmed")
        print(f"  spammers were immediately escalated and reviewed as a cluster.")
        print(f"  Replace _moderate() with a real classifier or LLM call.")


if __name__ == "__main__":
    PriorityQueueFlow()
