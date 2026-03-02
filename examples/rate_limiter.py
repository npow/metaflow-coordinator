"""
Rate-Limited API Gateway — shared concurrency throttle for external API calls.

When many parallel workers all call an external API (OpenAI, Anthropic, a
payment processor, a rate-limited internal service), they quickly saturate
the API's concurrency or RPM limit, triggering 429 errors and forcing
workers into exponential backoff — wasted time and noisy logs.

A coordinator-side semaphore gives every worker fair, ordered access.
Workers call POST /acquire to enter the semaphore (blocks until a slot
opens), do their work, then call POST /release.  The coordinator admits
at most max_concurrent workers at a time — across the entire fleet.

Real-world swap-in: replace mock_llm_call() with any API call:
    import anthropic
    client = anthropic.Anthropic()
    def classify(text):
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=5,
            messages=[{"role": "user", "content": f"Sentiment of: {text}\\nAnswer positive/negative/neutral"}]
        )
        return msg.content[0].text.strip().lower()

Architecture:
   start ─── run_coordinator (semaphore + result store)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Protocol:
   Worker → POST /acquire    (blocks until slot opens)
   Worker → calls the API    (mock: 30 ms sleep)
   Worker → POST /release    (frees the slot, submits result)
   Worker → POST /done       (count workers; shut down when all done)

Run:
    python examples/rate_limiter.py run --n_workers 8 --max_concurrent 3
"""
import time

from metaflow import FlowSpec, Parameter, current, step

from metaflow_coordinator import SemaphoreService, await_service, coordinator_join, worker_join, HTTPX_ERRORS

# ── Sentiment dataset ────────────────────────────────────────────────────────

REVIEWS = [
    ("The product arrived quickly and works exactly as described.",        "positive"),
    ("Battery died after two days. Complete waste of money.",              "negative"),
    ("Average quality. Nothing special but does the job.",                 "neutral"),
    ("Best purchase I've made this year! Highly recommended.",             "positive"),
    ("The size was completely wrong even though I followed the guide.",    "negative"),
    ("Decent for the price. Wouldn't pay more for it.",                    "neutral"),
    ("Absolutely love it. Already ordered a second one.",                  "positive"),
    ("Stopped working after a week. No response from support.",            "negative"),
    ("Exactly what I needed. Fast shipping too.",                          "positive"),
    ("Looks cheap in person. Returning it.",                               "negative"),
    ("Works fine. Instructions could be clearer.",                         "neutral"),
    ("Exceeded expectations. The quality is outstanding.",                 "positive"),
    ("Missing pieces in the box. Had to call support.",                    "negative"),
    ("Solid product. Used it daily for six months with no issues.",        "positive"),
    ("Overpriced for what you get. Similar items cost half as much.",      "negative"),
    ("Good value. Would buy again.",                                       "positive"),
    ("Very noisy. Disturbs everyone in the room.",                        "negative"),
    ("Just okay. Does what it says, nothing more.",                        "neutral"),
    ("Fantastic build quality. Feels premium.",                            "positive"),
    ("Arrived damaged. Packaging was terrible.",                           "negative"),
    ("Does the job. Not exciting but reliable.",                           "neutral"),
    ("Transformative product. Changed how I work.",                        "positive"),
    ("Poor instructions. Took two hours to set up something simple.",      "negative"),
    ("Nice design, comfortable to use for long sessions.",                 "positive"),
]

# ── Mock LLM ─────────────────────────────────────────────────────────────────

_POS = ["love", "best", "excellent", "highly", "fantastic", "great",
        "outstanding", "solid", "premium", "transformative", "nice",
        "exceeded", "quickly", "perfect", "reliable"]
_NEG = ["died", "waste", "wrong", "stopped", "cheap", "missing",
        "damaged", "noisy", "overpriced", "poor", "terrible", "broken"]


def mock_llm_call(text: str) -> str:
    """
    Keyword-based classifier with 30 ms simulated latency.

    Replace with a real API call:
        response = anthropic.Anthropic().messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=5,
            messages=[{"role": "user", "content": f"Sentiment: {text}"}]
        )
        return response.content[0].text.strip().lower()
    """
    time.sleep(0.03)
    tl = text.lower()
    pos = sum(1 for w in _POS if w in tl)
    neg = sum(1 for w in _NEG if w in tl)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


# ── Flow ─────────────────────────────────────────────────────────────────────

class RateLimiterFlow(FlowSpec):
    n_workers      = Parameter("n_workers",      default=8, type=int,
                               help="Parallel workers")
    max_concurrent = Parameter("max_concurrent", default=3, type=int,
                               help="Max concurrent API calls across all workers")

    @step
    def start(self):
        self.coordinator_id     = current.run_id
        self.n_workers_int      = int(self.n_workers)
        self.max_concurrent_int = int(self.max_concurrent)
        self.reviews            = REVIEWS
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ──────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        Semaphore service: admit at most max_concurrent workers into the API
        simultaneously.  Workers call /acquire (blocks), do their work, then
        /release and /done.  All blocking happens in asyncio — zero threads-per-
        worker on the coordinator side.
        """
        sem = SemaphoreService(
            max_concurrent=self.max_concurrent_int,
            n_workers=self.n_workers_int,
            drain_delay=2.0,
        )
        sem.run(service_id=self.coordinator_id, namespace="rate-limiter")
        self.semaphore_stats = sem.stats
        self.next(self.join)

    # ── Workers ───────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.svc_url        = await_service(self.coordinator_id, namespace="rate-limiter", timeout=120)
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """
        For each assigned document:
          1. Acquire a semaphore slot  (may wait)
          2. Call the API              (always ≤ max_concurrent workers here at once)
          3. Release the slot; store result locally
        Results are aggregated in join_workers.
        """
        import httpx

        my_docs = [
            (i, text, label)
            for i, (text, label) in enumerate(self.reviews)
            if i % self.n_workers_int == self.input
        ]
        results_local: dict = {}
        classified = 0

        for doc_id, text, _ in my_docs:
            try:
                httpx.post(
                    f"{self.svc_url}/acquire",
                    json={"worker_id": self.input},
                    timeout=120,   # generous: may wait if all slots are taken
                ).raise_for_status()
            except HTTPX_ERRORS:
                break

            label = mock_llm_call(text)
            results_local[doc_id] = label

            try:
                httpx.post(f"{self.svc_url}/release", json={}, timeout=10).raise_for_status()
            except HTTPX_ERRORS:
                break

            classified += 1

        try:
            httpx.post(f"{self.svc_url}/done", json={}, timeout=10)
        except HTTPX_ERRORS:
            pass

        self.results_local = results_local
        self.classified    = classified
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.classification_results = {}
        for inp in inputs:
            self.classification_results.update(inp.results_local)
        self.total_classified = sum(inp.classified for inp in inputs)
        self.next(self.join)

    # ── Final join ────────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        from collections import Counter

        s   = self.semaphore_stats
        n   = len(REVIEWS)
        mc  = self.max_concurrent_int
        nw  = self.n_workers_int
        avg = s["total_wait_ms"] / max(s["calls"], 1)

        print(f"\nSentiment Classification — {nw} workers, max {mc} concurrent API calls")
        print(f"  Reviews classified: {self.total_classified}/{n}")
        print(f"  Total API calls:    {s['calls']}")
        print(f"  Avg slot wait:      {avg:.1f} ms")
        print(f"  Peak slot wait:     {s['peak_wait_ms']:.1f} ms")

        counts = Counter(self.classification_results.values())
        print("\n  Sentiment distribution:")
        for label in ["positive", "negative", "neutral"]:
            count = counts.get(label, 0)
            bar   = "█" * count
            print(f"    {label:<10s} {bar} {count}")

        correct = sum(
            1 for i, (_, true_label) in enumerate(REVIEWS)
            if i in self.classification_results
            and self.classification_results[i] == true_label
        )
        print(f"\n  Accuracy: {correct}/{n} ({correct / n:.0%})")
        print(f"\n  With max_concurrent={mc}: at most {mc} workers called the")
        print(f"  API simultaneously — regardless of how many workers are running.")
        print(f"  Replace mock_llm_call() with a real API to use this in production.")


if __name__ == "__main__":
    RateLimiterFlow()
