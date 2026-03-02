"""
Agent Ensemble — self-consistency classification of support tickets.

Multiple agents independently classify each support ticket into a category.
The coordinator tallies the votes and picks the majority (self-consistency
decoding, Wang et al. 2022).  Even with a 25% per-agent error rate, a 5-agent
ensemble is correct ~97% of the time.

Real-world scenario: automated triage of customer support tickets, moderation
queues, or any multi-label classification task where a single LLM can hallucinate.

Swap `mock_classifier()` for a real API call to deploy this at scale:
    import anthropic
    client = anthropic.Anthropic()
    def classify(ticket, agent_id):
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(ticket=ticket)}]
        )
        return msg.content[0].text.strip()

Architecture:
   start ─── run_coordinator (dispatches tickets, aggregates votes)
           ╲─ launch_agents ─── run_agent × n_agents ─── join_agents ─── join ─── end

Run:
    python examples/agent_ensemble.py run --n_agents 5 --error_rate 0.25
"""
import random
import threading

from metaflow import FlowSpec, Parameter, current, step

from metaflow_coordinator import FastAPIService, await_service, coordinator_join, worker_join, HTTPX_ERRORS

# ── Dataset ────────────────────────────────────────────────────────────────

CATEGORIES = ["bug_report", "feature_request", "billing", "general_inquiry"]

TICKETS = [
    {"id": 0,  "text": "App crashes whenever I try to upload a file larger than 5MB.",         "label": "bug_report"},
    {"id": 1,  "text": "Would love a dark mode option — my eyes are tired at night.",           "label": "feature_request"},
    {"id": 2,  "text": "I was charged twice this month. Please refund the duplicate payment.",  "label": "billing"},
    {"id": 3,  "text": "How do I export my data to CSV?",                                       "label": "general_inquiry"},
    {"id": 4,  "text": "The dashboard shows a 404 error after I log in.",                       "label": "bug_report"},
    {"id": 5,  "text": "Can you add support for Slack notifications?",                          "label": "feature_request"},
    {"id": 6,  "text": "My invoice shows the wrong company name.",                              "label": "billing"},
    {"id": 7,  "text": "What's the difference between the Pro and Enterprise plans?",           "label": "general_inquiry"},
    {"id": 8,  "text": "Getting a null pointer exception in the API client library.",           "label": "bug_report"},
    {"id": 9,  "text": "Please add an undo button to the editor.",                              "label": "feature_request"},
    {"id": 10, "text": "Can I get a receipt for my last payment?",                              "label": "billing"},
    {"id": 11, "text": "Is there an on-premises deployment option?",                            "label": "general_inquiry"},
]

# ── Classifier ─────────────────────────────────────────────────────────────

_KEYWORDS = {
    "bug_report":       ["crash", "error", "broken", "bug", "exception", "null", "404", "500", "not working"],
    "feature_request":  ["add", "would love", "please add", "support for", "can you", "wish", "feature"],
    "billing":          ["charge", "invoice", "payment", "refund", "receipt", "billed", "subscription", "paid"],
    "general_inquiry":  ["how", "what", "where", "is there", "difference", "option", "can i"],
}

_CLASSIFY_PROMPT = """Classify the following customer support ticket into exactly one category:
  bug_report | feature_request | billing | general_inquiry

Ticket: {ticket}
Category:"""


def mock_classifier(ticket_text: str, agent_id: int, error_rate: float = 0.25) -> str:
    """
    Keyword-based classifier with configurable hallucination rate.

    To use a real LLM:
        response = anthropic.Anthropic().messages.create(
            model="claude-opus-4-6", max_tokens=20,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(ticket=ticket_text)}]
        )
        return response.content[0].text.strip().lower()
    """
    text_lower = ticket_text.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text_lower)
        for cat, kws in _KEYWORDS.items()
    }
    predicted = max(scores, key=scores.get) if max(scores.values()) > 0 else "general_inquiry"

    rng = random.Random(hash((ticket_text, agent_id)))
    if rng.random() < error_rate:
        others = [c for c in CATEGORIES if c != predicted]
        return rng.choice(others)
    return predicted


class AgentEnsembleFlow(FlowSpec):
    n_agents   = Parameter("n_agents",   default=5,    type=int,   help="Number of parallel agents")
    error_rate = Parameter("error_rate", default=0.25, type=float, help="Per-agent error rate (0.0–1.0)")

    @step
    def start(self):
        self.coordinator_id  = current.run_id
        self.n_agents_int    = int(self.n_agents)
        self.error_rate_f    = float(self.error_rate)
        self.next(self.run_coordinator, self.launch_agents)

    # ── Coordinator ────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """Dispatch tickets to all agents, aggregate votes, compute consensus."""
        from collections import Counter
        from fastapi import FastAPI

        n_agents  = self.n_agents_int
        tickets   = TICKETS
        n_tickets = len(tickets)

        # answers[ticket_id] = [agent_answer, ...]
        answers:   dict[int, list] = {t["id"]: [] for t in tickets}
        consensus: dict[int, str]  = {}
        done_agents = [0]
        lock        = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title="agent-ensemble")

        @app.get("/tickets")
        async def get_tickets():
            """Return all tickets — every agent classifies every ticket."""
            return {"tickets": tickets}

        @app.post("/answer")
        async def submit_answer(body: dict):
            t_id   = body["ticket_id"]
            answer = body["category"]
            with lock:
                answers[t_id].append(answer)
                if len(answers[t_id]) == n_agents and t_id not in consensus:
                    consensus[t_id] = Counter(answers[t_id]).most_common(1)[0][0]
            return {"ok": True, "votes_so_far": len(answers[t_id])}

        @app.post("/done")
        async def agent_done():
            with lock:
                done_agents[0] += 1
                if done_agents[0] >= n_agents:
                    done_event.set()
            return {"ok": True, "agents_done": done_agents[0]}

        @app.get("/health")
        async def health():
            with lock:
                return {"consensus_ready": len(consensus), "total": n_tickets}

        svc = FastAPIService(app=app, done=done_event, drain_delay=2.0)
        svc.run(service_id=self.coordinator_id, namespace="agent-ensemble")

        # Evaluate against ground truth
        results = []
        for t in tickets:
            pred = consensus.get(t["id"], "?")
            results.append({
                "ticket":    t["text"][:60] + "...",
                "label":     t["label"],
                "consensus": pred,
                "correct":   pred == t["label"],
                "votes":     answers[t["id"]],
            })

        correct = sum(r["correct"] for r in results)
        self.ensemble_results = results
        self.ensemble_accuracy = round(correct / n_tickets, 3)
        self.next(self.join)

    # ── Agents ─────────────────────────────────────────────────────────────

    @step
    def launch_agents(self):
        self.svc_url       = await_service(self.coordinator_id, namespace="agent-ensemble", timeout=120)
        self.agent_indices = list(range(self.n_agents_int))
        self.next(self.run_agent, foreach="agent_indices")

    @step
    def run_agent(self):
        """Fetch all tickets, classify each independently, then report done."""
        import httpx

        agent_id = self.input
        tickets  = httpx.get(f"{self.svc_url}/tickets", timeout=10).json()["tickets"]

        for ticket in tickets:
            category = mock_classifier(ticket["text"], agent_id=agent_id, error_rate=self.error_rate_f)
            try:
                httpx.post(
                    f"{self.svc_url}/answer",
                    json={"ticket_id": ticket["id"], "category": category, "agent_id": agent_id},
                    timeout=10,
                ).raise_for_status()
            except HTTPX_ERRORS:
                break

        try:
            httpx.post(f"{self.svc_url}/done", timeout=10).raise_for_status()
        except HTTPX_ERRORS:
            pass

        self.classifications = len(tickets)
        self.next(self.join_agents)

    @step
    @worker_join
    def join_agents(self, inputs):
        self.total_classifications = sum(inp.classifications for inp in inputs)
        self.next(self.join)

    # ── Final join ─────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        n = len(self.ensemble_results)
        per_agent = 1.0 - self.error_rate_f

        print(f"\nSupport Ticket Classification — {self.n_agents_int} agents, "
              f"{self.error_rate_f:.0%} per-agent error rate")
        print(f"  Expected per-agent accuracy:  ~{per_agent:.0%}")
        print(f"  Ensemble accuracy (majority): {self.ensemble_accuracy:.0%}  "
              f"({sum(r['correct'] for r in self.ensemble_results)}/{n})")
        print()

        # Show disagreements to illustrate the ensemble's benefit
        wrong = [r for r in self.ensemble_results if not r["correct"]]
        agreed = [r for r in self.ensemble_results if len(set(r["votes"])) == 1]

        print("  Results (✓ correct  ✗ wrong):")
        for r in self.ensemble_results:
            mark  = "✓" if r["correct"] else "✗"
            votes = ", ".join(f"{c}×{r['votes'].count(c)}"
                              for c in sorted(set(r["votes"]), key=r["votes"].count, reverse=True))
            print(f"  {mark}  {r['ticket']:<62s}  [{votes}]")

        if wrong:
            print(f"\n  {len(wrong)} ticket(s) misclassified by the majority — "
                  f"likely needs more agents or a stronger model.")
        if agreed:
            print(f"  {len(agreed)}/{n} tickets had unanimous agreement across all agents.")


if __name__ == "__main__":
    AgentEnsembleFlow()
