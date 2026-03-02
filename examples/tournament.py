"""
Model Tournament — eliminate-half-per-round hyperparameter search.

Standard grid search evaluates ALL configurations in parallel.  A tournament
lets you quickly find a good model by progressively eliminating the weakest
half, spending more compute on round 2 than round 1 (successive halving).

This example uses scikit-learn's Wine dataset (no download, no credentials).
Workers train real RandomForestClassifier models with different hyperparameters
and return cross-validated accuracy.  The coordinator manages the bracket.

Real-world use: AutoML, prompt-template A/B testing, neural architecture
search — anywhere you have many candidates and expensive evaluation.

Architecture:
   start ─── run_coordinator (tournament bracket)
           ╲─ launch_workers ─── run_worker × n_workers ─── join_workers ─── join ─── end

Round protocol:
   Worker                             Coordinator
     │  GET  /model_to_eval  →        │  serve next model for current round
     │                                │     or {"waiting": true} between rounds
     │                                │     or {"done":    true} when finished
     │  POST /score          →        │  record; when all models scored, advance bracket
     │  loop until {"done"}           │
     │  POST /done           →        │  count workers; shut down when all done

Run:
    python examples/tournament.py run --n_workers 4 --n_rounds 2
"""
import math
import threading
import time

from metaflow import FlowSpec, Parameter, current, step

from metaflow_session_service import FastAPIService, await_service, coordinator_join, worker_join, HTTPX_ERRORS

# ── Candidate model configs ────────────────────────────────────────────────
# All are RandomForestClassifier(random_state=42) on the Wine dataset.
# Mix of good and bad configs to make the tournament interesting.

MODEL_CONFIGS = [
    {"model_id": 0,  "n_estimators": 5,   "max_depth": 2,    "min_samples_split": 10},
    {"model_id": 1,  "n_estimators": 10,  "max_depth": None, "min_samples_split": 2},
    {"model_id": 2,  "n_estimators": 50,  "max_depth": 4,    "min_samples_split": 2},
    {"model_id": 3,  "n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    {"model_id": 4,  "n_estimators": 200, "max_depth": None, "min_samples_split": 5},
    {"model_id": 5,  "n_estimators": 30,  "max_depth": 3,    "min_samples_split": 4},
    {"model_id": 6,  "n_estimators": 100, "max_depth": 6,    "min_samples_split": 2},
    {"model_id": 7,  "n_estimators": 5,   "max_depth": None, "min_samples_split": 2},
]


class TournamentFlow(FlowSpec):
    n_workers = Parameter("n_workers", default=4, type=int, help="Parallel evaluators")
    n_rounds  = Parameter("n_rounds",  default=2, type=int, help="Elimination rounds (halves field each time)")

    @step
    def start(self):
        self.coordinator_id = current.run_id
        self.n_workers_int  = int(self.n_workers)
        self.n_rounds_int   = int(self.n_rounds)
        self.next(self.run_coordinator, self.launch_workers)

    # ── Coordinator ────────────────────────────────────────────────────────

    @step
    def run_coordinator(self):
        """
        Run the elimination bracket.

        Each round: serve all active models, collect scores, eliminate bottom half.
        After n_rounds, the best model(s) survive.  Workers that call /model_to_eval
        between rounds receive {"waiting": true} and should retry shortly.
        """
        from collections import deque
        from fastapi import FastAPI

        n_workers = self.n_workers_int
        n_rounds  = self.n_rounds_int

        active_models  = list(MODEL_CONFIGS)
        round_queue    = deque(active_models)
        round_scores:  dict = {}          # model_id → score this round
        current_round  = [0]
        advancing      = [False]          # True while bracket is advancing between rounds
        finished       = [False]
        bracket_log    = []
        done_workers   = [0]
        lock           = threading.Lock()
        done_event     = threading.Event()

        app = FastAPI(title="tournament")

        @app.get("/model_to_eval")
        async def next_model():
            with lock:
                if finished[0]:
                    return {"done": True}
                if advancing[0]:
                    return {"waiting": True, "round": current_round[0]}
                return round_queue.popleft() if round_queue else {"waiting": True, "round": current_round[0]}

        @app.post("/score")
        async def submit_score(body: dict):
            model_id = body["model_id"]
            score    = body["score"]
            with lock:
                round_scores[model_id] = score

                if len(round_scores) < len(active_models):
                    return {"ok": True}

                # ── All active models scored — run elimination ──
                ranked   = sorted(round_scores.items(), key=lambda kv: -kv[1])
                keep_n   = max(1, len(ranked) // 2)
                survivors = {mid for mid, _ in ranked[:keep_n]}
                eliminated = [mid for mid, _ in ranked[keep_n:]]

                bracket_log.append({
                    "round":     current_round[0],
                    "scores":    dict(round_scores),
                    "survivors": list(survivors),
                    "eliminated": eliminated,
                })

                active_models[:] = [m for m in active_models if m["model_id"] in survivors]
                current_round[0] += 1
                round_scores.clear()

                if current_round[0] >= n_rounds or len(active_models) == 1:
                    finished[0] = True
                    done_event.set()
                else:
                    advancing[0] = True
                    round_queue.clear()
                    for m in active_models:
                        round_queue.append(m)
                    advancing[0] = False

            return {"ok": True}

        @app.post("/done")
        async def worker_done():
            with lock:
                done_workers[0] += 1
                if done_workers[0] >= n_workers:
                    done_event.set()   # also fire if workers finish before coordinator
            return {"ok": True}

        @app.get("/health")
        async def health():
            with lock:
                return {"round": current_round[0], "active": len(active_models), "finished": finished[0]}

        svc = FastAPIService(app=app, done=done_event, drain_delay=3.0)
        svc.run(service_id=self.coordinator_id, namespace="tournament")

        self.bracket_log    = bracket_log
        self.final_survivors = active_models
        self.next(self.join)

    # ── Workers ────────────────────────────────────────────────────────────

    @step
    def launch_workers(self):
        self.svc_url        = await_service(self.coordinator_id, namespace="tournament", timeout=120)
        self.worker_indices = list(range(self.n_workers_int))
        self.next(self.run_worker, foreach="worker_indices")

    @step
    def run_worker(self):
        """
        Evaluate assigned model configs using 5-fold cross-validation on Wine.
        The coordinator assigns models; workers just train and score.
        """
        import httpx
        from sklearn.datasets       import load_wine
        from sklearn.ensemble        import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X, y     = load_wine(return_X_y=True)
        evaluated = 0

        while True:
            try:
                resp = httpx.get(f"{self.svc_url}/model_to_eval", timeout=10).json()
            except HTTPX_ERRORS:
                break

            if resp.get("done"):
                break
            if resp.get("waiting"):
                time.sleep(0.3)
                continue

            model_id = resp["model_id"]
            clf = RandomForestClassifier(
                n_estimators     = resp["n_estimators"],
                max_depth        = resp["max_depth"],
                min_samples_split= resp["min_samples_split"],
                random_state     = 42,
            )
            cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            score     = float(cv_scores.mean())

            try:
                httpx.post(
                    f"{self.svc_url}/score",
                    json={"model_id": model_id, "score": score},
                    timeout=10,
                ).raise_for_status()
            except HTTPX_ERRORS:
                break

            evaluated += 1

        try:
            httpx.post(f"{self.svc_url}/done", timeout=10).raise_for_status()
        except HTTPX_ERRORS:
            pass

        self.models_evaluated = evaluated
        self.next(self.join_workers)

    @step
    @worker_join
    def join_workers(self, inputs):
        self.total_evaluations = sum(inp.models_evaluated for inp in inputs)
        self.next(self.join)

    # ── Final join ─────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        total_models = len(MODEL_CONFIGS)
        print(f"\nRandomForest Tournament on Wine dataset")
        print(f"  {total_models} configs → {self.n_rounds_int} rounds → "
              f"{len(self.final_survivors)} survivor(s)")
        print(f"  Total evaluations: {self.total_evaluations}  "
              f"(vs {total_models * self.n_rounds_int} for full grid search)")

        for entry in self.bracket_log:
            r       = entry["round"]
            scores  = entry["scores"]
            ranked  = sorted(scores.items(), key=lambda kv: -kv[1])
            print(f"\n  Round {r + 1}  ({len(ranked)} models):")
            for mid, score in ranked:
                cfg   = MODEL_CONFIGS[mid]
                tag   = "  ✓ survives" if mid in entry["survivors"] else "  ✗ eliminated"
                print(f"    Model {mid:2d}: acc={score:.4f}  "
                      f"n_est={cfg['n_estimators']:3d}  "
                      f"depth={str(cfg['max_depth']):4s}  "
                      f"split={cfg['min_samples_split']}{tag}")

        if self.final_survivors:
            w = self.final_survivors[0]
            # Run final eval on winner to show clean result
            from sklearn.datasets       import load_wine
            from sklearn.ensemble        import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            import numpy as np
            X, y = load_wine(return_X_y=True)
            clf = RandomForestClassifier(
                n_estimators     = w["n_estimators"],
                max_depth        = w["max_depth"],
                min_samples_split= w["min_samples_split"],
                random_state     = 42,
            )
            final_acc = cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean()
            print(f"\n  🏆  Champion: Model {w['model_id']}  "
                  f"n_estimators={w['n_estimators']}  "
                  f"max_depth={w['max_depth']}  "
                  f"min_samples_split={w['min_samples_split']}")
            print(f"     10-fold CV accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    TournamentFlow()
