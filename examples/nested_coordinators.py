"""
Nested Coordinators — tree topology for hierarchical search.

Every other example in this library uses a star: one coordinator, N workers.
This example uses a tree:

    top coordinator
    ├── group coordinator 0  (linear_svc)       → 4 worker tasks
    ├── group coordinator 1  (random_forest)    → 4 worker tasks
    └── group coordinator 2  (gradient_boosting)→ 4 worker tasks

Each group coordinator manages hyperparameter trials for one model family.
When a group finishes, its coordinator finds the best config and reports it
up to the top coordinator.  The top coordinator picks the overall winner.

The tree topology has two practical advantages over a flat design:
  1. The top coordinator sees only K champion results (one per family),
     not N×K raw results — easier to interpret and aggregate.
  2. Each level can apply family-specific logic (e.g. different stopping
     criteria per group) without polluting the top coordinator.

The pattern generalises to any hierarchical search: neural architecture
search (top = architecture family, sub = weight hyperparams), geographic
partitioning (top = region, sub = city), or multi-stage recommendation
(top = recall, sub = re-ranking).

Metaflow DAG (nested splits):
   start
   ├── run_top_coordinator          (waits for K group bests)
   └── launch_groups
       └── [foreach group_id]
           └── run_group
               ├── run_group_coordinator   (waits for 4 workers, reports to top)
               └── launch_group_workers
                   └── [foreach config_id] run_group_worker → join_group_workers
               └── join_group
       └── join_groups
   └── join
   └── end

Service IDs (namespace = "nested-search"):
   top coordinator:   {run_id}
   group coordinator: {run_id}/group-{id}

Run:
    python examples/nested_coordinators.py run
"""
import threading

from metaflow import FlowSpec, current, pypi, step

from metaflow_coordinator import FastAPIService, await_service, coordinator_join, worker_join

# ── Search space ──────────────────────────────────────────────────────────────

ALL_GROUP_CONFIGS = [
    {
        "family": "linear_svc",
        "configs": [
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
    },
    {
        "family": "random_forest",
        "configs": [
            {"n_estimators": 10,  "max_depth": 3},
            {"n_estimators": 50,  "max_depth": 5},
            {"n_estimators": 100, "max_depth": None},
            {"n_estimators": 200, "max_depth": None},
        ],
    },
    {
        "family": "gradient_boosting",
        "configs": [
            {"n_estimators": 50,  "learning_rate": 0.10, "max_depth": 2},
            {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
            {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 4},
        ],
    },
]

N_GROUPS  = len(ALL_GROUP_CONFIGS)
N_CONFIGS = len(ALL_GROUP_CONFIGS[0]["configs"])   # same for all groups


def _evaluate(family: str, config: dict) -> float:
    """5-fold cross-validation on the Wine dataset."""
    from sklearn.datasets import load_wine
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X, y = load_wine(return_X_y=True)
    if family == "linear_svc":
        clf = LinearSVC(max_iter=2000, **config)
    elif family == "random_forest":
        clf = RandomForestClassifier(random_state=42, **config)
    else:
        clf = GradientBoostingClassifier(random_state=42, **config)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return float(cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean())


# ── Flow ──────────────────────────────────────────────────────────────────────

class NestedCoordinatorsFlow(FlowSpec):

    @step
    def start(self):
        self.coordinator_id    = current.run_id
        self.all_group_configs = ALL_GROUP_CONFIGS
        self.group_ids         = list(range(N_GROUPS))
        self.next(self.run_top_coordinator, self.launch_groups)

    # ── Level 0: top coordinator ──────────────────────────────────────────────

    @step
    def run_top_coordinator(self):
        """
        Registers at {run_id}.  Accepts POST /group_result from each group
        coordinator.  Blocks until all K groups have reported.
        """
        from fastapi import FastAPI

        group_results: dict[int, dict] = {}
        lock       = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title="top-coordinator")

        @app.post("/group_result")
        async def group_result(body: dict):
            with lock:
                group_results[body["group_id"]] = body
                if len(group_results) >= N_GROUPS:
                    done_event.set()
            return {"ok": True, "received": len(group_results)}

        @app.get("/health")
        async def health():
            return {"groups_received": len(group_results), "total": N_GROUPS}

        svc = FastAPIService(app=app, done=done_event, drain_delay=2.0)
        svc.run(service_id=self.coordinator_id, namespace="nested-search")

        best = max(group_results.values(), key=lambda r: r["best_score"])
        self.overall_best       = best
        self.all_group_summaries = sorted(
            group_results.values(), key=lambda r: r["group_id"]
        )
        self.next(self.join)

    # ── Level 1: group fan-out ────────────────────────────────────────────────

    @step
    def launch_groups(self):
        """Wait for the top coordinator to register, then fan out over families."""
        self.top_url = await_service(
            self.coordinator_id, namespace="nested-search", timeout=120
        )
        self.next(self.run_group, foreach="group_ids")

    @step
    def run_group(self):
        """Entry point for each group branch — sets per-group state."""
        self.group_id     = self.input
        self.group_config = self.all_group_configs[self.group_id]
        self.config_ids   = list(range(len(self.group_config["configs"])))
        self.next(self.run_group_coordinator, self.launch_group_workers)

    # ── Level 1: group coordinator ────────────────────────────────────────────

    @step
    def run_group_coordinator(self):
        """
        Sub-coordinator for one model family.

        1. Registers at {run_id}/group-{id} so workers can find it.
        2. Waits for all N_CONFIGS workers to POST /trial_result.
        3. Picks the best config and POSTs it to the top coordinator.

        This step is both a worker (to the top coordinator) and a coordinator
        (to the group workers) — the defining property of a tree topology.
        """
        import httpx
        from fastapi import FastAPI

        group_id     = self.group_id
        group_config = self.group_config
        n_configs    = len(group_config["configs"])
        top_url      = self.top_url
        coord_id     = self.coordinator_id

        trial_results: dict[int, float] = {}
        lock       = threading.Lock()
        done_event = threading.Event()

        app = FastAPI(title=f"group-coordinator-{group_id}")

        @app.post("/trial_result")
        async def trial_result(body: dict):
            with lock:
                trial_results[body["config_id"]] = body["score"]
                if len(trial_results) >= n_configs:
                    done_event.set()
            return {"ok": True}

        @app.get("/health")
        async def health():
            return {"received": len(trial_results), "total": n_configs}

        svc = FastAPIService(app=app, done=done_event, drain_delay=1.0)
        svc.run(
            service_id=f"{coord_id}/group-{group_id}",
            namespace="nested-search",
        )

        # All workers done — find best config and report up the tree.
        best_config_id = max(trial_results, key=trial_results.get)
        best_score     = trial_results[best_config_id]
        best_config    = group_config["configs"][best_config_id]

        httpx.post(
            f"{top_url}/group_result",
            json={
                "group_id":    group_id,
                "family":      group_config["family"],
                "best_config": best_config,
                "best_score":  best_score,
                "all_scores":  trial_results,
            },
            timeout=30,
        ).raise_for_status()

        self.group_best = {
            "group_id":    group_id,
            "family":      group_config["family"],
            "best_config": best_config,
            "best_score":  best_score,
        }
        self.next(self.join_group)

    # ── Level 2: worker fan-out ───────────────────────────────────────────────

    @step
    def launch_group_workers(self):
        """Discover the group coordinator URL, then fan out over configs."""
        self.group_url = await_service(
            f"{self.coordinator_id}/group-{self.group_id}",
            namespace="nested-search",
            timeout=120,
        )
        self.next(self.run_group_worker, foreach="config_ids")

    @pypi(packages={"scikit-learn": ">=1.4"})
    @step
    def run_group_worker(self):
        """Train and evaluate one hyperparameter config; report to group coordinator."""
        import httpx

        config_id    = self.input
        group_config = self.group_config
        family       = group_config["family"]
        config       = group_config["configs"][config_id]

        score = _evaluate(family, config)

        httpx.post(
            f"{self.group_url}/trial_result",
            json={"config_id": config_id, "config": config, "score": score},
            timeout=30,
        ).raise_for_status()

        self.trial_result = {
            "config_id": config_id,
            "family":    family,
            "config":    config,
            "score":     score,
        }
        self.next(self.join_group_workers)

    @step
    @worker_join
    def join_group_workers(self, inputs):
        self.group_worker_results = [inp.trial_result for inp in inputs]
        self.next(self.join_group)

    # ── Level 1 join ──────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join_group(self, inputs):
        """Merges run_group_coordinator and join_group_workers for this group."""
        self.next(self.join_groups)

    @step
    @worker_join
    def join_groups(self, inputs):
        """Foreach join across all group branches."""
        self.all_group_bests = [inp.group_best for inp in inputs]
        self.next(self.join)

    # ── Top-level join ────────────────────────────────────────────────────────

    @step
    @coordinator_join
    def join(self, inputs):
        """Merges run_top_coordinator and join_groups."""
        self.next(self.end)

    @step
    def end(self):
        n_total = N_GROUPS * N_CONFIGS
        print(f"\nHierarchical Model Search — "
              f"{N_GROUPS} families × {N_CONFIGS} configs = {n_total} total trials\n")

        for summary in self.all_group_summaries:
            family = summary["family"]
            scores = summary["all_scores"]      # keys are strings after JSON round-trip
            best   = summary["best_config"]
            print(f"  {family}:")
            for cid in sorted(scores, key=int):
                cfg     = ALL_GROUP_CONFIGS[summary["group_id"]]["configs"][int(cid)]
                score   = scores[cid]
                marker  = "  ← group winner" if cfg == best else ""
                cfg_str = "  ".join(f"{k}={v}" for k, v in cfg.items())
                print(f"    config {cid}  {cfg_str:<40s}  {score:.4f}{marker}")
            print()

        best    = self.overall_best
        cfg_str = "  ".join(f"{k}={v}" for k, v in best["best_config"].items())
        print(f"  Overall winner:  {best['family']}")
        print(f"  Best config:     {cfg_str}")
        print(f"  CV accuracy:     {best['best_score']:.4f}")
        print(f"\n  Coordination tree:")
        print(f"    1 top coordinator     — collected {N_GROUPS} group results")
        print(f"    {N_GROUPS} group coordinators — each collected {N_CONFIGS} worker results")
        print(f"    {n_total} workers            — each trained one model")
        print(f"\n  The top coordinator never saw raw trial scores — only the")
        print(f"  per-family champion.  Add more families without changing the")
        print(f"  top coordinator: it only cares that K groups report a result.")


if __name__ == "__main__":
    NestedCoordinatorsFlow()
