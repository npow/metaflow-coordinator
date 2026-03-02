"""
Step decorators for automatic artifact merging in coordinator+worker DAGs.

coordinator_join  — for the final two-branch join (coordinator ∥ workers).
worker_join       — for foreach-reduce steps (join_workers).

Both decorators run smart_merge_artifacts before the step body, so the step
body only needs to contain custom aggregation logic and self.next().

Usage::

    # Final join — merges coordinator branch + worker branch
    @step
    @coordinator_join
    def join(self, inputs):
        self.next(self.end)

    # Suppress warnings for a known conflict:
    @step
    @coordinator_join(skip=["svc_url"])
    def join(self, inputs):
        self.svc_url = next(inp.svc_url for inp in inputs if hasattr(inp, "svc_url"))
        self.next(self.end)

    # Foreach reduce — merges symmetric start-attrs, skips per-worker attrs
    @step
    @worker_join
    def join_workers(self, inputs):
        self.results = [inp.result for inp in inputs]   # custom aggregation only
        self.next(self.join)

Merge semantics
---------------
coordinator_join:
  - Artifact on one branch       → copied unconditionally.
  - Same value on all branches   → copied once.
  - Different values (conflict)  → first branch wins + warning emitted.
    Pass skip=["name"] to suppress the warning and handle it manually.

worker_join:
  - Artifact on one branch       → copied unconditionally.
  - Same value on all branches   → copied once.
  - Different values (conflict)  → silently skipped (expected in foreach reduce).
    The step body handles per-worker attrs via explicit aggregation.

In both cases, any attribute set by the step body after the decorator runs
overrides whatever the auto-merge wrote.
"""
from __future__ import annotations

import functools
import warnings
from typing import Callable


def _artifact_names(inp) -> set[str]:
    """Return user artifact names for a join input, or empty set on failure.

    Uses ``inp._datastore.items()`` — the same API that Metaflow's built-in
    ``merge_artifacts`` uses internally.  Each join-step input is a frozen
    FlowSpec clone; its ``_datastore`` maps artifact names to content hashes.
    """
    try:
        return {var for var, _ in inp._datastore.items() if not var.startswith("_")}
    except Exception:
        return set()


def smart_merge_artifacts(
    flow_self,
    inputs,
    skip: list[str] | None = None,
    on_conflict: str = "warn_and_copy",
) -> None:
    """
    Merge artifacts from all branches into *flow_self*.

    Args:
        flow_self:    The ``self`` of the join step.
        inputs:       The ``inputs`` parameter of the join step.
        skip:         Artifact names to leave untouched (user handles manually).
        on_conflict:  What to do when the same artifact has different values
                      across branches:
                      ``"warn_and_copy"`` — first branch wins, warning emitted.
                      ``"silent_skip"``   — silently skip (no value set).
    """
    skip_set = set(skip or [])

    per_input: dict = {}
    for inp in inputs:
        per_input[inp] = _artifact_names(inp) - skip_set

    all_names: set[str] = set().union(*per_input.values()) if per_input else set()

    flow_cls = type(flow_self)

    for name in all_names:
        # Skip class-level properties (Parameters converted by Metaflow's
        # _init_parameters into read-only properties).  Matches the check in
        # Metaflow's own TaskDataStore.persist().
        if isinstance(getattr(flow_cls, name, None), property):
            continue

        holders = [
            (inp, getattr(inp, name))
            for inp, names in per_input.items()
            if name in names
        ]

        if len(holders) == 1:
            setattr(flow_self, name, holders[0][1])
            continue

        values = [v for _, v in holders]
        try:
            all_equal = all(v == values[0] for v in values[1:])
        except Exception:
            all_equal = False

        if all_equal:
            setattr(flow_self, name, values[0])
        elif on_conflict == "warn_and_copy":
            warnings.warn(
                f"coordinator_join: artifact {name!r} has different values across "
                f"branches; using value from first branch. "
                f"Pass skip={[name]!r} to suppress this warning.",
                stacklevel=3,
            )
            setattr(flow_self, name, values[0])
        # else "silent_skip": leave unset, step body handles it


def _make_join_decorator(on_conflict: str) -> Callable:
    """Factory that returns a join decorator with the given conflict strategy."""

    def join_decorator(
        func: Callable | None = None,
        *,
        skip: list[str] | None = None,
    ) -> Callable:
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(self, inputs) -> None:
                try:
                    smart_merge_artifacts(
                        self, inputs, skip=skip, on_conflict=on_conflict
                    )
                except Exception:
                    # Graph-parsing mode or unexpected failure — let step body run.
                    pass
                f(self, inputs)

            return wrapper

        if func is not None:
            # Used as @decorator (no parentheses)
            return decorator(func)
        # Used as @decorator(...) (with parentheses)
        return decorator

    return join_decorator


coordinator_join = _make_join_decorator("warn_and_copy")
coordinator_join.__doc__ = """
Step decorator that auto-merges artifacts from a two-branch coordinator+workers
join.  Conflicts (same artifact name, different values) emit a warning and keep
the first branch's value.  Pass skip=[...] to handle specific attrs manually.
"""

worker_join = _make_join_decorator("silent_skip")
worker_join.__doc__ = """
Step decorator that auto-merges artifacts in a foreach-reduce (join_workers)
step.  Symmetric start-attrs are copied automatically; per-worker attrs that
differ across inputs are silently skipped so the step body can aggregate them.
"""
