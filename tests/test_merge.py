"""Unit tests for smart_merge_artifacts, coordinator_join, worker_join."""
import warnings

import pytest

from metaflow_coordinator.merge import smart_merge_artifacts, coordinator_join, worker_join


class FakeDatastore:
    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items.items()


class FakeInput:
    def __init__(self, attrs):
        self._datastore = FakeDatastore(attrs)
        for k, v in attrs.items():
            setattr(self, k, v)


class FakeFlow:
    pass


def test_merge_single_branch():
    flow = FakeFlow()
    inp = FakeInput({"x": 42, "y": "hello"})
    smart_merge_artifacts(flow, [inp])
    assert flow.x == 42
    assert flow.y == "hello"


def test_merge_identical_values():
    flow = FakeFlow()
    a = FakeInput({"x": 10})
    b = FakeInput({"x": 10})
    smart_merge_artifacts(flow, [a, b])
    assert flow.x == 10


def test_merge_conflict_warn_and_copy():
    flow = FakeFlow()
    a = FakeInput({"x": 1})
    b = FakeInput({"x": 2})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        smart_merge_artifacts(flow, [a, b], on_conflict="warn_and_copy")
    assert flow.x == 1  # first branch wins
    assert any("x" in str(warning.message) for warning in w)


def test_merge_conflict_silent_skip():
    flow = FakeFlow()
    a = FakeInput({"x": 1})
    b = FakeInput({"x": 2})
    smart_merge_artifacts(flow, [a, b], on_conflict="silent_skip")
    assert not hasattr(flow, "x")


def test_merge_skips_listed_names():
    flow = FakeFlow()
    a = FakeInput({"x": 1, "y": 99})
    b = FakeInput({"x": 2, "y": 99})
    smart_merge_artifacts(flow, [a, b], skip=["x"], on_conflict="warn_and_copy")
    assert not hasattr(flow, "x")  # skipped
    assert flow.y == 99


def test_merge_skips_class_properties():
    """Parameters become class-level properties; must not raise."""
    class FlowWithParam:
        @property
        def n_workers(self):
            return 4

    flow = FlowWithParam()
    inp = FakeInput({"n_workers": 4, "result": "ok"})
    smart_merge_artifacts(flow, [inp])
    assert flow.result == "ok"
    assert flow.n_workers == 4  # property unchanged, not overwritten


def test_coordinator_join_decorator_merges():
    flow = FakeFlow()
    inp = FakeInput({"score": 0.95})

    @coordinator_join
    def join(self, inputs):
        pass

    join(flow, [inp])
    assert flow.score == 0.95


def test_worker_join_decorator_silent_skip_on_conflict():
    flow = FakeFlow()
    a = FakeInput({"result": 1, "shared": "same"})
    b = FakeInput({"result": 2, "shared": "same"})

    @worker_join
    def join_workers(self, inputs):
        pass

    join_workers(flow, [a, b])
    assert not hasattr(flow, "result")  # conflicting → silently skipped
    assert flow.shared == "same"        # identical → copied
