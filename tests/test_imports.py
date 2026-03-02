"""Smoke-test that all public API symbols are importable."""


def test_service_types():
    from metaflow_coordinator import FastAPIService, ProcessService, SessionServiceGroup
    assert FastAPIService
    assert ProcessService
    assert SessionServiceGroup


def test_completion_types():
    from metaflow_coordinator import (
        CompletionTracker,
        ResultCollector,
        BarrierService,
        WorkQueue,
        SemaphoreService,
    )
    assert CompletionTracker
    assert ResultCollector
    assert BarrierService
    assert WorkQueue
    assert SemaphoreService


def test_rendezvous():
    from metaflow_coordinator import await_service, register_service
    assert await_service
    assert register_service


def test_decorators():
    from metaflow_coordinator import coordinator_join, worker_join, session_service
    assert coordinator_join
    assert worker_join
    assert session_service


def test_exceptions():
    from metaflow_coordinator import ServiceNotReadyError, ServiceConfigError
    assert issubclass(ServiceNotReadyError, Exception)
    assert issubclass(ServiceConfigError, Exception)
