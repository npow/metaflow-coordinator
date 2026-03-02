"""
metaflow-coordinator — ephemeral service coordination for Metaflow.

Spin up FastAPI apps or external processes inside a Metaflow step,
register their URLs for discovery by parallel worker steps, then tear
down cleanly when all workers are done.

Typical usage::

    from metaflow_coordinator import (
        FastAPIService,
        ProcessService,
        CompletionTracker,
        SessionServiceGroup,
        await_service,
        session_service,
    )
"""
from .completion import BarrierService, CompletionTracker, ResultCollector, SemaphoreService, WorkQueue
from .http import HTTPX_ERRORS
from .decorators import session_service
from .exceptions import ServiceConfigError, ServiceNotReadyError
from .merge import coordinator_join, smart_merge_artifacts, worker_join
from .rendezvous import (
    await_service,
    discover_services,
    load_checkpoint,
    register_service,
    save_checkpoint,
)
from .service import (
    FastAPIService,
    HttpReady,
    ProcessService,
    SessionService,
    SessionServiceGroup,
    SocketReady,
)

__all__ = [
    # Services
    "SessionService",
    "FastAPIService",
    "ProcessService",
    "SessionServiceGroup",
    "CompletionTracker",
    "ResultCollector",
    "BarrierService",
    "WorkQueue",
    "SemaphoreService",
    # Readiness strategies
    "SocketReady",
    "HttpReady",
    # Rendezvous
    "register_service",
    "await_service",
    "discover_services",
    "save_checkpoint",
    "load_checkpoint",
    # Decorators
    "session_service",
    "coordinator_join",
    "worker_join",
    # Merge helper
    "smart_merge_artifacts",
    # Exceptions
    "ServiceNotReadyError",
    "ServiceConfigError",
    # HTTP helpers
    "HTTPX_ERRORS",
]
