"""
@session_service step decorator.

Wraps a Metaflow step so that after the step body runs (which should only
call self.next(...)), a SessionService is started and the process blocks
until the service is done.
"""
from __future__ import annotations

import functools
from typing import Callable

from .service import SessionService


def session_service(
    factory: Callable,
    service_id_attr: str = "coordinator_id",
    namespace: str = "metaflow-svc",
    on_complete: Callable | None = None,
) -> Callable:
    """
    Decorator for a Metaflow step that should host a SessionService.

    Args:
        factory:          Callable ``(step_self) -> SessionService``.
                          Called after the step body runs so it can read ``self.*``.
        service_id_attr:  Name of the ``self`` attribute holding the run-unique
                          service identifier (default ``"coordinator_id"``).
        namespace:        Rendezvous namespace for service discovery.
        on_complete:      Optional ``(step_self) -> None`` called after the
                          service exits cleanly.

    Example::

        @session_service(
            factory=lambda self: CompletionTracker(n_workers=self.n_workers_int),
            service_id_attr="coordinator_id",
            namespace="my-flow",
        )
        @step
        def run_coordinator(self):
            self.next(self.join)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self) -> None:
            # Execute the step body first (records self.next(...))
            func(self)

            # Build the service (deferred so it can read self.*)
            svc: SessionService = factory(self)

            service_id = getattr(self, service_id_attr)
            svc.run(service_id=service_id, namespace=namespace)

            if on_complete is not None:
                on_complete(self)

        return wrapper

    return decorator
