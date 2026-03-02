class ServiceNotReadyError(Exception):
    """Raised when await_service times out before the endpoint is registered."""


class ServiceConfigError(Exception):
    """Raised when a SessionService is misconfigured."""
