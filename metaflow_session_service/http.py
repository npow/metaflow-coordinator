"""Shared httpx exception constants for consistent worker error handling."""
from __future__ import annotations

import httpx

#: Exception types raised when the coordinator closes its connection.
#: Catch these in worker loops to exit gracefully when the service shuts down::
#:
#:     while True:
#:         try:
#:             item = httpx.post(f"{url}/pull", json={}).json()
#:         except HTTPX_ERRORS:
#:             break
HTTPX_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)
