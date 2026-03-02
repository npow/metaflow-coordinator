"""
Namespace-parameterized rendezvous: a service writes its URL to a well-known
location; worker tasks poll until it appears.

Local mode  → /tmp/mf-svc-{namespace}-{service_id_safe}.json
Remote mode → s3://{bucket}/{prefix}/{namespace}/{service_id}/endpoint
              (detected via AWS_BATCH_JOB_ID or METAFLOW_KUBERNETES_WORKLOAD)

Checkpoint  → same path + "-checkpoint" suffix (local) or "/checkpoint" (S3)
"""
from __future__ import annotations

import json
import os
import time


def _is_remote() -> bool:
    return bool(
        os.environ.get("AWS_BATCH_JOB_ID")
        or os.environ.get("METAFLOW_KUBERNETES_WORKLOAD")
    )


def _safe(s: str) -> str:
    """Replace path-unsafe characters for local file names."""
    return s.replace("/", "__").replace(":", "_")


def _local_path(namespace: str, service_id: str, suffix: str = "") -> str:
    return f"/tmp/mf-svc-{_safe(namespace)}-{_safe(service_id)}{suffix}.json"


def _s3_key(namespace: str, service_id: str) -> tuple[str, str]:
    """Returns (bucket, key) for the rendezvous object."""
    root = os.environ.get("METAFLOW_DATASTORE_SYSROOT_S3", "")
    if root.startswith("s3://"):
        root = root[5:]
    bucket, _, prefix = root.partition("/")
    key = f"{prefix}/{namespace}/{service_id}/endpoint".lstrip("/")
    return bucket, key


def register_service(namespace: str, service_id: str, url: str) -> None:
    payload = json.dumps({"url": url})

    if _is_remote():
        import boto3
        bucket, key = _s3_key(namespace, service_id)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=payload)
        print(f"[metaflow-coordinator] registered {service_id!r} in s3://{bucket}/{key}")
    else:
        path = _local_path(namespace, service_id)
        with open(path, "w") as f:
            f.write(payload)
        print(f"[metaflow-coordinator] registered {service_id!r} at {path}")


def discover_services(
    service_id: str,
    names: list[str],
    namespace: str = "metaflow-svc",
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> dict[str, str]:
    """
    Discover multiple services from a SessionServiceGroup concurrently.

    ``names`` must match the keys passed to ``SessionServiceGroup``.
    Fetches all URLs in parallel and returns a dict mapping name → URL.
    Raises ServiceNotReadyError if any service times out.

    Example::

        SessionServiceGroup({"echo": echo_svc, "tracker": tracker}).run(...)

        # in the worker step:
        urls = discover_services(
            self.coordinator_id,
            names=["echo", "tracker"],
            namespace="echo-example",
            timeout=120,
        )
        self.echo_url    = urls["echo"]
        self.tracker_url = urls["tracker"]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch(name: str) -> tuple[str, str]:
        url = await_service(
            f"{service_id}/{name}",
            namespace=namespace,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        return name, url

    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        futures = {ex.submit(fetch, name): name for name in names}
        results: dict[str, str] = {}
        for fut in as_completed(futures):
            name, url = fut.result()  # re-raises ServiceNotReadyError if any failed
            results[name] = url

    return results


def await_service(
    service_id: str,
    namespace: str = "metaflow-svc",
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> str:
    """
    Poll until the service registers its URL, then return it.
    Raises ServiceNotReadyError if timeout is exceeded.
    """
    from .exceptions import ServiceNotReadyError

    deadline = time.monotonic() + timeout
    attempt = 0

    while time.monotonic() < deadline:
        try:
            url = _read_endpoint(namespace, service_id)
            if url:
                return url
        except Exception:
            pass
        wait = min(poll_interval * (1.5 ** attempt), 15)
        time.sleep(wait)
        attempt += 1

    raise ServiceNotReadyError(
        f"Service {service_id!r} (namespace={namespace!r}) did not register within {timeout}s."
    )


def save_checkpoint(namespace: str, service_id: str, data: dict) -> None:
    """Persist arbitrary JSON so a restarted service can resume."""
    payload = json.dumps(data)
    if _is_remote():
        import boto3
        bucket, key = _s3_key(namespace, service_id)
        chk_key = key.replace("/endpoint", "/checkpoint")
        boto3.client("s3").put_object(Bucket=bucket, Key=chk_key, Body=payload)
    else:
        path = _local_path(namespace, service_id, suffix="-checkpoint")
        with open(path, "w") as f:
            f.write(payload)


def load_checkpoint(namespace: str, service_id: str) -> dict | None:
    """Return previously persisted checkpoint data, or None if none exists."""
    try:
        if _is_remote():
            import boto3
            bucket, key = _s3_key(namespace, service_id)
            chk_key = key.replace("/endpoint", "/checkpoint")
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=chk_key)
            return json.loads(obj["Body"].read())
        else:
            path = _local_path(namespace, service_id, suffix="-checkpoint")
            if not os.path.exists(path):
                return None
            with open(path) as f:
                return json.loads(f.read())
    except Exception:
        return None


def _read_endpoint(namespace: str, service_id: str) -> str | None:
    if _is_remote():
        import boto3
        bucket, key = _s3_key(namespace, service_id)
        try:
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())["url"]
        except Exception:
            return None
    else:
        path = _local_path(namespace, service_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.loads(f.read())["url"]
