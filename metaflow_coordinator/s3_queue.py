"""
s3_queue.py

S3-backed distributed task queue for metaflow-gha workers.

Layout under {bucket}/{prefix}/gha-queue/{run_id}/:
  tasks/{task_id}.json       — task definition (written once on push)
  ready/{step_name}/{task_id} — task is claimable (empty body, sentinel key)
  claimed/{task_id}          — task is being worked on (JSON: worker_id, claimed_at)
  done/{task_id}             — task completed successfully (JSON: completed_at)
  failed/{task_id}           — task failed permanently (JSON: error, attempts)
  waiting/{task_id}          — task blocked on parent(s) (JSON: parent_task_ids)

Atomic claiming uses S3 conditional writes (IfNoneMatch: "*"), available since
boto3 >= 1.35.2 (August 2024). Competing workers receive a 412 or 409 and retry.

Task definition schema (tasks/{task_id}.json):
  {
    "task_id":        str,            # unique per task attempt
    "run_id":         str,
    "step_name":      str,
    "pathspec":       str,            # FlowName/run_id/step_name/task_id
    "input_paths":    list[str],      # metaflow input pathspecs for join steps
    "parent_task_ids": list[str],     # task_ids that must be done before this runs
    "attempt":        int,            # 0-based retry count
    "max_retries":    int,
    "timeout_seconds": int,
    "env_id":         str,            # hash of requirements for GHA cache key
    "code_package_url": str,         # s3:// URL of flow code package
    "flow_name":      str,
    "metaflow_args":  list[str],      # extra args for `python flow.py step ...`
  }
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# S3 path helpers
# ---------------------------------------------------------------------------

def _s3_root(bucket: str, prefix: str, run_id: str) -> str:
    """Base prefix for all queue keys for this run."""
    parts = [p.strip("/") for p in [prefix, "gha-queue", run_id] if p]
    return "/".join(parts)


def _task_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/tasks/{task_id}.json"


def _ready_key(bucket: str, prefix: str, run_id: str, step_name: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/ready/{step_name}/{task_id}"


def _claimed_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/claimed/{task_id}"


def _done_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/done/{task_id}"


def _failed_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/failed/{task_id}"


def _waiting_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/waiting/{task_id}"


def _workers_dispatched_key(bucket: str, prefix: str, run_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/workers_dispatched"


def _log_key(bucket: str, prefix: str, run_id: str, task_id: str) -> str:
    return f"{_s3_root(bucket, prefix, run_id)}/logs/{task_id}"


def _list_all_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    """Paginate through all keys under prefix, handling >1000 item lists."""
    keys: list[str] = []
    kwargs: dict = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return keys


def _bucket_prefix_from_env() -> tuple[str, str]:
    """Parse bucket and prefix from METAFLOW_DATASTORE_SYSROOT_S3."""
    root = os.environ.get("METAFLOW_DATASTORE_SYSROOT_S3", "")
    if root.startswith("s3://"):
        root = root[5:]
    bucket, _, prefix = root.partition("/")
    return bucket, prefix


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def push_task(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    task: dict,
) -> None:
    """
    Write task definition and enqueue it.

    If task["parent_task_ids"] is non-empty, the task goes to waiting/ until
    all parents complete. Otherwise it goes directly to ready/.
    """
    task_id = task["task_id"]

    # Write task definition
    s3.put_object(
        Bucket=bucket,
        Key=_task_key(bucket, prefix, run_id, task_id),
        Body=json.dumps(task),
        ContentType="application/json",
    )

    parent_ids = task.get("parent_task_ids") or []
    if parent_ids:
        # Check if all parents are already done
        pending_parents = [
            pid for pid in parent_ids
            if not _is_done(s3, bucket, prefix, run_id, pid)
        ]
        if pending_parents:
            s3.put_object(
                Bucket=bucket,
                Key=_waiting_key(bucket, prefix, run_id, task_id),
                Body=json.dumps({"parent_task_ids": pending_parents}),
                ContentType="application/json",
            )
            return

    _enqueue_ready(s3, bucket, prefix, run_id, task["step_name"], task_id)


def claim_task(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    worker_id: str,
    preferred_step: str | None = None,
) -> dict | None:
    """
    Atomically claim a ready task and return its definition, or None if queue empty.

    Step-affine: if preferred_step is given, tries tasks at that step first to
    avoid environment switching overhead.
    """
    ready_prefix = f"{_s3_root(bucket, prefix, run_id)}/ready/"

    def _candidates_for_step(step: str) -> list[str]:
        return _list_all_keys(
            s3, bucket, f"{_s3_root(bucket, prefix, run_id)}/ready/{step}/"
        )

    def _all_candidates() -> list[str]:
        return _list_all_keys(s3, bucket, ready_prefix)

    # Build candidate list: preferred step first
    candidates: list[str] = []
    if preferred_step:
        candidates = _candidates_for_step(preferred_step)
    if not candidates:
        candidates = _all_candidates()

    for ready_key in candidates:
        # Extract task_id from key: ready/{step_name}/{task_id}
        task_id = ready_key.rsplit("/", 1)[-1]

        # Attempt atomic claim: write claimed/ only if it doesn't exist
        claimed_key = _claimed_key(bucket, prefix, run_id, task_id)
        try:
            s3.put_object(
                Bucket=bucket,
                Key=claimed_key,
                Body=json.dumps({
                    "worker_id": worker_id,
                    "claimed_at": time.time(),
                }),
                ContentType="application/json",
                IfNoneMatch="*",  # atomic: fail if key already exists
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("412", "PreconditionFailed", "ConditionalRequestConflict"):
                # Another worker claimed this task — try next candidate
                continue
            raise

        # Claimed successfully — delete from ready/
        s3.delete_object(Bucket=bucket, Key=ready_key)

        # Load and return task definition
        obj = s3.get_object(
            Bucket=bucket,
            Key=_task_key(bucket, prefix, run_id, task_id),
        )
        return json.loads(obj["Body"].read())

    return None  # queue empty


def complete_task(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    task_id: str,
) -> None:
    """
    Mark task as done and unblock any waiting tasks whose last parent this was.
    """
    s3.put_object(
        Bucket=bucket,
        Key=_done_key(bucket, prefix, run_id, task_id),
        Body=json.dumps({"completed_at": time.time()}),
        ContentType="application/json",
    )
    # Remove claimed marker
    try:
        s3.delete_object(Bucket=bucket, Key=_claimed_key(bucket, prefix, run_id, task_id))
    except Exception:
        pass

    _unblock_waiting(s3, bucket, prefix, run_id, completed_task_id=task_id)


def fail_task(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    task_id: str,
    error: str,
    attempt: int,
    max_retries: int,
) -> None:
    """
    Re-queue the task for retry, or move it to failed/ if retries exhausted.
    """
    try:
        s3.delete_object(Bucket=bucket, Key=_claimed_key(bucket, prefix, run_id, task_id))
    except Exception:
        pass

    if attempt < max_retries:
        # Reload task definition, bump attempt, re-enqueue
        obj = s3.get_object(
            Bucket=bucket,
            Key=_task_key(bucket, prefix, run_id, task_id),
        )
        task = json.loads(obj["Body"].read())
        task["attempt"] = attempt + 1

        # Generate a new task_id for the retry to avoid stale claimed markers
        new_task_id = f"{task_id}-retry{attempt + 1}"
        task["task_id"] = new_task_id

        s3.put_object(
            Bucket=bucket,
            Key=_task_key(bucket, prefix, run_id, new_task_id),
            Body=json.dumps(task),
            ContentType="application/json",
        )
        _enqueue_ready(s3, bucket, prefix, run_id, task["step_name"], new_task_id)
    else:
        s3.put_object(
            Bucket=bucket,
            Key=_failed_key(bucket, prefix, run_id, task_id),
            Body=json.dumps({"error": error, "attempts": attempt + 1}),
            ContentType="application/json",
        )


def reclaim_stale(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    stale_after_seconds: int = 3600,
) -> int:
    """
    Scan claimed/ for tasks that have been claimed longer than stale_after_seconds
    and move them back to ready/. Returns number of tasks reclaimed.

    Called periodically by workers (e.g., every 5 minutes).
    """
    claimed_prefix = f"{_s3_root(bucket, prefix, run_id)}/claimed/"
    now = time.time()
    reclaimed = 0

    for key in _list_all_keys(s3, bucket, claimed_prefix):
        task_id = key.rsplit("/", 1)[-1]

        try:
            claim_obj = s3.get_object(Bucket=bucket, Key=key)
            claim_data = json.loads(claim_obj["Body"].read())
        except Exception:
            continue

        age = now - claim_data.get("claimed_at", now)
        if age < stale_after_seconds:
            continue

        # Load task definition to get step_name
        try:
            task_obj = s3.get_object(
                Bucket=bucket,
                Key=_task_key(bucket, prefix, run_id, task_id),
            )
            task = json.loads(task_obj["Body"].read())
        except Exception:
            continue

        # Move back to ready (ignore errors — another worker may have just finished it)
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            _enqueue_ready(s3, bucket, prefix, run_id, task["step_name"], task_id)
            reclaimed += 1
        except Exception:
            pass

    return reclaimed


def list_pending(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
) -> dict[str, list[str]]:
    """
    Returns a summary dict with keys 'ready', 'claimed', 'waiting', 'done', 'failed',
    each containing a list of task_ids in that state.
    """
    root = _s3_root(bucket, prefix, run_id)
    result: dict[str, list[str]] = {
        "ready": [], "claimed": [], "waiting": [], "done": [], "failed": []
    }

    state_map = {
        "ready": f"{root}/ready/",
        "claimed": f"{root}/claimed/",
        "waiting": f"{root}/waiting/",
        "done": f"{root}/done/",
        "failed": f"{root}/failed/",
    }

    for state, pfx in state_map.items():
        for key in _list_all_keys(s3, bucket, pfx):
            if state == "ready":
                task_id = key.split("/ready/", 1)[1].split("/", 1)[-1]
            else:
                task_id = key.rsplit("/", 1)[-1]
            result[state].append(task_id)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def mark_workers_dispatched(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    n_workers: int,
) -> bool:
    """
    Atomically mark that workers have been dispatched for this run.
    Returns True if this call was first (caller should dispatch workers).
    Returns False if workers were already dispatched by a prior call.
    """
    key = _workers_dispatched_key(bucket, prefix, run_id)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps({"n_workers": n_workers, "dispatched_at": time.time()}),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("412", "PreconditionFailed", "ConditionalRequestConflict"):
            return False
        raise


def write_task_log(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    task_id: str,
    content: str,
) -> None:
    """Replace the S3 log object for this task with the current full log content."""
    s3.put_object(
        Bucket=bucket,
        Key=_log_key(bucket, prefix, run_id, task_id),
        Body=content.encode("utf-8", errors="replace"),
        ContentType="text/plain",
    )


def read_task_log(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    task_id: str,
) -> str | None:
    """Read the current log content for a task, or None if not yet written."""
    try:
        obj = s3.get_object(
            Bucket=bucket,
            Key=_log_key(bucket, prefix, run_id, task_id),
        )
        return obj["Body"].read().decode("utf-8", errors="replace")
    except ClientError:
        return None


def _enqueue_ready(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    step_name: str,
    task_id: str,
) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=_ready_key(bucket, prefix, run_id, step_name, task_id),
        Body=b"",
    )


def _is_done(s3: Any, bucket: str, prefix: str, run_id: str, task_id: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=_done_key(bucket, prefix, run_id, task_id))
        return True
    except Exception:
        return False


def _unblock_waiting(
    s3: Any,
    bucket: str,
    prefix: str,
    run_id: str,
    completed_task_id: str,
) -> None:
    """
    Scan waiting/ tasks and unblock any whose parent set is now fully done.
    """
    waiting_prefix = f"{_s3_root(bucket, prefix, run_id)}/waiting/"

    for key in _list_all_keys(s3, bucket, waiting_prefix):
        task_id = key.rsplit("/", 1)[-1]

        try:
            wait_obj = s3.get_object(Bucket=bucket, Key=key)
            wait_data = json.loads(wait_obj["Body"].read())
        except Exception:
            continue

        parents = wait_data.get("parent_task_ids", [])
        if completed_task_id not in parents:
            continue

        remaining = [p for p in parents if not _is_done(s3, bucket, prefix, run_id, p)]

        if not remaining:
            # All parents done — move to ready
            try:
                task_obj = s3.get_object(
                    Bucket=bucket,
                    Key=_task_key(bucket, prefix, run_id, task_id),
                )
                task = json.loads(task_obj["Body"].read())
                s3.delete_object(Bucket=bucket, Key=key)
                _enqueue_ready(s3, bucket, prefix, run_id, task["step_name"], task_id)
            except Exception:
                pass
        elif len(remaining) < len(parents):
            # Update waiting entry with reduced parent set
            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps({"parent_task_ids": remaining}),
                    ContentType="application/json",
                )
            except Exception:
                pass
