"""Durable single-job smoke accounting and an external provider watchdog."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.posttrain.records import atomic_json
from src.posttrain.schema import require

CAP_SECONDS = 24 * 3600
SHUTDOWN_MARGIN = 600
POLL_SECONDS = 15
TERMINAL = {"job_succeeded", "job_failed", "job_stopped", "job_cancelled"}


@contextmanager
def locked_ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {"schema": 1, "cap_seconds": CAP_SECONDS, "jobs": [], "blocked": None}
        )
        require(
            state["schema"] == 1 and state["cap_seconds"] == CAP_SECONDS,
            "budget ledger contract differs",
        )
        yield state
        atomic_json(path, state)


def consumed(state, now):
    return sum(
        max(0, (j["released_at"] or now) - j["allocated_at"])
        for j in state["jobs"]
        if j["allocated_at"] is not None
    )


def reserve(state, name, workspace, now):
    require(not state["blocked"], "provider accounting/control failure blocks launches")
    require(
        not any(j["released_at"] is None for j in state["jobs"]),
        "one smoke job at a time",
    )
    require(
        not any(j["name"] == name for j in state["jobs"]),
        "job name must be unique in ledger",
    )
    require(
        consumed(state, now) + SHUTDOWN_MARGIN < state["cap_seconds"],
        "implementation budget exhausted",
    )
    row = {
        "name": name,
        "workspace": workspace,
        "reserved_at": now,
        "submitted_at": None,
        "allocated_at": None,
        "released_at": None,
        "last_observed_at": now,
        "status": "reserved",
        "stop_requested_at": None,
        "observations": [],
        "errors": [],
    }
    state["jobs"].append(row)
    return row


def reconcile(state, row, observation, now):
    require(now >= row["last_observed_at"], "watchdog clock moved backward")
    if observation["allocated_at"] is not None:
        require(
            row["reserved_at"] - 5 <= observation["allocated_at"] <= now + 5,
            "provider allocation timestamp outside reservation",
        )
        if row["allocated_at"] is None:
            row["allocated_at"] = observation["allocated_at"]
        else:
            require(
                row["allocated_at"] == observation["allocated_at"],
                "provider allocation timestamp changed",
            )
    row["status"] = observation["status"]
    row["last_observed_at"] = now
    row["observations"].append({"at": now, **observation})
    if observation["released"]:
        require(
            row["allocated_at"] is not None,
            "released job lacks allocation history; accounting unresolved",
        )
        row["released_at"] = now
    return (
        row["released_at"] is None
        and consumed(state, now) >= state["cap_seconds"] - SHUTDOWN_MARGIN
    )


class MissingJob(RuntimeError):
    pass


class OwnershipError(RuntimeError):
    pass


class InspireProvider:
    def _call(self, *args):
        completed = subprocess.run(
            ["inspire", "--json", *args],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
            env={**os.environ, "TZ": "UTC"},
        )
        # The CLI writes successful JSON to stdout and error JSON to stderr.
        stream = completed.stdout if completed.returncode == 0 else completed.stderr
        try:
            payload = json.loads(stream)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"provider returned non-JSON (exit {completed.returncode}): {stream[-2000:]}"
            ) from exc
        if payload.get("error", {}).get("type") == "JobNotFound":
            raise MissingJob("named job is not visible")
        require(
            completed.returncode == 0 and payload.get("success"),
            f"provider command failed: {payload.get('error')}",
        )
        return payload["data"]

    def owned_status(self, row):
        data = self._call("job", "status", row["name"], "--workspace", row["workspace"])
        if float(data["created_at"]) / 1000 < row["reserved_at"] - 2:
            raise OwnershipError("named remote job predates this reservation")
        return data

    def observe(self, row):
        name, workspace = row["name"], row["workspace"]
        data = self.owned_status(row)
        resource = data["resource"]
        require(
            resource["gpu"] == 16 and resource["nodes"] == 1,
            "smoke allocation must be one 16-NPU node",
        )
        instances = self._call("job", "instances", name, "--workspace", workspace)[
            "items"
        ]
        allocated = row["allocated_at"]
        terminal = data["status"] in TERMINAL and not data.get("nodes")
        events = []
        if allocated is None or terminal:
            events = self._call(
                "job", "events", name, "--workspace", workspace, "--tail", "200"
            )["items"]
        if allocated is None:
            scheduled = [
                event_time(e) for e in events if e.get("reason") == "Scheduled"
            ]
            if scheduled:
                allocated = min(scheduled)
            elif data.get("nodes") or any(i.get("node") for i in instances):
                raise ValueError("allocated node lacks a provider Scheduled timestamp")
        # Instances retain their historical node after a pod is deleted. A
        # terminal job alone cannot distinguish that history from retention;
        # require provider deletion evidence newer than its last creation.
        created = [
            event_time(e) for e in events if e.get("reason") == "SuccessfulCreatePod"
        ]
        deleted = [
            event_time(e) for e in events if e.get("reason") == "SuccessfulDeletePod"
        ]
        cleanup = max(deleted) if deleted else None
        deleted_latest = bool(
            cleanup is not None
            and allocated is not None
            and cleanup > max([allocated, *created])
        )
        released = terminal and (
            not any(i.get("node") for i in instances) or deleted_latest
        )
        return {
            "status": data["status"],
            "allocated_at": allocated,
            "released": released,
            "nodes": data.get("nodes", []),
            "instances": instances,
            "pod_deleted_at": cleanup,
        }

    def stop(self, row):
        self.owned_status(row)
        return self._call("job", "stop", row["name"], "--workspace", row["workspace"])


def event_time(event):
    return (
        datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=ZoneInfo("UTC"))
        .timestamp()
    )


def watch(path, job, provider=None, clock=time.time, sleep=time.sleep):
    provider = provider or InspireProvider()
    path = Path(path)
    # A second watchdog must not race control actions or overwrite observations.
    with path.with_suffix(".watch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while True:
            with locked_ledger(path) as state:
                row = next(j for j in state["jobs"] if j["name"] == job)
                if row["released_at"] is not None:
                    return
                if row["submitted_at"] is None:
                    # The watchdog is alive before submission; reservation prevents another launcher.
                    row["watchdog_ready_at"] = clock()
                    submitted = False
                else:
                    submitted = True
                snapshot = dict(row)
            if not submitted:
                sleep(1)
                continue
            try:
                observation = provider.observe(snapshot)
                with locked_ledger(path) as state:
                    row = next(j for j in state["jobs"] if j["name"] == job)
                    stop = reconcile(state, row, observation, clock()) or bool(
                        state["blocked"]
                    )
                    if row["released_at"] is not None:
                        return
                failures = 0
            except OwnershipError:
                with locked_ledger(path) as state:
                    state["blocked"] = "remote job ownership mismatch; no stop sent"
                return
            except MissingJob:
                if clock() - snapshot["submitted_at"] < 120:
                    sleep(1)
                    continue
                with locked_ledger(path) as state:
                    state["blocked"] = (
                        "submitted job missing; reconcile submission before another launch"
                    )
                    next(j for j in state["jobs"] if j["name"] == job)["errors"].append(
                        {"at": clock(), "error": "submitted job missing"}
                    )
                sleep(POLL_SECONDS)
                continue
            except Exception as exc:  # noqa: BLE001 - retain uncertainty and fail closed
                failures += 1
                with locked_ledger(path) as state:
                    row = next(j for j in state["jobs"] if j["name"] == job)
                    row["errors"].append(
                        {"at": clock(), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    state["blocked"] = (
                        "provider observation failed; reconcile before another launch"
                    )
                stop = True
            if stop:
                try:
                    provider.stop(snapshot)
                    with locked_ledger(path) as state:
                        row = next(j for j in state["jobs"] if j["name"] == job)
                        row["stop_requested_at"] = clock()
                except Exception as exc:  # noqa: BLE001 - never infer successful release
                    with locked_ledger(path) as state:
                        state["blocked"] = f"provider stop failed: {exc}"
                        next(j for j in state["jobs"] if j["name"] == job)[
                            "errors"
                        ].append({"at": clock(), "error": str(exc)})
            sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        if not args.job:
            parser.error("--watch requires --job")
        watch(args.ledger, args.job)
    else:
        with locked_ledger(args.ledger) as state:
            print(
                json.dumps(
                    {
                        **state,
                        "consumed_node_hours": consumed(state, time.time()) / 3600,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
