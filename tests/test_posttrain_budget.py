import json

import pytest

from scripts.posttrain.budget import (
    CAP_SECONDS,
    SHUTDOWN_MARGIN,
    consumed,
    locked_ledger,
    reconcile,
    reserve,
    watch,
)


def observation(status, start, released=False):
    return {
        "status": status,
        "allocated_at": start,
        "released": released,
        "nodes": [] if released else ["node"],
        "instances": [],
    }


def test_queue_startup_retention_failure_and_restart(tmp_path):
    path = tmp_path / "budget.json"
    with locked_ledger(path) as state:
        row = reserve(state, "one", "workspace", 0)
        reconcile(state, row, observation("queued", None), 50)
        assert consumed(state, 50) == 0
        reconcile(state, row, observation("starting", 60), 65)
        assert consumed(state, 65) == 5
        reconcile(state, row, observation("failed_retaining", 60), 120)
        assert consumed(state, 120) == 60
        with pytest.raises(ValueError, match="one smoke"):
            reserve(state, "two", "workspace", 130)
    with locked_ledger(path) as restored:
        assert consumed(restored, 150) == 90
        reconcile(restored, restored["jobs"][0], observation("failed", 60, True), 180)
        assert consumed(restored, 200) == 120
        reserve(restored, "two", "workspace", 200)


def test_external_watchdog_survives_lost_workload_and_stops_at_shared_cap(tmp_path):
    path = tmp_path / "budget.json"
    with locked_ledger(path) as state:
        previous = reserve(state, "previous", "workspace", 0)
        reconcile(state, previous, observation("done", 0, True), 80000)
        current = reserve(state, "current", "workspace", 80000)
        current["submitted_at"] = 80000
    now = [CAP_SECONDS - SHUTDOWN_MARGIN]

    class Provider:
        stopped = False

        def observe(self, row):
            # No learner process is consulted; an allocated crashed/retained job still costs time.
            return observation(
                "failed_retaining" if not self.stopped else "stopped",
                80000,
                self.stopped,
            )

        def stop(self, row):
            assert row["name"] == "current"
            self.stopped = True

    provider = Provider()
    watch(
        path,
        "current",
        provider,
        clock=lambda: now[0],
        sleep=lambda n: now.__setitem__(0, now[0] + n),
    )
    assert provider.stopped
    state = json.loads(path.read_text())
    assert state["jobs"][-1]["released_at"] == CAP_SECONDS - SHUTDOWN_MARGIN + 15
    assert consumed(state, now[0]) == CAP_SECONDS - SHUTDOWN_MARGIN + 15


def test_provider_failure_stops_and_blocks_next_launch(tmp_path):
    path = tmp_path / "budget.json"
    with locked_ledger(path) as state:
        row = reserve(state, "one", "workspace", 10)
        row["submitted_at"] = 10
    now = [20]

    class Provider:
        stopped = False

        def observe(self, row):
            if not self.stopped:
                raise RuntimeError("provider unavailable")
            return observation("stopped", 15, True)

        def stop(self, row):
            self.stopped = True

    watch(
        path,
        "one",
        Provider(),
        clock=lambda: now[0],
        sleep=lambda n: now.__setitem__(0, now[0] + n),
    )
    with locked_ledger(path) as state:
        assert state["blocked"] and state["jobs"][0]["released_at"] == 35
        with pytest.raises(ValueError, match="blocks launches"):
            reserve(state, "two", "workspace", 36)


def test_submission_timeout_keeps_observable_intent(tmp_path, monkeypatch):
    import subprocess

    from scripts.posttrain import qualify

    path = tmp_path / "budget.json"

    class Watchdog:
        def __init__(self, *args, **kwargs):
            with locked_ledger(path) as state:
                state["jobs"][-1]["watchdog_ready_at"] = 1

        def poll(self):
            return None

    def uncertain_create(*args, **kwargs):
        state = json.loads(path.read_text())
        assert state["jobs"][-1]["submitted_at"] is not None
        raise subprocess.TimeoutExpired("create", 120)

    monkeypatch.setattr(qualify.subprocess, "Popen", Watchdog)
    monkeypatch.setattr(qualify.subprocess, "run", uncertain_create)
    job = {
        k: "fixture"
        for k in (
            "name",
            "workspace",
            "project",
            "group",
            "quota",
            "image",
            "command",
            "shm_size",
        )
    }
    with pytest.raises(subprocess.TimeoutExpired):
        qualify.launch({"job": job}, path)
    assert json.loads(path.read_text())["jobs"][-1]["submitted_at"] is not None


def test_delayed_provider_visibility_keeps_watchdog_alive(tmp_path):
    from scripts.posttrain.budget import MissingJob

    path = tmp_path / "budget.json"
    with locked_ledger(path) as state:
        row = reserve(state, "one", "workspace", 0)
        row["submitted_at"] = 0
    now = [121]

    class Provider:
        calls = 0
        stopped = False

        def observe(self, row):
            self.calls += 1
            if self.calls == 1:
                raise MissingJob("late")
            now[0] += 45  # provider latency is allocated time too
            return observation(
                "stopped" if self.stopped else "running", 10, self.stopped
            )

        def stop(self, row):
            now[0] += 45
            self.stopped = True

    provider = Provider()
    watch(
        path,
        "one",
        provider,
        clock=lambda: now[0],
        sleep=lambda n: now.__setitem__(0, now[0] + n),
    )
    state = json.loads(path.read_text())
    assert provider.calls == 3 and provider.stopped and state["blocked"]
    assert consumed(state, now[0]) == 276


def test_watchdog_never_stops_an_unowned_preexisting_job(tmp_path):
    from scripts.posttrain.budget import InspireProvider

    path = tmp_path / "budget.json"
    with locked_ledger(path) as state:
        row = reserve(state, "collision", "workspace", 100)
        row["submitted_at"] = 100

    class Provider(InspireProvider):
        def _call(self, *args):
            assert args[1] != "stop", "an unowned job must never receive stop"
            return {"created_at": "1000"}

    watch(path, "collision", Provider(), clock=lambda: 110)
    state = json.loads(path.read_text())
    assert state["blocked"] == "remote job ownership mismatch; no stop sent"
    assert state["jobs"][0]["stop_requested_at"] is None


def test_missing_allocation_history_cannot_be_charged_as_zero():
    state = {"schema": 1, "cap_seconds": CAP_SECONDS, "jobs": [], "blocked": None}
    row = reserve(state, "one", "workspace", 0)
    row["submitted_at"] = 1
    with pytest.raises(ValueError, match="allocation history"):
        reconcile(state, row, observation("failed", None, True), 1000)
    assert row["released_at"] is None


def test_provider_forces_utc_for_event_formatting(monkeypatch):
    import os
    from types import SimpleNamespace

    from scripts.posttrain.budget import InspireProvider

    monkeypatch.setenv("TZ", "Asia/Shanghai")

    def call(argv, **kwargs):
        assert kwargs["env"]["TZ"] == "UTC"
        return SimpleNamespace(returncode=0, stdout='{"success":true,"data":{}}')

    monkeypatch.setattr("scripts.posttrain.budget.subprocess.run", call)
    assert InspireProvider()._call("job", "events") == {}
    assert os.environ["TZ"] == "Asia/Shanghai"


def test_missing_job_error_is_read_from_stderr(monkeypatch):
    from types import SimpleNamespace

    from scripts.posttrain.budget import InspireProvider, MissingJob

    monkeypatch.setattr(
        "scripts.posttrain.budget.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=16,
            stdout="",
            stderr='{"success":false,"error":{"type":"JobNotFound"}}',
        ),
    )
    with pytest.raises(MissingJob):
        InspireProvider()._call("job", "status")


def test_release_requires_deleted_latest_pod_when_instance_node_is_historical():
    from scripts.posttrain.budget import InspireProvider

    events = [{"time": "2026-10-07 12:40:04", "reason": "SuccessfulCreatePod"}]

    class Provider(InspireProvider):
        def _call(self, *args):
            if args[1] == "status":
                return {
                    "created_at": "1791376802000",
                    "status": "job_stopped",
                    "resource": {"gpu": 16, "nodes": 1},
                }
            if args[1] == "instances":
                return {"items": [{"status": "instance_failed", "node": "historical"}]}
            return {"items": events}

    row = {
        "name": "one",
        "workspace": "workspace",
        "reserved_at": 1791376794,
        "allocated_at": 1791376805,
    }
    assert not Provider().observe(row)["released"]
    events.append({"time": "2026-10-07 12:40:14", "reason": "SuccessfulDeletePod"})
    result = Provider().observe(row)
    assert result["released"] and result["pod_deleted_at"] == 1791376814
    events.append({"time": "2026-10-07 12:40:14", "reason": "SuccessfulCreatePod"})
    assert not Provider().observe(row)["released"]
    events.append({"time": "2026-10-07 12:40:20", "reason": "SuccessfulCreatePod"})
    assert not Provider().observe(row)["released"]
