"""Join qualification receipts to actual benchmark, replay and allocation data."""

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from src.evaluation.benchmarks.run import parse_score
from src.posttrain.config import Scorer
from src.posttrain.records import file_hash, identity
from src.posttrain.schema import decode, require
from src.posttrain.scoring.client import request_for_record
from src.posttrain.scoring.contracts import validate_results


def benchmark(read, names, prefix, *, prompts, source, image_witness):
    """Validate every declared sample/channel; summaries cannot replace envelopes."""
    manifest = read(prefix + "manifest.json")
    report = read(prefix + "report.json")
    require(
        manifest["code"] == source
        and manifest["prompts"] == prompts
        and not manifest["visual_only"],
        "benchmark source/prompt/scoring identity differs",
    )
    config = manifest["config"]
    arms = {a["name"]: a for a in config["arms"]}
    require(bool(arms) and set(report["arms"]) == set(arms), "benchmark arms differ")
    specs = [decode(Scorer, s) for s in config["scorers"]]
    require(bool(specs), "benchmark needs actual scorers")
    expected = {
        identity(
            {
                "policy": manifest["policies"][a],
                "prompt": p,
                "arm": arm,
                "sampling": config["sampling"],
            }
        ): (a, p)
        for a, arm in arms.items()
        for p in prompts
    }
    record_names = {n for n in names if n.startswith(prefix + "records/")}
    require(
        record_names == {prefix + f"records/{s}.json" for s in expected},
        "incomplete or unexpected benchmark records",
    )
    required_scores = {
        prefix + f"scores/{spec.id}/{sample}.json"
        for spec in specs
        for sample in expected
    }
    require(
        {n for n in names if n.startswith(prefix + "scores/")} == required_scores,
        "incomplete or unexpected score envelopes",
    )
    values = {}
    for sample, (arm, prompt) in expected.items():
        name = prefix + f"records/{sample}.json"
        record = read(name)
        require(
            record["sample_id"] == sample
            and record["arm"] == arm
            and record["prompt"] == prompt
            and record["checkpoint"] == arms[arm]["checkpoint"],
            "benchmark record identity differs",
        )
        for kind in ("image", "latent"):
            path = Path(record[kind + "_path"])
            if path.exists():
                require(
                    file_hash(path) == record[kind + "_sha256"],
                    "benchmark artifact bytes differ",
                )
            else:
                require(image_witness(name, record, kind), "missing artifact witness")
        for spec in specs:
            row = read(prefix + f"scores/{spec.id}/{sample}.json")
            require(
                row["scorer"] == spec.id
                and type(row["attempts"]) is int
                and 1 <= row["attempts"] <= spec.attempts,
                "score service/attempt identity differs",
            )
            # Transport is not semantic identity. Construct the same request
            # from a reference without requiring inaccessible remote PNG bytes.
            request = request_for_record(record, replace(spec, transport="shared_file"))
            validate_results([request], [parse_score(row)], spec.outputs, complete=True)
            if spec.structured_fields:
                diagnostic = row["metadata"].get("rubric_diagnostic")
                require(bool(diagnostic), "missing structured rubric diagnostic")
                diagnostic = (
                    json.loads(diagnostic)
                    if isinstance(diagnostic, str)
                    else diagnostic
                )
                require(
                    isinstance(diagnostic, dict)
                    and diagnostic.get("rubric") == request.structured.get("rubric"),
                    "rubric diagnostic inputs differ",
                )
            for channel in config["evaluation"]["channels"]:
                role, metric = channel.split(".")
                if role == spec.id:
                    values.setdefault((arm, channel), []).append(row["values"][metric])
    for arm, metrics in report["arms"].items():
        require(
            set(metrics) == set(config["evaluation"]["channels"]),
            "benchmark metric declarations differ",
        )
        for channel, metric in metrics.items():
            actual = values[(arm, channel)]
            require(
                metric["complete"]
                and metric["expected"] == metric["scored"] == len(prompts)
                and not metric["failed_or_missing"]
                and metric["mean_on_successes"] == statistics.mean(actual),
                "benchmark summary differs from actual scores",
            )
    return {
        "arms": arms,
        "images": len(expected),
        "score_envelopes": len(required_scores),
    }


def replay(read, names, prefix, *, start, finish, ranks, record_witness=None):
    """Select the committed attempt and retain all semantic replay fields."""
    attempts = {}
    for name in names:
        if name.startswith(prefix + "completion/") and name.endswith(".json"):
            row = read(name)
            if row["step"] == finish:
                attempt, rank = Path(name).stem.rsplit("-rank-", 1)
                attempts.setdefault(attempt, set()).add(int(rank))
    require(
        len(attempts) == 1 and next(iter(attempts.values())) == set(range(ranks)),
        "missing unique committed replay attempt",
    )
    attempt = next(iter(attempts))
    collections, updates = {}, {}
    for folder, result in [("collections", collections), ("updates", updates)]:
        for name in names:
            if name.startswith(prefix + folder + "/" + attempt + "-round-"):
                row = read(name)
                if start <= row.get("step", -1) - (folder == "updates") < finish:
                    if record_witness is not None:
                        require(
                            record_witness(name, row),
                            "replay record differs from pinned target witness",
                        )
                    key = Path(name).name.removeprefix(attempt)
                    if folder == "collections":
                        row = {
                            **row,
                            "records": [
                                {
                                    k: v
                                    for k, v in r.items()
                                    if k
                                    not in {
                                        "image_path",
                                        "latent_path",
                                        "latent_sha256",
                                    }
                                }
                                for r in row["records"]
                            ],
                        }
                    else:
                        row = {
                            k: v
                            for k, v in row.items()
                            if k not in {"seconds", "peak_allocated_bytes", "attempt"}
                        }
                    result[key] = row
    require(bool(collections) and bool(updates), "missing replay records")
    return collections, updates


def active_allocation(state, job, now, *, cap_seconds, max_age):
    """Require fresh watchdog evidence for the sole live owned allocation."""
    require(
        state["schema"] == 1 and state["cap_seconds"] == cap_seconds,
        "budget ledger contract differs",
    )
    require(not state["blocked"], "allocation accounting is blocked")
    jobs = [j for j in state["jobs"] if j["name"] == job]
    require(
        len(jobs) == 1
        and job.startswith("inko-posttrain-")
        and jobs[0]["allocated_at"] is not None
        and jobs[0]["released_at"] is None
        and not jobs[0]["stop_requested_at"]
        and [j["name"] for j in state["jobs"] if j["released_at"] is None] == [job],
        "qualification requires the sole active allocation",
    )
    row = jobs[0]
    require(
        type(now) in {int, float}
        and math.isfinite(now)
        and row["allocated_at"] <= row["last_observed_at"] <= now
        and now - row["last_observed_at"] <= max_age
        and bool(row["observations"])
        and row["observations"][-1]["at"] == row["last_observed_at"]
        and row["observations"][-1]["allocated_at"] == row["allocated_at"]
        and not row["observations"][-1]["released"],
        "missing fresh active-allocation observation",
    )
    return row


def allocation(
    state,
    receipt,
    execution,
    results,
    matrix_sha256,
    *,
    start,
    cap_seconds,
    max_age,
):
    require(
        state["schema"] == 1 and state["cap_seconds"] == cap_seconds,
        "budget ledger contract differs",
    )
    require(
        receipt["execution_sha256"] == identity(execution)
        and receipt["results_sha256"] == identity(results)
        and execution["matrix_sha256"] == matrix_sha256
        and receipt["start_sha256"] == identity(start)
        and start["job"] == receipt["job"],
        "allocation execution identity differs",
    )
    proof = active_allocation(
        start["ledger"],
        start["job"],
        start["started_at"],
        cap_seconds=cap_seconds,
        max_age=max_age,
    )
    jobs = [j for j in state["jobs"] if j["name"] == receipt["job"]]
    require(
        len(jobs) == 1
        and receipt["job"].startswith("inko-posttrain-")
        and jobs[0]["allocated_at"] is not None
        and jobs[0]["released_at"] is not None
        and jobs[0]["released_at"] >= jobs[0]["allocated_at"],
        "qualification has no matching released allocation",
    )
    row = jobs[0]
    require(
        all(row[k] == proof[k] for k in ("allocated_at", "reserved_at", "workspace"))
        and type(receipt["finished_at"]) in {int, float}
        and math.isfinite(receipt["finished_at"])
        and row["allocated_at"]
        <= start["started_at"]
        <= receipt["finished_at"]
        <= row["released_at"],
        "execution is outside its owned allocation interval",
    )
    return receipt["job"]


def compare_replay(first, second, *, updates, samples):
    require(first == second, "boundary replay records differ")
    require(
        len(first[1]) == updates
        and sum(len(v["records"]) for v in first[0].values()) == samples,
        "boundary replay coverage incomplete",
    )
