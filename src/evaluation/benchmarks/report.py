"""Raw scorer reports with paired, scene-clustered uncertainty and coverage."""

import random
import statistics
from collections import defaultdict


def paired_interval(rows, repeats, confidence, seed):
    groups = defaultdict(list)
    for scene, difference in rows:
        groups[scene].append(difference)
    if len(groups) < 2:
        return None
    clusters = list(groups.values())
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        draw = [
            value
            for cluster in rng.choices(clusters, k=len(clusters))
            for value in cluster
        ]
        means.append(statistics.mean(draw))
    means.sort()
    tail = (1 - confidence) / 2
    return [means[int(tail * (repeats - 1))], means[int((1 - tail) * (repeats - 1))]]


def build_report(records, scores, config):
    samples = {r["sample_id"]: r for r in records}
    arms = [a["name"] for a in config["arms"]]
    indexed = {}
    for score in scores:
        key = (score["sample_id"], score["scorer"])
        if key in indexed or key[0] not in samples:
            raise ValueError("duplicate or unexpected score in report")
        indexed[key] = score
    report = {
        "scope": "Development prompt evaluation; scorer metrics are not independent reward-model validation.",
        "arms": {},
        "paired": {},
        "uncertainty": "Paired cluster bootstrap by scene; conditional on these fixed noise seeds.",
    }
    for scorer, spec in config["scorers"].items():
        metric = spec["primary_output"]
        by_arm = {}
        for arm in arms:
            selected = [r for r in records if r["arm"] == arm]
            good, failures = {}, []
            for record in selected:
                result = indexed.get((record["sample_id"], scorer))
                if result is None or result["status"] != "ok":
                    failures.append(record["prompt"]["id"])
                else:
                    good[record["prompt"]["id"]] = (
                        record["prompt"]["scene_id"],
                        result["values"][metric],
                    )
            by_arm[arm] = good
            report["arms"].setdefault(arm, {})[scorer] = {
                "metric": metric,
                "expected": len(selected),
                "scored": len(good),
                "failed_or_missing": failures,
                "mean_on_successes": statistics.mean(v for _, v in good.values())
                if good
                else None,
                "complete": len(good) == len(selected),
            }
        reference = arms[0]
        for arm in arms[1:]:
            common = sorted(by_arm[reference].keys() & by_arm[arm].keys())
            rows = [
                (by_arm[reference][p][0], by_arm[arm][p][1] - by_arm[reference][p][1])
                for p in common
            ]
            expected = sum(r["arm"] == reference for r in records)
            settings = config["evaluation"]
            report["paired"].setdefault(arm, {})[scorer] = {
                "reference": reference,
                "paired": len(rows),
                "expected": expected,
                "mean_delta": statistics.mean(v for _, v in rows) if rows else None,
                "confidence": settings["confidence"],
                "interval": paired_interval(
                    rows,
                    settings["bootstrap_repeats"],
                    settings["confidence"],
                    settings["bootstrap_seed"],
                ),
                "coverage_complete": len(rows) == expected,
                "selection_status": "descriptive_only; inspect blinded images"
                if len(rows) == expected
                else "incomplete; no win claim",
            }
    return report
