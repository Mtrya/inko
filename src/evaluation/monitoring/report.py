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


def build_report(records, scores, config, *, suite="development"):
    samples = {r["sample_id"]: r for r in records}
    arms = [a.name for a in config.arms]
    indexed = {}
    for score in scores:
        key = score["sample_id"]
        if key in indexed or key not in samples:
            raise ValueError("duplicate or unexpected score in report")
        indexed[key] = score
    report = {
        "suite": suite,
        "scope": f"{suite} prompt evaluation; scorer metrics are not independent reward-model validation.",
        "arms": {},
        "paired": {},
        "uncertainty": "Paired cluster bootstrap by scene; conditional on these fixed noise seeds.",
    }
    for channel in config.evaluation.channels:
        metric = channel
        by_arm = {}
        for arm in arms:
            selected = [r for r in records if r["arm"] == arm]
            good, failures = {}, []
            for record in selected:
                result = indexed.get(record["sample_id"])
                if result is None or channel not in result["values"]:
                    failures.append(record["prompt"]["id"])
                else:
                    good[record["prompt"]["id"]] = (
                        record["prompt"]["scene_id"],
                        result["values"][metric],
                    )
            by_arm[arm] = good
            values = sorted(v for _, v in good.values())
            report["arms"].setdefault(arm, {})[channel] = {
                "metric": metric,
                "expected": len(selected),
                "scored": len(good),
                "failed_or_missing": failures,
                "mean_on_successes": (
                    statistics.mean(v for _, v in good.values()) if good else None
                ),
                "complete": len(good) == len(selected),
                "population_stddev": statistics.pstdev(values) if values else None,
                "minimum": values[0] if values else None,
                "maximum": values[-1] if values else None,
                "quantiles": {
                    str(p): quantile(values, p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)
                },
            }
        reference = arms[0]
        for arm in arms[1:]:
            common = sorted(by_arm[reference].keys() & by_arm[arm].keys())
            rows = [
                (by_arm[reference][p][0], by_arm[arm][p][1] - by_arm[reference][p][1])
                for p in common
            ]
            expected = sum(r["arm"] == reference for r in records)
            settings = config.evaluation
            report["paired"].setdefault(arm, {})[channel] = {
                "reference": reference,
                "paired": len(rows),
                "expected": expected,
                "mean_delta": statistics.mean(v for _, v in rows) if rows else None,
                "confidence": settings.confidence,
                "interval": paired_interval(
                    rows,
                    settings.bootstrap_repeats,
                    settings.confidence,
                    settings.bootstrap_seed,
                ),
                "coverage_complete": len(rows) == expected,
                "selection_status": (
                    "descriptive_only; inspect blinded images"
                    if len(rows) == expected
                    else "incomplete; no win claim"
                ),
            }
            if channel.startswith("rubric."):
                values = [by_arm[arm][p][1] for p in common]
                category = channel.split(".", 1)[1]
                orders = [
                    indexed[r["sample_id"]]["judgment"]["orders"]
                    for r in records
                    if r["arm"] == arm and r["prompt"]["id"] in common
                ]
                report["paired"][arm][channel].update(
                    wins=sum(v == 1 for v in values),
                    ties=sum(v == 0.5 for v in values),
                    losses=sum(v == 0 for v in values),
                    win_credit=statistics.mean(values) if values else None,
                    order_disagreements=sum(
                        a[category]["winner"] != b[category]["winner"]
                        for a, b in orders
                    ),
                )
    return report


def quantile(values, probability):
    """Linear interpolation between the sorted empirical values."""
    if not values:
        return None
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
