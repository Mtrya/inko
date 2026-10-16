"""Global round reward orientation and population statistics."""

import torch

from .schema import require


def combine_round(sample_ids, score_records, config):
    require(
        bool(sample_ids) and len(set(sample_ids)) == len(sample_ids),
        "unique reward sample set required",
    )
    sample_ids = sorted(sample_ids)
    expected = {(sample, spec.id) for sample in sample_ids for spec in config.scorers}
    joined = {}
    for record in score_records:
        key = (record["sample_id"], record["scorer"])
        require(key in expected and key not in joined, "duplicate/unknown score join")
        require(record["status"] == "ok", f"required score failed: {key}")
        joined[key] = record
    require(joined.keys() == expected, "required score set is incomplete")
    combined = torch.zeros(len(sample_ids), dtype=torch.float32)
    statistics = {}
    raw = {}
    directions = {
        f"{scorer.id}.{output.name}": output.direction
        for scorer in config.scorers
        for output in scorer.outputs
    }
    for channel in config.rewards.channels:
        service, metric = channel.channel.split(".", 1)
        values = torch.tensor(
            [joined[(sample, service)]["values"][metric] for sample in sample_ids],
            dtype=torch.float32,
        )
        require(bool(torch.isfinite(values).all()), "nonfinite required reward")
        raw[channel.channel] = values.tolist()
        direction = directions[channel.channel]
        oriented = values if direction == "higher" else -values
        mean, std = oriented.mean(), oriented.std(correction=0)
        require(
            bool(torch.isfinite(mean) & torch.isfinite(std)),
            "nonfinite reward population statistics",
        )
        transformed = (
            (oriented - mean) / (std + config.rewards.epsilon)
            if channel.standardization == "round"
            else oriented
        )
        if bool((oriented == oriented[0]).all()) and channel.standardization == "round":
            transformed = torch.zeros_like(oriented)
        combined += channel.weight * transformed
        statistics[channel.channel] = {
            "mean": float(mean),
            "std": float(std),
            "direction": direction,
            "standardization": channel.standardization,
            "transformed": transformed.tolist(),
        }
    require(bool(torch.isfinite(combined).all()), "nonfinite combined rewards")
    return {
        "sample_ids": sample_ids,
        "raw": raw,
        "statistics": statistics,
        "combined": combined.tolist(),
    }
