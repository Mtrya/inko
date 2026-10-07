"""Global round reward orientation, population statistics and detached weights."""

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
    for channel in config.rewards.channels:
        service, metric = channel.channel.split(".", 1)
        values = torch.tensor(
            [joined[(sample, service)]["values"][metric] for sample in sample_ids],
            dtype=torch.float32,
        )
        require(bool(torch.isfinite(values).all()), "nonfinite required reward")
        raw[channel.channel] = values.tolist()
        oriented = values if channel.direction == "higher" else -values
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
            "direction": channel.direction,
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


def transform_round(sample_ids, score_records, config):
    result = combine_round(sample_ids, score_records, config)
    sample_ids = result["sample_ids"]
    combined = torch.tensor(result["combined"], dtype=torch.float32)
    if config.rewards.weighting == "mean_one_exp":
        # Subtract the reward maximum before division: finite extreme rewards
        # cannot overflow into +inf logits and poison softmax.
        weights = len(sample_ids) * torch.softmax(
            (combined - combined.max()) / config.rewards.temperature, dim=0
        )
    else:
        require(
            bool((combined >= 0).all()) and bool(combined.max() > 0),
            "paper-max requires nonnegative rewards and a positive global maximum",
        )
        weights = torch.exp(combined / combined.max())
    require(bool(torch.isfinite(weights).all()), "nonfinite RT-DM weights")
    return {
        **result,
        "weights": weights.tolist(),
        "diagnostics": {
            "reward_min": float(combined.min()),
            "reward_max": float(combined.max()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "effective_samples": float(weights.sum().square() / weights.square().sum()),
        },
    }


def group_advantages(reward, groups, *, scale, clip, epsilon):
    """Complete-round population moments; exact prompt IDs define groups."""
    ids = reward["sample_ids"]
    require(
        set(groups) == set(ids) and len(ids) == len(set(ids)),
        "complete NFT group map required",
    )
    require(
        scale in {"group", "round"} and clip > 0 and epsilon > 0,
        "invalid NFT advantage transform",
    )
    values = torch.tensor(reward["combined"], dtype=torch.float32)
    require(
        values.shape == (len(ids),) and bool(torch.isfinite(values).all()),
        "invalid NFT rewards",
    )
    advantages = torch.zeros_like(values)
    moments = {}
    round_std = values.std(correction=0)
    require(bool(torch.isfinite(round_std)), "nonfinite NFT round population std")
    for group in sorted(set(groups.values())):
        positions = [i for i, key in enumerate(ids) if groups[key] == group]
        require(len(positions) >= 2, "NFT group must have at least two samples")
        local = values[positions]
        mean, std = local.mean(), local.std(correction=0)
        denominator = std if scale == "group" else round_std
        require(
            bool(torch.isfinite(mean) & torch.isfinite(denominator)),
            "nonfinite NFT group moments",
        )
        if not bool((local == local[0]).all()):
            advantages[positions] = (local - mean) / (denominator + epsilon)
        moments[group] = {
            "sample_ids": [ids[i] for i in positions],
            "mean": float(mean),
            "std": float(std),
            "denominator_std": float(denominator),
        }
    probabilities = 0.5 + advantages.clamp(-clip, clip) / (2 * clip)
    require(bool(torch.isfinite(advantages).all()), "nonfinite NFT advantages")
    return {
        **reward,
        "group_ids": [groups[key] for key in ids],
        "groups": moments,
        "round_std": float(round_std),
        "advantage_scale": scale,
        "advantages": advantages.tolist(),
        "probabilities": probabilities.tolist(),
    }
