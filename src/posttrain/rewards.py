"""Collect complete round scores, combine reward channels and release learners."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from .records import atomic_json, file_hash, identity
from .schema import require
from .scoring.client import request_for_record


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


def collection_path(output, attempt, step, rank):
    return (
        Path(output)
        / "collections"
        / f"{attempt}-round-{step:06d}-rank-{rank:03d}.json"
    )


def reward_path(output, attempt, step):
    return Path(output) / "rewards" / f"{attempt}-round-{step:06d}.json"


def publish_collection(
    config, rollouts, prompts, vae, stats, output, attempt, step, rank
):
    from src.utils.vae_codec import decode_latents

    directory = Path(output) / "images" / attempt / f"round-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    mean, std = stats
    for rollout in rollouts:
        require(rollout.collection_round == step, "collection round changed")
        images = decode_latents(rollout.terminal * std + mean, vae)
        for index, (key, prompt_id, image) in enumerate(
            zip(rollout.sample_ids, rollout.prompt_ids, images, strict=True)
        ):
            image_path = directory / f"{identity(key)}.png"
            temporary = image_path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            latent_path = image_path.with_suffix(".pt")
            latent = {
                "terminal": rollout.terminal[index].cpu(),
                "conditioning": tuple(t[index].cpu() for t in rollout.conditioning),
            }
            torch.save(latent, latent_path)
            records.append(
                {
                    "sample_id": key,
                    "prompt": prompts[prompt_id],
                    "generator_step": rollout.generator_step,
                    "policy_version": rollout.collection_round,
                    "group_id": prompt_id,
                    "seed": rollout.seeds[index],
                    "image_path": str(image_path),
                    "image_sha256": file_hash(image_path),
                    "latent_path": str(latent_path),
                }
            )
    atomic_json(
        collection_path(output, attempt, step, rank),
        {"step": step, "rank": rank, "records": records},
    )
    return records


def acquire_round(config, clients, records, *, on_scored=None):
    """Complete each required request before admitting any optimizer update."""
    require(
        len({r["sample_id"] for r in records}) == len(records),
        "duplicate collected sample",
    )
    jobs = []
    for spec in config.scorers:
        client = clients[spec.id]
        requests = [request_for_record(record, spec) for record in records]
        for start in range(0, len(requests), spec.batch_size):
            jobs.append((client, requests[start : start + spec.batch_size]))
    scores = []
    counts = {}
    start = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=sum(s.concurrency for s in config.scorers)
    ) as pool:
        futures = [pool.submit(client.acquire, batch) for client, batch in jobs]
        for future in as_completed(futures):
            batch = future.result()
            scores.extend(batch)
            for row in batch:
                values = counts.setdefault(
                    row["scorer"], {"retry_attempts": 0, "failed_requests": 0}
                )
                values["retry_attempts"] += max(0, row["attempts"] - 1)
                values["failed_requests"] += row["status"] != "ok"
            if on_scored is not None:
                on_scored(counts)
    ids = sorted(r["sample_id"] for r in records)
    combined = combine_round(ids, scores, config)
    return {
        "reward": combined,
        "scores": scores,
        "acquisition_seconds": time.monotonic() - start,
    }


def score_ready_round(config, services, output, attempt, step, *, on_scored=None):
    policy_version = step
    paths = [
        collection_path(output, attempt, step, rank)
        for rank in range(len(config.execution.learner_devices))
    ]
    destination = reward_path(output, attempt, step)
    if destination.exists() or not all(path.exists() for path in paths):
        return False
    rows = []
    for rank, path in enumerate(paths):
        packet = json.loads(path.read_text())
        require(
            packet["step"] == step and packet["rank"] == rank,
            "collection round/rank mismatch",
        )
        rows.extend(packet["records"])
    expected = {
        f"{step}:{slot}:{draw}"
        for slot in range(config.collection.prompts_per_round)
        for draw in range(config.collection.draws_per_prompt)
    }
    require(
        len(rows) == len(expected) and {r["sample_id"] for r in rows} == expected,
        "complete global collection required",
    )
    # Inactive ART collections advance the collection ID without an update.
    started = json.loads(
        (Path(output) / "attempts" / f"{attempt}-round-{step:06d}.json").read_text()
    )
    generator_step = started["step"]
    require(
        all(
            r["policy_version"] == policy_version
            and r["generator_step"] == generator_step
            for r in rows
        ),
        "collection policy changed",
    )
    result = acquire_round(config, services.clients, rows, on_scored=on_scored)
    atomic_json(
        destination,
        {
            "step": step,
            "policy_version": policy_version,
            **result,
        },
    )
    return True


def wait_for_rewards(config, output, attempt, step, records):
    """Wait for the round's complete combined-reward map, keyed by sample ID."""
    path = reward_path(output, attempt, step)
    deadline = time.monotonic() + config.execution.round_timeout
    while not path.exists():
        require(
            time.monotonic() < deadline, "terminal scoring barrier deadline exceeded"
        )
        time.sleep(0.1)
    packet = json.loads(path.read_text())
    require(packet["step"] == step, "reward/collection round mismatch")
    reward = packet["reward"]
    values = dict(zip(reward["sample_ids"], reward["combined"], strict=True))
    require(
        {r["sample_id"] for r in records} <= values.keys(),
        "missing local reward",
    )
    return values, reward
