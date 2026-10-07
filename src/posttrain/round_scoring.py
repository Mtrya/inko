"""Terminal-image scoring barrier using the shared backend-independent client."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .records import atomic_json, file_hash, identity
from .schema import require
from .scoring.client import request_for_record


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
    import torch

    from src.utils.vae_codec import decode_latents

    from .rollout import tensor_digest

    directory = Path(output) / "images" / attempt / f"round-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    mean, std = stats
    for rollout in rollouts:
        images = decode_latents(rollout.terminal * std + mean, vae)
        for index, (key, prompt_id, image) in enumerate(
            zip(rollout.sample_ids, rollout.prompt_ids, images, strict=True)
        ):
            image_path = directory / f"{identity(key)}.png"
            temporary = image_path.with_suffix(".png.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            latent_path = image_path.with_suffix(".pt")
            torch.save(
                {
                    "terminal": rollout.terminal[index].cpu(),
                    "selected_input": rollout.selected_input[index].cpu(),
                    "selected_time": rollout.selected_time[index].cpu(),
                    "selected_index": rollout.selected_index[index].cpu(),
                    "selected_prediction": rollout.selected_prediction[index].cpu(),
                },
                latent_path,
            )
            records.append(
                {
                    "sample_id": key,
                    "prompt": prompts[prompt_id],
                    "student_version": step,
                    "seed": rollout.seeds[index],
                    "image_path": str(image_path),
                    "image_sha256": file_hash(image_path),
                    "latent_path": str(latent_path),
                    "latent_sha256": file_hash(latent_path),
                    "selected_index": int(rollout.selected_index[index]),
                    "selected_time": float(rollout.selected_time[index]),
                    "conditioning": [
                        tensor_digest(t[index]) for t in rollout.conditioning
                    ],
                    "unconditional": None
                    if rollout.unconditional is None
                    else [tensor_digest(t[index]) for t in rollout.unconditional],
                    "noise_specification": {
                        "version": "cold-cpu-fp32-v2",
                        "round": step,
                        "sample": key,
                        "collection_seed": rollout.seeds[index],
                    },
                }
            )
    atomic_json(
        collection_path(output, attempt, step, rank),
        {"step": step, "rank": rank, "records": records},
    )
    return records


def acquire_round(config, clients, records):
    """Complete each required request before admitting any reward-weighted update."""
    from .rewards import transform_round

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
    start = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=sum(s.concurrency for s in config.scorers)
    ) as pool:
        futures = [pool.submit(client.acquire, batch) for client, batch in jobs]
        for future in futures:
            scores.extend(future.result())
    result = transform_round(sorted(r["sample_id"] for r in records), scores, config)
    return {
        "reward": result,
        "scores": scores,
        "acquisition_seconds": time.monotonic() - start,
    }


def score_ready_round(config, services, output, attempt, step):
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
    require(
        all(r["student_version"] == step for r in rows), "collection policy changed"
    )
    result = acquire_round(config, services.clients, rows)
    atomic_json(
        destination,
        {"step": step, "collections": [file_hash(path) for path in paths], **result},
    )
    return True


def wait_for_rewards(config, output, attempt, step, rank, records):
    path = reward_path(output, attempt, step)
    deadline = time.monotonic() + config.execution.round_timeout
    while not path.exists():
        require(
            time.monotonic() < deadline, "terminal scoring barrier deadline exceeded"
        )
        time.sleep(0.1)
    packet = json.loads(path.read_text())
    require(
        packet["step"] == step
        and packet["collections"][rank]
        == file_hash(collection_path(output, attempt, step, rank)),
        "reward/collection identity mismatch",
    )
    reward = packet["reward"]
    weights = dict(zip(reward["sample_ids"], reward["weights"], strict=True))
    require({r["sample_id"] for r in records} <= weights.keys(), "missing local reward")
    return weights, reward
