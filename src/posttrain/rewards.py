"""Poll streaming scores and assemble the full-round ART reward barrier."""

import io
import json
import time
from pathlib import Path

import torch

from .records import atomic_json
from .schema import require
from .scoring.contracts import image_payload


def combine_round(sample_ids, score_records, config):
    require(
        bool(sample_ids) and len(set(sample_ids)) == len(sample_ids),
        "unique reward sample set required",
    )
    sample_ids = sorted(sample_ids)
    expected = set(sample_ids)
    joined = {}
    for record in score_records:
        key = record["sample_id"]
        require(key in expected and key not in joined, "duplicate/unknown score join")
        joined[key] = record
    require(joined.keys() == expected, "required score set is incomplete")
    combined = torch.zeros(len(sample_ids), dtype=torch.float32)
    statistics = {}
    raw = {}
    for channel in config.rewards.channels:
        values = torch.tensor(
            [joined[sample]["values"][channel.channel] for sample in sample_ids],
            dtype=torch.float32,
        )
        require(bool(torch.isfinite(values).all()), "nonfinite required reward")
        raw[channel.channel] = values.tolist()
        oriented = values
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
            "direction": "higher",
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


def reward_path(output, attempt, step):
    return Path(output) / "rewards" / f"{attempt}-round-{step:06d}.json"


def submit_rollout(rollout, vae, stats, submissions, panel):
    from src.utils.vae_codec import decode_latents

    mean, std = stats
    images = decode_latents(rollout.terminal * std + mean, vae)
    generated_at = time.time()
    for key, prompt_id, image in zip(
        rollout.sample_ids, rollout.prompt_ids, images, strict=True
    ):
        _, slot, draw = key.split(":")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        submissions.submit(key, slot, int(draw), image_payload(stream.getvalue()))
        # Keep complete prompt groups so every reviewed pair has both candidates.
        if int(slot) < 4:
            panel.mkdir(parents=True, exist_ok=True)
            path = panel / f"{slot}-{draw}.png"
            path.write_bytes(stream.getvalue())
            atomic_json(
                path.with_suffix(".json"),
                {
                    "sample_id": key,
                    "prompt_id": prompt_id,
                    "collection": submissions.collection_id,
                    "generator_step": rollout.generator_step,
                },
            )
    return generated_at


class RoundScores:
    def __init__(self, config, client, output, attempt, step):
        self.config, self.client, self.output, self.attempt, self.step = (
            config,
            client,
            Path(output),
            attempt,
            step,
        )
        self.scores, self.offset, self.progress = [], 0, None

    def poll(self, on_scored=None):
        started_path = (
            self.output / "attempts" / f"{self.attempt}-round-{self.step:06d}.json"
        )
        if not started_path.exists():
            return False
        started = json.loads(started_path.read_text())
        collection = started["collection"]
        packet = self.client.read(collection, self.offset)
        self.scores.extend(packet["results"])
        self.offset = packet["next"]
        progress = packet["progress"]
        counters = (
            progress["accepted"],
            progress["completed"],
            progress["pairs_completed"],
            *(v["calls"] for _, v in sorted(progress["backends"].items())),
        )
        if counters != self.progress:
            atomic_json(
                self.output / "progress" / f"{self.attempt}-scoring.json", progress
            )
            self.progress = counters
            if on_scored is not None:
                on_scored(progress["backends"])
        if packet["errors"]:
            raise RuntimeError(f"collection scoring failed: {packet['errors']}")
        if progress["completed"] != progress["expected"]:
            return False
        expected = {
            f"{self.step}:{slot}:{draw}"
            for slot in range(self.config.collection.prompts_per_round)
            for draw in range(self.config.collection.draws_per_prompt)
        }
        reward = combine_round(expected, self.scores, self.config)
        atomic_json(
            reward_path(self.output, self.attempt, self.step),
            {
                "step": self.step,
                "reward": reward,
                "scores": self.scores,
                "progress": progress,
            },
        )
        self.client.release(collection)
        active = self.output / "active_collection.json"
        if active.exists() and json.loads(active.read_text())["id"] == collection:
            active.unlink()
        self.scores, self.offset, self.progress = [], 0, None
        self.step += 1
        return True


def wait_for_rewards(config, output, attempt, step, sample_ids):
    path = reward_path(output, attempt, step)
    progress_path = Path(output) / "progress" / f"{attempt}-scoring.json"
    started = time.time()
    while not path.exists():
        latest = max(
            started,
            progress_path.stat().st_mtime if progress_path.exists() else started,
        )
        require(
            time.time() - latest < config.execution.round_timeout,
            "scoring made no progress before its deadline",
        )
        time.sleep(0.2)
    packet = json.loads(path.read_text())
    require(packet["step"] == step, "reward/collection round mismatch")
    reward = packet["reward"]
    values = dict(zip(reward["sample_ids"], reward["combined"], strict=True))
    require(set(sample_ids) <= values.keys(), "missing local reward")
    return values, reward
