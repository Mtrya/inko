"""Build benchmark blind panels and join completed ballots to raw scorer changes."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

from scripts.eval.blind_panel import cmd_panel, tally_votes
from src.evaluation.benchmarks.run import parse_score
from src.posttrain.config import Scorer
from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import decode, require
from src.posttrain.scoring.client import request_for_record
from src.posttrain.scoring.contracts import validate_results


def load_records(run):
    rows = [
        json.loads(p.read_text())
        for p in sorted((Path(run) / "records").glob("*.json"))
    ]
    require(bool(rows), "benchmark has no generated records")
    require(
        len({(r["arm"], r["prompt"]["id"]) for r in rows}) == len(rows),
        "duplicate benchmark prompt/arm record",
    )
    return rows


def panel(run, output, seed):
    run, output = Path(run), Path(output)
    rows = load_records(run)
    arms = sorted({r["arm"] for r in rows})
    require(len(arms) >= 2, "blind panel requires two benchmark arms")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "manifest_sha256": file_hash(run / "manifest.json"),
        "seed": seed,
        "images": {r["sample_id"]: r["image_sha256"] for r in rows},
    }
    marker = output / "panel.json"
    if marker.exists():
        require(
            json.loads(marker.read_text()) == identity, "blind panel identity differs"
        )
        return identity
    # Ballots and answer keys cannot be replaced after a reviewer starts voting.
    require(
        not (output / "ballot.json").exists(),
        "incomplete panel has a ballot; inspect it before recovery",
    )
    suite = {}
    for row in rows:
        require(
            file_hash(row["image_path"]) == row["image_sha256"],
            "panel image bytes changed",
        )
        destination = output / "arms" / row["arm"] / f"{row['prompt']['id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["image_path"], destination)
        suite[row["prompt"]["id"]] = row["prompt"]
    prompts = output / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in suite.values())
    )
    cmd_panel(
        argparse.Namespace(
            arm=[f"{a}={output / 'arms' / a}" for a in arms],
            prompts=str(prompts),
            out=str(output),
            seed=seed,
        )
    )
    atomic_json(marker, identity)
    return identity


def join(run, answer_key, ballot):
    run = Path(run)
    rows = load_records(run)
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    key = json.loads(Path(answer_key).read_text())
    votes = json.loads(Path(ballot).read_text())
    tally = tally_votes(key, votes)  # Reject missing, blank or extra votes.
    indexed = {(r["prompt"]["id"], r["arm"]): r for r in rows}
    prompt_ids = {p for p, _ in indexed}
    require(set(key) == prompt_ids, "ballot prompt inventory differs from benchmark")
    expected_arms = {a["name"] for a in config["arms"]}
    require(
        all(
            set(mapping.values()) == expected_arms
            and len(mapping) == len(expected_arms)
            for mapping in key.values()
        ),
        "answer key arm inventory differs",
    )
    scores = {}
    by_sample = {r["sample_id"]: r for r in rows}
    specs = {s["id"]: decode(Scorer, s) for s in config["scorers"]}
    for path in sorted((run / "scores").glob("*/*.json")):
        row = json.loads(path.read_text())
        score_key = (row["sample_id"], row["scorer"])
        require(
            score_key not in scores
            and score_key[0] in by_sample
            and score_key[1] in specs,
            "unexpected or duplicate joined score",
        )
        spec = specs[score_key[1]]
        validate_results(
            [request_for_record(by_sample[score_key[0]], spec)],
            [parse_score(row)],
            spec.outputs,
            complete=False,
        )
        scores[score_key] = row
    directions = {
        f"{s['id']}.{c['name']}": c["direction"]
        for s in config["scorers"]
        for c in s["outputs"]
    }
    reference = config["arms"][0]["name"]
    result = {
        "manifest_sha256": file_hash(run / "manifest.json"),
        "answer_key_sha256": file_hash(answer_key),
        "ballot_sha256": file_hash(ballot),
        "human": tally,
        "comparisons": {},
    }
    for arm in [a["name"] for a in config["arms"][1:]]:
        for channel in config["evaluation"]["channels"]:
            scorer, metric = channel.split(".")
            joined, missing = [], []
            for prompt_id in sorted(prompt_ids):
                pair = [indexed[(prompt_id, a)] for a in (reference, arm)]
                scored = [scores.get((r["sample_id"], scorer)) for r in pair]
                if any(s is None or s["status"] != "ok" for s in scored):
                    missing.append(prompt_id)
                    continue
                winner = tally["per_prompt"][prompt_id]
                raw = scored[1]["values"][metric] - scored[0]["values"][metric]
                improvement = raw if directions[channel] == "higher" else -raw
                # A ballot that chooses another arm cannot order this pair.
                human = (
                    0
                    if winner == "tie"
                    else (1 if winner == arm else -1 if winner == reference else None)
                )
                joined.append(
                    {
                        "prompt_id": prompt_id,
                        "scene_id": pair[0]["prompt"]["scene_id"],
                        "raw_delta": raw,
                        "scorer_improvement": improvement,
                        "human_pair_preference": human,
                        "scorer_up_human_down": human == -1 and improvement > 0,
                    }
                )
            comparable = [r for r in joined if r["human_pair_preference"] is not None]
            result["comparisons"].setdefault(arm, {})[channel] = {
                "reference": reference,
                "expected": len(prompt_ids),
                "scored": len(joined),
                "failed_or_missing": missing,
                "human_pair_comparable": len(comparable),
                "scorer_up_human_down": sum(
                    r["scorer_up_human_down"] for r in comparable
                ),
                "mean_raw_delta": (
                    statistics.mean(r["raw_delta"] for r in joined) if joined else None
                ),
                "rows": joined,
                "complete": not missing,
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("panel")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    review = sub.add_parser("join")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--answer-key", type=Path, required=True)
    review.add_argument("--ballot", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "panel":
        panel(args.run, args.output, args.seed)
    else:
        atomic_json(args.output, join(args.run, args.answer_key, args.ballot))
