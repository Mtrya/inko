"""DPG's released mPLUG questions, sequential dependencies and native means."""

import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .common import grid, load_source, mean, prompt


def questions(source):
    grouped = {}
    # DictReader consumes the header. Upstream's extra `i == 0` skip loses q1.
    with (Path(source) / "dpg_bench/dpg_bench.csv").open() as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["item_id"], []).append(row)
    return grouped


def prepare(assets, settings):
    result = []
    for i, (identifier, rows) in enumerate(questions(assets["source"]).items()):
        text = (
            (Path(assets["source"]) / "dpg_bench/prompts" / f"{identifier}.txt")
            .read_text()
            .strip()
        )
        result.append(
            prompt(
                "dpg",
                "released",
                identifier,
                text,
                {"questions": rows},
                [settings["seed"] + i * 4 + j for j in range(4)],
            )
        )
    return result


def layout(groups, root):
    for group in groups:
        width, height = grid(
            group["samples"], root / f"{group['prompt']['prompt_id']}.png"
        )
        if width != height:
            raise ValueError("the official DPG cropper requires square draws")


def apply_dependencies(answers, questions):
    scores = {
        int(q["proposition_id"]): float(a == "yes") for q, a in zip(questions, answers)
    }
    for question in questions:
        qid = int(question["proposition_id"])
        parents = [int(p.strip()) for p in question["dependency"].split(",")]
        if any(parent and scores[parent] == 0 for parent in parents):
            scores[qid] = 0.0
    return scores


def evaluate(groups, assets, settings, root, emit):
    upstream = load_source(Path(assets["source"]) / "dpg_bench/compute_dpg_bench.py")
    model = upstream.MPLUG(ckpt=assets["model"], device="cuda:0")
    emit.loaded()
    for group in groups:
        qs = group["prompt"]["metadata"]["questions"]
        native = {"qid2tuple": {}, "qid2dependency": {}, "qid2question": {}}
        for q in qs:
            qid = int(q["proposition_id"])
            native["qid2tuple"][qid] = q["tuple"]
            native["qid2dependency"][qid] = [int(p) for p in q["dependency"].split(",")]
            native["qid2question"][qid] = q["question_natural_language"]

        class Capture:
            def __init__(self):
                self.answers = []

            def vqa(self, image, question):
                answer = model.vqa(image, question)
                self.answers.append(answer)
                return answer

        capture = Capture()
        identifier = group["prompt"]["prompt_id"]
        args = SimpleNamespace(pic_num=4, res_path=str(root.parent / "dpg-native.txt"))
        score, _, _ = upstream.compute_dpg_one_sample(
            args,
            {identifier: native},
            str(root / f"{identifier}.png"),
            capture,
            group["samples"][0]["width"],
        )
        draws = []
        for draw in range(4):
            responses = capture.answers[draw * len(qs) : (draw + 1) * len(qs)]
            adjusted = apply_dependencies(responses, qs)
            draws.append(
                {
                    "answers": responses,
                    "question_scores": adjusted,
                    "score": mean(adjusted.values()),
                }
            )
        emit(group, "vqa", {"score": score, "draws": draws})


def summarize(rows, assets):
    l2 = defaultdict(list)
    for row in rows:
        # Upstream category readout uses unfiltered answers from the last draw.
        answers = row["native"]["draws"][-1]["answers"]
        for q, answer in zip(row["prompt"]["metadata"]["questions"], answers):
            l2[q["tuple"].split("(")[0].strip()].append(float(answer == "yes"))
    l1 = defaultdict(list)
    for key, values in l2.items():
        l1[key.split("-")[0].strip()].extend(values)
    return {
        "overall": 100 * mean(r["native"]["score"] for r in rows),
        "level1": {k: 100 * mean(v) for k, v in l1.items()},
        "level2": {k: 100 * mean(v) for k, v in l2.items()},
    }
