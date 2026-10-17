"""TIIF testmini's paired descriptions, Qwen VLM questions and PaddleOCR."""

import random
import re
from collections import defaultdict
from pathlib import Path

from src.evaluation.records import read_jsonl

from .common import link_image, load_source, mean, prompt, write_json


def prepare(assets, settings):
    root = Path(assets["source"]) / "data"
    result = []
    scene = 0
    for path in sorted((root / "testmini_prompts").glob("*.jsonl")):
        rows = read_jsonl(path)
        questions = read_jsonl(
            root
            / "testmini_eval_prompts"
            / path.name.replace("_prompts", "_eval_prompts")
        )
        if len(rows) != len(questions):
            raise ValueError(f"TIIF prompt/question count mismatch: {path.name}")
        for index, (row, question) in enumerate(zip(rows, questions)):
            if row["type"] != question["type"]:
                raise ValueError(f"TIIF dimension mismatch: {path.name}:{index}")
            metadata = {"type": row["type"], "index": index, "questions": question}
            if row["type"] == "text":
                metadata["text"] = re.findall(r'"([^"\n]+)"', row["short_description"])
                if not metadata["text"]:
                    raise ValueError(f"missing TIIF text ground truth: {index}")
            for variant in ("short", "long"):
                result.append(
                    prompt(
                        "tiif",
                        "testmini",
                        f"{row['type']}/{index}",
                        row[f"{variant}_description"],
                        metadata,
                        [settings["seed"] + scene],
                        variant,
                    )
                )
            scene += 1
    return result


def layout(groups, root):
    for group in groups:
        row, sample = group["prompt"], group["samples"][0]
        meta = row["metadata"]
        path = (
            root
            / meta["type"]
            / sample["model"]
            / f"{row['variant']}_description"
            / f"{meta['index']}.png"
        )
        link_image(sample["image_path"], path)


def evaluate(groups, assets, settings, root, emit):
    source = Path(assets["source"]) / "eval"
    if settings["dimension"] == "ocr":
        from paddleocr import PaddleOCR

        native = load_source(source / "paddleocr_models.py")
        metrics = load_source(source / "cal_gned_and_recall_models.py")
        ocr = PaddleOCR(lang="en", **assets.get("ocr_options", {}))
        emit.loaded()
        for group in groups:
            result = native.extract_text_with_paddleocr(
                group["samples"][0]["image_path"], ocr
            )
            words = [item[1][0].lower() for item in result] if result else [""]
            gt = [word.lower() for word in group["prompt"]["metadata"]["text"]]
            emit(
                group,
                "ocr",
                {
                    "ocr": result,
                    "text": gt,
                    "GNED": metrics.matching_based_nled(gt, words),
                    "recall": metrics.calculate_recall(gt, words),
                },
            )
        return
    import openai

    native = load_source(source / "eval_with_vlm.py")
    client = openai.OpenAI(
        base_url=assets["base_url"], api_key="local", timeout=600, max_retries=2
    )
    emit.loaded()
    for group in groups:
        row = group["prompt"]
        meta = row["metadata"]
        questions = meta["questions"]["yn_question_list"]
        # Make the released random template selection stable across interruption.
        random.seed(row["seeds"][0])
        text = native.format_questions_prompt(native.raw_prompt, questions)
        response = native.generate_with_prompt(
            text, group["samples"][0]["image_path"], client, model=settings["judge"]
        )
        predictions = native.extract_yes_no(response, questions)
        result = {
            "attribute": meta["type"],
            "desc": f"{row['variant']}_description",
            "line_idx": meta["index"],
            "questions": questions,
            "gt_answers": meta["questions"]["yn_answer_list"],
            "model_pred": predictions,
            "model_output": response,
            "judge_prompt": text,
        }
        write_json(
            root.parent
            / "vlm"
            / group["samples"][0]["model"]
            / meta["type"]
            / row["variant"]
            / f"{meta['index']}.json",
            result,
        )
        emit(group, "vlm", result)


# The nine groups in the released summary_dimension_results.py.
GROUPS = {
    "Basic_Attribute": [
        "shape+color",
        "color+texture",
        "texture+color",
        "shape+texture",
    ],
    "Basic_Relation": [
        "2d_spatial_relation",
        "3d_spatial_relation",
        "action+2d",
        "action+3d",
    ],
    "Basic_Reasoning": ["numeracy", "negation", "differentiation", "comparison"],
    "Advanced_Attribute+Relation": [
        "action+color",
        "action+texture",
        "color+2d",
        "color+3d",
        "shape+2d",
        "shape+3d",
        "texture+2d",
        "texture+3d",
    ],
    "Advanced_Attribute+Reasoning": [
        f"{a}+{b}"
        for a in ("numeracy", "comparison", "differentiation", "negation")
        for b in ("color", "texture")
    ],
    "Advanced_Relation+Reasoning": [
        f"{a}+{b}"
        for a in ("numeracy", "comparison", "differentiation", "negation")
        for b in ("2d", "3d")
    ],
    "Advanced_Text Generation": ["text"],
    "Advanced_Style Control": ["style"],
    "Real_Complex": ["real_world"],
}


def summarize(rows, assets):
    result = {}
    for variant in ("short", "long"):
        attributes, ocr = defaultdict(list), []
        for row in rows:
            if row["prompt"]["variant"] != variant:
                continue
            native = row["native"]
            if row["dimension"] == "ocr":
                ocr.append(native)
            else:
                attributes[native["attribute"]].extend(
                    float(a.strip().lower() == b.strip().lower())
                    for a, b in zip(native["gt_answers"], native["model_pred"])
                )
        scores = {key: mean(values) for key, values in attributes.items()}
        groups = {
            key: mean(scores[a] for a in names if a in scores)
            for key, names in GROUPS.items()
        }
        result[variant] = {
            "attributes": scores,
            "groups": groups,
            "overall": mean(v for v in groups.values() if v is not None),
            "GNED": mean(r["GNED"] for r in ocr),
            "recall": mean(r["recall"] for r in ocr),
        }
    return result
