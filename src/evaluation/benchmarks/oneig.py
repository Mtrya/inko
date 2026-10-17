"""OneIG English's five released scorers and fine-grained readouts."""

import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from .common import grid, mean, prompt

CATEGORIES = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
}

# Families in the released fine_grained_analysis.py; each style has equal weight.
STYLE_FAMILIES = {
    "traditional": [
        "abstract_expressionism",
        "art_nouveau",
        "baroque",
        "chinese_ink_painting",
        "cubism",
        "fauvism",
        "impressionism",
        "line_art",
        "minimalism",
        "pointillism",
        "pop_art",
        "rococo",
        "ukiyo-e",
    ],
    "media": [
        "clay",
        "crayon",
        "graffiti",
        "lego",
        "comic",
        "pencil_sketch",
        "stone_sculpture",
        "watercolor",
    ],
    "anime": [
        "celluloid",
        "chibi",
        "cyberpunk",
        "ghibli",
        "impasto",
        "pixar",
        "pixel_art",
        "3d_rendering",
    ],
}


def prepare(assets, settings):
    with (Path(assets["source"]) / "OneIG-Bench.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return [
        prompt(
            "oneig",
            "en",
            f"{CATEGORIES[row['category']]}/{row['id']}",
            row["prompt_en"],
            {"native": row, "category": CATEGORIES[row["category"]]},
            [settings["seed"] + i * 4 + j for j in range(4)],
            "en",
        )
        for i, row in enumerate(rows)
    ]


def layout(groups, root):
    for group in groups:
        row = group["prompt"]
        meta = row["metadata"]
        grid(
            group["samples"],
            root
            / meta["category"]
            / group["samples"][0]["model"]
            / f"{meta['native']['id']}.png",
        )


def dimensions(row):
    meta = row["metadata"]
    category = meta["category"]
    result = ["diversity"]
    if category in ("anime", "human", "object"):
        result.append("alignment")
    if category in ("text", "reasoning"):
        result.append(category)
    if category == "anime" and meta["native"]["class"]:
        result.append("style")
    return result


def alignment_values(answers, dependencies):
    raw = {
        int(qid): [float(a == "Yes") for a in values] for qid, values in answers.items()
    }
    adjusted = {qid: list(values) for qid, values in raw.items()}
    for qid, parents in dependencies.items():
        for draw in range(4):
            if any(p and raw[p][draw] == 0 for p in parents):
                adjusted[int(qid)][draw] = 0.0
    return [mean(values[d] for values in adjusted.values()) for d in range(4)]


def evaluate(groups, assets, settings, root, emit):
    import torch
    from PIL import Image

    source = Path(assets["source"])
    os.chdir(source)
    sys.path.insert(0, str(source))
    from scripts.utils import inference
    from scripts.utils.utils import split_mxn_grid

    dimension = settings["dimension"]
    if dimension in ("alignment", "text"):
        model = inference.Qwen2_5VLBatchInferencer(assets["qwen"])
    elif dimension == "reasoning":
        model = inference.LLM2CLIP(
            processor_model=assets["clip_processor"],
            model_name=assets["llm2clip_vision"],
            llm_model_name=assets["llm2clip_text"],
        )
    elif dimension == "style":
        csd = inference.CSDStyleEmbedding(assets["csd"])
        encoder = inference.SEStyleEmbedding(assets["style_encoder"])
        csd_ref = torch.load(source / "scripts/style/CSD_embed.pt", weights_only=False)
        se_ref = torch.load(source / "scripts/style/SE_embed.pt", weights_only=False)
    else:
        from dreamsim import dreamsim

        model, preprocess = dreamsim(
            pretrained=True, device="cuda", cache_dir=assets["dreamsim"]
        )

    if dimension == "text":
        from scripts.text import text_utils as text

        with (source / "scripts/text/text_content.csv").open() as stream:
            texts = {r["id"]: r["text_content"] for r in csv.DictReader(stream)}
    if dimension == "reasoning":
        ground_truth = json.loads(
            (source / "scripts/reasoning/gt_answer.json").read_text()
        )
    emit.loaded()
    for group in groups:
        row = group["prompt"]
        meta = row["metadata"]
        identifier = meta["native"]["id"]
        image = (
            root / meta["category"] / group["samples"][0]["model"] / f"{identifier}.png"
        )
        with tempfile.TemporaryDirectory(dir=root.parent) as scratch:
            paths = split_mxn_grid(str(image), (2, 2), scratch)
            if len(paths) != 4:
                raise ValueError(f"OneIG cropper omitted a draw for {row['prompt_id']}")
            if dimension == "alignment":
                data = json.loads(
                    (
                        source / f"scripts/alignment/Q_D/{meta['category']}.json"
                    ).read_text()
                )[identifier]
                questions = json.loads(data["question"])
                dependencies = json.loads(data["dependency"])
                answers = {
                    qid: model.infer_semantic(paths, question)
                    for qid, question in questions.items()
                }
                values = alignment_values(answers, dependencies)
                result = {
                    "questions": questions,
                    "dependencies": dependencies,
                    "answers": answers,
                    "draw_scores": values,
                    "score": mean(values),
                }
            elif dimension == "text":
                gt = texts[identifier]
                outputs = model.infer_ocr(paths, 256 if len(gt.split()) > 60 else 128)
                gt_clean = text.preprocess_string(gt)
                values = []
                for output in text.clean_and_remove_hallucinations(outputs):
                    cleaned = text.preprocess_string(output)
                    distance = float(text.levenshtein_distance(cleaned, gt_clean))
                    matched, ratio, count = text.calculate_char_match_ratio(
                        gt_clean, cleaned
                    )
                    values.append(
                        {
                            "ED": distance,
                            "CR": float(distance == 0),
                            "WAC": ratio,
                            "matched": matched,
                            "count": count,
                        }
                    )
                result = {"text": gt, "outputs": outputs, "draw_scores": values}
            elif dimension == "reasoning":
                values = model.text_img_similarity_score(
                    paths, ground_truth[identifier]
                )
                if values is None or len(values) != 4:
                    raise ValueError(f"LLM2CLIP failed for {identifier}")
                result = {
                    "answer": ground_truth[identifier],
                    "draw_scores": values,
                    "score": mean(values),
                }
            elif dimension == "style":
                style = meta["native"]["class"].lower().replace(" ", "_")
                values = []
                for path in paths:
                    a = max(
                        torch.max(
                            csd.get_style_embedding(path) @ csd_ref[style].T
                        ).item(),
                        0,
                    )
                    b = max(
                        torch.max(
                            encoder.get_style_embedding(path) @ se_ref[style].T
                        ).item(),
                        0,
                    )
                    values.append({"CSD": a, "StyleEncoder": b, "score": (a + b) / 2})
                result = {
                    "style": style,
                    "draw_scores": values,
                    "score": mean(v["score"] for v in values),
                }
            else:
                values = []
                with torch.no_grad():
                    for i in range(4):
                        for j in range(i + 1, 4):
                            a, b = (
                                preprocess(Image.open(paths[k])).to("cuda")
                                for k in (i, j)
                            )
                            values.append(
                                {"draws": [i, j], "distance": model(a, b).item()}
                            )
                result = {"pairs": values, "score": mean(v["distance"] for v in values)}
            emit(group, dimension, result)


def summarize(rows, assets):
    result = {}
    for dimension in ("alignment", "text", "reasoning", "style", "diversity"):
        selected = [r for r in rows if r["dimension"] == dimension]
        if not selected:
            continue
        if dimension == "text":
            draws = [d for r in selected for d in r["native"]["draw_scores"]]
            ed, cr = mean(d["ED"] for d in draws), mean(d["CR"] for d in draws)
            wac = sum(d["matched"] for d in draws) / sum(d["count"] for d in draws)
            result[dimension] = {
                "ED": ed,
                "CR": cr,
                "WAC": wac,
                "score": 1 - min(100, ed) * (1 - cr) * (1 - wac) / 100,
            }
        else:
            result[dimension] = {"score": mean(r["native"]["score"] for r in selected)}
        breakdown = defaultdict(list)
        for row in selected:
            meta = row["prompt"]["metadata"]["native"]
            tags = [meta["category"], meta["prompt_length"], meta["type"]]
            if meta["class"]:
                tags.append(meta["class"])
            value = (
                [
                    mean(d[k] for d in row["native"]["draw_scores"])
                    for k in ("ED", "CR", "WAC")
                ]
                if dimension == "text"
                else row["native"]["score"]
            )
            for tag in tags:
                if tag:
                    breakdown[tag].append(value)
        result[dimension]["categories"] = {
            k: {
                "count": len(v),
                "score": [mean(x[i] for x in v) for i in range(3)]
                if dimension == "text"
                else mean(v),
            }
            for k, v in breakdown.items()
        }
        if dimension == "style":
            styles = defaultdict(list)
            for row in selected:
                styles[row["native"]["style"]].append(row["native"]["score"])
            result[dimension]["families"] = {
                family: mean(mean(styles[s]) for s in names if s in styles)
                for family, names in STYLE_FAMILIES.items()
            }
    return result
