"""Catalog joins, native layouts, dependencies and interrupted scoring."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.evaluation.benchmarks import dpg, geneval, oneig, qwen_image, run, tiif
from src.evaluation.benchmarks.common import grid, join_images, prompt, write_json
from src.evaluation.records import append_jsonl, read_jsonl


def images(tmp_path, row):
    samples = []
    for draw, seed in enumerate(row["seeds"]):
        path = tmp_path / f"{draw}.png"
        Image.new("RGB", (3, 2), (draw * 50, 20, 30)).save(path)
        samples.append(
            {
                **{k: row[k] for k in ("benchmark", "split", "prompt_id", "variant")},
                "model": "model",
                "draw": draw,
                "seed": seed,
                "width": 3,
                "height": 2,
                "image_path": path.name,
            }
        )
    return samples


def test_join_uses_native_keys_and_draw_order(tmp_path):
    row = prompt("oneig", "en", "anime/007", "scene", {}, [13, 21, 35, 58])
    samples = images(tmp_path, row)
    groups = join_images([row], samples[::-1], "model", tmp_path)
    assert [s["seed"] for s in groups[0]["samples"]] == [13, 21, 35, 58]
    grid(groups[0]["samples"], tmp_path / "grid.png")
    with Image.open(tmp_path / "grid.png") as image:
        assert image.size == (6, 4)
        assert [image.getpixel(p) for p in [(0, 0), (3, 0), (0, 2), (3, 2)]] == [
            (0, 20, 30),
            (50, 20, 30),
            (100, 20, 30),
            (150, 20, 30),
        ]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "size", "variant"])
def test_incomplete_or_mismatched_images_cannot_score(tmp_path, defect):
    row = prompt("tiif", "testmini", "text/1", "scene", {}, [3], "long")
    samples = images(tmp_path, row)
    if defect == "missing":
        samples.clear()
    elif defect == "duplicate":
        samples *= 2
    elif defect == "seed":
        samples[0]["seed"] = 99
    elif defect == "size":
        samples[0]["width"] = 4
    else:
        samples[0]["variant"] = "short"
    with pytest.raises(ValueError):
        join_images([row], samples, "model", tmp_path)


def test_geneval_native_metadata_and_macro_average(tmp_path):
    row = prompt(
        "geneval",
        "original",
        "7",
        "a cat",
        {"native": {"prompt": "a cat", "tag": "single_object"}},
        [4],
    )
    groups = join_images([row], images(tmp_path, row), "model", tmp_path)
    geneval.layout(groups, tmp_path / "native")
    assert json.loads((tmp_path / "native/00007/metadata.jsonl").read_text()) == {
        "prompt": "a cat",
        "tag": "single_object",
    }
    assert (tmp_path / "native/00007/samples/0000.png").read_bytes() == (
        tmp_path / "0.png"
    ).read_bytes()
    result = geneval.summarize(
        [
            {
                "native": [
                    {"tag": "single_object", "correct": True},
                    {"tag": "single_object", "correct": True},
                    {"tag": "counting", "correct": False},
                ]
            }
        ],
        {},
    )
    assert result["overall"] == 0.5


def test_dpg_keeps_first_question_and_propagates_dependencies(tmp_path):
    directory = tmp_path / "dpg_bench"
    directory.mkdir()
    (directory / "dpg_bench.csv").write_text(
        "item_id,proposition_id,dependency\nx,1,0\nx,2,1\nx,3,2\n"
    )
    questions = dpg.questions(tmp_path)["x"]
    assert [q["proposition_id"] for q in questions] == ["1", "2", "3"]
    assert dpg.apply_dependencies(["no", "yes", "yes"], questions) == {1: 0, 2: 0, 3: 0}


def test_oneig_parents_use_raw_answers():
    answers = {"1": ["No", "Yes", "Yes", "Yes"], "2": ["Yes"] * 4, "3": ["Yes"] * 4}
    assert oneig.alignment_values(answers, {"1": [0], "2": [1], "3": [2]}) == [
        1 / 3,
        1,
        1,
        1,
    ]


def test_oneig_style_families_weight_styles_equally():
    rows = [
        {
            "dimension": "style",
            "prompt": {
                "metadata": {
                    "native": {
                        "category": "Anime_Stylization",
                        "prompt_length": "short",
                        "type": "T, P",
                        "class": style,
                    }
                }
            },
            "native": {"style": style, "score": score},
        }
        for style, score in [
            ("art_nouveau", 0),
            ("art_nouveau", 0),
            ("impressionism", 1),
        ]
    ]
    scores = oneig.summarize(rows, {})["style"]
    assert scores["score"] == 1 / 3
    assert scores["families"]["traditional"] == 0.5


def test_tiif_preserves_local_index_and_pairs_seed(tmp_path):
    for directory in ("testmini_prompts", "testmini_eval_prompts"):
        (tmp_path / "data" / directory).mkdir(parents=True)
    for kind in ("text", "style"):
        append_jsonl(
            tmp_path / f"data/testmini_prompts/{kind}_prompts.jsonl",
            [
                {
                    "type": kind,
                    "short_description": 'Write "tea", "cake".',
                    "long_description": "long",
                }
            ],
        )
        append_jsonl(
            tmp_path / f"data/testmini_eval_prompts/{kind}_eval_prompts.jsonl",
            [
                {
                    "type": kind,
                    "yn_question_list": ["Visible?"],
                    "yn_answer_list": ["Yes"],
                }
            ],
        )
    rows = tiif.prepare({"source": str(tmp_path)}, {"seed": 17})
    assert [(r["prompt_id"], r["variant"], r["seeds"]) for r in rows] == [
        ("style/0", "short", [17]),
        ("style/0", "long", [17]),
        ("text/0", "short", [18]),
        ("text/0", "long", [18]),
    ]
    assert rows[-1]["metadata"]["text"] == ["tea", "cake"]


def test_tiif_weights_questions_within_attribute_then_groups():
    rows = []
    for attr, answers in [
        ("shape+color", ["no", "no"]),
        ("shape+texture", ["yes"]),
        ("style", ["yes"] * 9),
    ]:
        rows.append(
            {
                "prompt": {"variant": "short"},
                "dimension": "vlm",
                "native": {
                    "attribute": attr,
                    "gt_answers": ["yes"] * len(answers),
                    "model_pred": answers,
                },
            }
        )
    scores = tiif.summarize(rows, {})["short"]
    assert scores["groups"]["Basic_Attribute"] == 0.5
    assert scores["overall"] == 0.75


def test_qwen_preserves_id_and_explicit_english_field(tmp_path):
    (tmp_path / "metadata").mkdir()
    write_json(
        tmp_path / "metadata/bench_metadata.json",
        [{"ID": 917, "dims_en": "Quality / Detail / Noise"}],
    )
    append_jsonl(
        tmp_path / "data.jsonl",
        [
            {
                "ID": 917,
                "prompt_en": "English",
                "prompt_cn": "中文",
                "dims_en": "Quality / Detail / Noise",
            }
        ],
    )
    rows = qwen_image.prepare(
        {"source": str(tmp_path), "data": str(tmp_path / "data.jsonl")}, {"seed": 5}
    )
    assert (rows[0]["prompt_id"], rows[0]["text"], rows[0]["variant"]) == (
        "917",
        "English",
        "en",
    )


def test_scoring_restart_reuses_published_judgments(tmp_path, monkeypatch):
    groups = [
        {
            "prompt": prompt("geneval", "original", str(i), "scene", {}, [i]),
            "samples": [{"model": "model", "draw": 0}],
        }
        for i in range(3)
    ]
    request = tmp_path / "request.json"
    write_json(
        request,
        {
            "benchmark": "geneval",
            "settings": {},
            "assets": {},
            "groups": groups,
            "model": "model",
        },
    )
    seen = []

    def interrupted(pending, assets, settings, root, emit):
        emit.loaded()
        seen.extend(g["prompt"]["prompt_id"] for g in pending)
        emit(pending[0], "detector", {"correct": False})
        raise InterruptedError

    monkeypatch.setattr(run, "adapter", lambda _: SimpleNamespace(evaluate=interrupted))
    monkeypatch.setattr("sys.argv", ["worker", str(request), "--dimension", "detector"])
    with pytest.raises(InterruptedError):
        run.main()
    assert seen == ["0", "1", "2"]
    published = (tmp_path / "scores.jsonl").read_bytes()
    with (tmp_path / "scores.jsonl").open("ab") as stream:
        stream.write(b'{"interrupted":')

    def finish(pending, assets, settings, root, emit):
        emit.loaded()
        assert [g["prompt"]["prompt_id"] for g in pending] == ["1", "2"]
        for group in pending:
            emit(group, "detector", {"correct": True})

    monkeypatch.setattr(run, "adapter", lambda _: SimpleNamespace(evaluate=finish))
    run.main()
    assert (tmp_path / "scores.jsonl").read_bytes().startswith(published)
    assert len(read_jsonl(tmp_path / "scores.jsonl")) == 3


def test_upstream_processes_read_released_utf8_text(tmp_path, monkeypatch):
    row = prompt("geneval", "original", "2", "a blue café", {}, [3])
    source = tmp_path / "images"
    directory = source / "model/_geneval/_original"
    directory.mkdir(parents=True)
    append_jsonl(directory / "prompts.jsonl", [row])
    append_jsonl(directory / "samples.jsonl", images(source, row))
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setattr(
        run, "adapter", lambda _: SimpleNamespace(layout=lambda groups, root: None)
    )
    calls = []

    def subprocess(command, *, env, check):
        assert env["PYTHONUTF8"] == "1"
        calls.append(command[-1])
        request = json.loads((tmp_path / "scores/request.json").read_text())
        if "--dimension" in command:
            append_jsonl(
                tmp_path / "scores/scores.jsonl",
                [{**request["groups"][0], "dimension": "detector", "native": []}],
            )
        else:
            write_json(tmp_path / "scores/summary.json", {"evaluated": 1})

    monkeypatch.setattr(run.subprocess, "run", subprocess)
    result = run.score(
        "geneval", {}, {"python": "/judge/python"}, "model", source, tmp_path / "scores"
    )
    assert result == {"evaluated": 1}
    assert calls == ["detector", "--summarize"]
