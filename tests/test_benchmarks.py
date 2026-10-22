"""Catalog joins, native layouts, dependencies and interrupted scoring."""

import json
import sys
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


class Emitter:
    def __init__(self):
        self.rows = []

    def loaded(self):
        pass

    def __call__(self, group, dimension, result):
        self.rows.append((dimension, result))


def tiif_vlm_group():
    return {
        "prompt": {
            "metadata": {
                "type": "style",
                "index": 0,
                "questions": {
                    "yn_question_list": ["Q1?", "Q2?"],
                    "yn_answer_list": ["Yes", "No"],
                },
            },
            "seeds": [3],
            "variant": "short",
        },
        "samples": [{"model": "model", "image_path": "image.png"}],
    }


def stub_tiif_judge(monkeypatch, failures):
    class FormatError(Exception):
        pass

    calls = []

    def generate(text, image, client, model):
        calls.append(text)
        return f"response-{len(calls)}"

    def extract(response, questions):
        if len(calls) <= failures:
            raise FormatError("mismatch")
        return ["yes"] * len(questions)

    native = SimpleNamespace(
        raw_prompt="template",
        OutputFormatError=FormatError,
        format_questions_prompt=lambda template, questions: template,
        generate_with_prompt=generate,
        extract_yes_no=extract,
    )
    monkeypatch.setattr(tiif, "load_source", lambda _: native)
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: None)
    )
    return FormatError, calls


def test_tiif_retries_judge_format_failures_with_stable_prompt(monkeypatch, tmp_path):
    _, calls = stub_tiif_judge(monkeypatch, failures=2)
    emitter = Emitter()
    tiif.evaluate(
        [tiif_vlm_group()],
        {"source": str(tmp_path), "base_url": "http://localhost"},
        {"dimension": "vlm", "judge": "judge"},
        tmp_path / "scores",
        emitter,
    )
    assert calls == ["template"] * 3
    assert emitter.rows[0][1]["model_pred"] == ["yes", "yes"]


def test_tiif_gives_up_after_bounded_retries(monkeypatch, tmp_path):
    error, calls = stub_tiif_judge(monkeypatch, failures=100)
    with pytest.raises(error):
        tiif.evaluate(
            [tiif_vlm_group()],
            {"source": str(tmp_path), "base_url": "http://localhost"},
            {"dimension": "vlm", "judge": "judge"},
            tmp_path / "scores",
            Emitter(),
        )
    assert len(calls) == 8


def qwen_vlm_group():
    return {
        "prompt": {
            "metadata": {"ID": 44, "dims_en": "Quality / Detail / Noise"},
            "text": "A medical science infographic poster.",
        },
        "samples": [{"model": "model", "image_path": "image.png"}],
    }


QWEN_QUALITY = {"Detail": {"Noise": {"score": 2}}}
QWEN_ALIGNMENT = {"Attributes": {"Quantity": {"score": 1}}}
QWEN_COLUMNS = {
    "Quality": "quality_judge_output",
    "Alignment": "alignment_judge_output",
}


def qwen_result(scores):
    parsed = {
        dimension: score for dimension, score in scores.items() if score is not None
    }
    return {
        "judge_model_output": json.dumps(parsed) if parsed else None,
        **{
            column: json.dumps(scores[dimension]) if dimension in scores else None
            for dimension, column in QWEN_COLUMNS.items()
        },
    }


def qwen_native_namespace(run_batch):
    numeric_scores = [
        (QWEN_QUALITY, 100.0),
        (QWEN_ALIGNMENT, 60.0),
        ({"Detail": {"Noise": {"score": 0}}}, 0.0),
    ]
    return SimpleNamespace(
        _run_batch_inference=run_batch,
        DIM_OUTPUT_MAP=QWEN_COLUMNS,
        compute_dimension_score=lambda score: {
            "level1_score": next(
                (value for candidate, value in numeric_scores if score == candidate),
                None,
            )
        },
    )


def stub_qwen_native(failures, image_failures=0, failing_scores=None):
    calls = []

    def run_batch(judge, args, inputs, meta, desc):
        calls.append(desc)
        if len(calls) <= failures:
            scores = {"Quality": None} if failing_scores is None else failing_scores
            return [qwen_result(scores)], 1, [scores], image_failures
        scores = {"Quality": QWEN_QUALITY}
        return [qwen_result(scores)], 0, [scores], image_failures

    return qwen_native_namespace(run_batch), calls


def test_qwen_retries_judge_parse_failures():
    native, calls = stub_qwen_native(failures=2)
    results = qwen_image.judge_batch(native, "judge", [qwen_vlm_group()], 4)
    assert json.loads(results[0]["judge_model_output"]) == {
        "Quality": {"Detail": {"Noise": {"score": 2}}}
    }
    assert calls == ["Q-Judger"] * 3


def test_qwen_gives_up_after_bounded_retries():
    native, calls = stub_qwen_native(failures=100)
    with pytest.raises(ValueError, match="failed to evaluate IDs"):
        qwen_image.judge_batch(native, "judge", [qwen_vlm_group()], 4)
    assert len(calls) == 8


def test_qwen_tolerates_rows_with_partially_parsed_dimensions():
    scores = {"Quality": None, "Alignment": QWEN_ALIGNMENT}
    native, calls = stub_qwen_native(failures=100, failing_scores=scores)
    results = qwen_image.judge_batch(native, "judge", [qwen_vlm_group()], 4)
    assert json.loads(results[0]["judge_model_output"]) == {
        "Alignment": {"Attributes": {"Quantity": {"score": 1}}}
    }
    assert len(calls) == 8


@pytest.mark.parametrize(
    "scores", [{"Quality": {}}, {"Quality": {"Detail": {"Noise": {"score": "N/A"}}}}]
)
def test_qwen_rejects_rows_without_aggregatable_scores(scores):
    native, calls = stub_qwen_native(failures=100, failing_scores=scores)
    with pytest.raises(ValueError, match="failed to evaluate IDs"):
        qwen_image.judge_batch(native, "judge", [qwen_vlm_group()], 4)
    assert len(calls) == 8


def test_qwen_preserves_scores_and_raw_outputs_across_attempts():
    attempts = iter(
        [
            {"Quality": QWEN_QUALITY, "Alignment": None},
            {"Quality": None, "Alignment": QWEN_ALIGNMENT},
        ]
    )

    def run_batch(*args):
        scores = next(attempts)
        return [qwen_result(scores)], 1, [scores], 0

    results = qwen_image.judge_batch(
        qwen_native_namespace(run_batch), "judge", [qwen_vlm_group()], 4
    )
    assert json.loads(results[0]["judge_model_output"]) == {
        "Quality": {"Detail": {"Noise": {"score": 2}}},
        "Alignment": {"Attributes": {"Quantity": {"score": 1}}},
    }
    assert json.loads(results[0]["quality_judge_output"]) == QWEN_QUALITY
    assert json.loads(results[0]["alignment_judge_output"]) == QWEN_ALIGNMENT


def test_qwen_keeps_the_first_successful_score_when_later_attempts_rescore_it():
    attempts = iter(
        [
            {"Quality": QWEN_QUALITY, "Alignment": None},
            {
                "Quality": {"Detail": {"Noise": {"score": 0}}},
                "Alignment": QWEN_ALIGNMENT,
            },
        ]
    )

    def run_batch(*args):
        scores = next(attempts)
        return [qwen_result(scores)], 1, [scores], 0

    results = qwen_image.judge_batch(
        qwen_native_namespace(run_batch), "judge", [qwen_vlm_group()], 4
    )
    assert json.loads(results[0]["judge_model_output"])["Quality"] == {
        "Detail": {"Noise": {"score": 2}}
    }
    assert json.loads(results[0]["quality_judge_output"]) == QWEN_QUALITY


def test_qwen_merges_rows_independently_and_accepts_zero_scores():
    zero = {"Detail": {"Noise": {"score": 0}}}
    attempts = iter(
        [
            [{"Quality": QWEN_QUALITY}, {"Quality": None}],
            [{"Quality": zero}, {"Quality": zero}],
        ]
    )

    def run_batch(*args):
        scores = next(attempts)
        return [qwen_result(row) for row in scores], 1, scores, 0

    groups = [qwen_vlm_group(), qwen_vlm_group()]
    groups[1]["prompt"]["metadata"]["ID"] = 45
    results = qwen_image.judge_batch(
        qwen_native_namespace(run_batch), "judge", groups, 4
    )
    assert [json.loads(row["judge_model_output"]) for row in results] == [
        {"Quality": {"Detail": {"Noise": {"score": 2}}}},
        {"Quality": {"Detail": {"Noise": {"score": 0}}}},
    ]


def test_qwen_does_not_retry_valid_na_dimensions_when_the_row_is_scored():
    scores = {
        "Quality": QWEN_QUALITY,
        "Alignment": {"Attributes": {"Quantity": {"score": "N/A"}}},
    }
    calls = []

    def run_batch(*args):
        calls.append(None)
        return [qwen_result(scores)], 0, [scores], 0

    results = qwen_image.judge_batch(
        qwen_native_namespace(run_batch), "judge", [qwen_vlm_group()], 4
    )
    assert json.loads(results[0]["judge_model_output"]) == {
        "Quality": {"Detail": {"Noise": {"score": 2}}}
    }
    assert len(calls) == 1


@pytest.mark.parametrize("failures", [0, 100])
def test_qwen_does_not_retry_broken_images(failures):
    native, calls = stub_qwen_native(failures=failures, image_failures=1)
    with pytest.raises(ValueError, match="failed to evaluate IDs"):
        qwen_image.judge_batch(native, "judge", [qwen_vlm_group()], 4)
    assert len(calls) == 1


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
