"""Inko's painting demo, with session history and reproducible image recipes."""

import html
import json
import math
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

# ZeroGPU must initialize before modules that import torch.
import spaces  # isort: split

import gradio as gr
import torch

from demo.generation import (
    CANVASES,
    HISTORY_LIMIT,
    append_history,
    canvas_selection,
    delete_results,
    enhance_prompt,
    fallback_canvas,
    read_record,
    sampling_settings,
    save_image,
    validate_dimensions,
    validate_prompt,
)
from src.pipeline import InkoPipeline

MODEL_ID = os.environ.get("INKO_MODEL_ID")
if not MODEL_ID:
    raise RuntimeError(
        "Set INKO_MODEL_ID to the final RL EMA export directory or repository."
    )
DEVICE = os.environ.get("INKO_DEVICE", "cuda")
pipe = InkoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
).to(DEVICE)

HERE = Path(__file__).resolve().parent
SHOWCASE_DIR = HERE / "showcase"
SHOWCASE = json.loads((SHOWCASE_DIR / "prompts.json").read_text())
RESULT_DIR = Path(tempfile.mkdtemp(prefix="inko-results-"))
gr.set_static_paths(paths=[SHOWCASE_DIR, RESULT_DIR])
PROGRESS = gr.Progress()
ACTIVITY_INDICATOR = (
    '<div class="activity-indicator" role="progressbar" '
    'aria-label="Request in progress"><span></span></div>'
)


def image_url(path):
    return "/gradio_api/file=" + quote(str(Path(path).resolve()), safe="/")


def carousel_html():
    cards = "".join(
        f'<button type="button" class="painting-card" data-index="{index}" '
        f'aria-label="Use {html.escape(item["title"], quote=True)}">'
        f'<img src="{image_url(SHOWCASE_DIR / item["thumbnail"])}" '
        f'width="{round(164 * item["width"] / item["height"])}" height="164" '
        f'alt="{html.escape(item["title"], quote=True)}" draggable="false">'
        f"<span>{html.escape(item['title'])}</span></button>"
        for index, item in enumerate(SHOWCASE)
    )
    duplicate = cards.replace('type="button"', 'type="button" tabindex="-1"')
    return (
        '<div class="showcase-strip" aria-label="Painting examples">'
        f'<div class="showcase-track"><div class="showcase-group">{cards}</div>'
        f'<div class="showcase-group" aria-hidden="true">{duplicate}</div></div></div>'
        '<div class="showcase-caption"><span>Choose a painting to load its prompt and settings.</span>'
        '<button type="button" class="motion-toggle" aria-label="Pause carousel">'
        '<span class="motion-icon" aria-hidden="true">Ⅱ</span>'
        '<span class="motion-label">Pause</span></button></div>'
    )


def history_html(records, selected_path=None):
    if not records:
        return '<p class="history-empty">Your recent generations will appear here.</p>'
    cards = "".join(
        f'<button type="button" class="history-card" data-index="{index}" '
        f'aria-pressed="{str(record["image"] == selected_path).lower()}" '
        f'aria-label="Generation {index + 1}, seed {record["seed"]}">'
        f'<img src="{image_url(record["image"])}" alt="Generation {index + 1}" '
        f'loading="lazy" draggable="false"></button>'
        for index, record in enumerate(records)
    )
    return (
        f'<div class="history-heading">Your generations <span>{len(records)} / {HISTORY_LIMIT}</span></div>'
        f'<div class="history-strip" aria-label="Generation history">{cards}</div>'
    )


def recipe_text(record):
    return (
        f"**{record['kind']}** · {record['width']} × {record['height']} · "
        f"{record['steps']} steps · CFG {record['guidance']:g} · Seed {record['seed']}"
    )


def canvas_ui(aspect, resolution, width, height, enhance):
    try:
        w, h = fallback_canvas(aspect, resolution, width, height)
    except ValueError as error:
        return (
            gr.update(visible=aspect == "Custom"),
            gr.update(visible=aspect != "Custom"),
            str(error),
        )
    if enhance and aspect != "Custom" and "Auto" in (aspect, resolution):
        text = "Auto canvas · chosen during enhancement"
    else:
        text = f"{w} × {h} pixels"
    return (
        gr.update(visible=aspect == "Custom"),
        gr.update(visible=aspect != "Custom"),
        text,
    )


def interaction_updates(is_busy):
    updates = {
        control: gr.update(interactive=not is_busy) for control in input_controls
    }
    if not is_busy:
        updates[generate_button] = gr.update(value="Generate", interactive=True)
    return updates


def prepare_request(
    prompt,
    enhance,
    aspect,
    resolution,
    width,
    height,
    seed,
    randomize_seed,
    steps,
    guidance,
):
    """Enhance a description and resolve an explicit, validated generation request."""
    yield {
        busy: True,
        status: "Enhancing prompt…" if enhance else "Preparing your painting…",
        progress_display: ACTIVITY_INDICATOR,
        reuse_button: gr.update(interactive=False),
        delete_button: gr.update(interactive=False),
        **interaction_updates(True),
        generate_button: gr.update(
            value="Enhancing…" if enhance else "Preparing…", interactive=False
        ),
    }
    try:
        sampling = sampling_settings(seed, randomize_seed, steps, guidance)
        prepared = enhance_prompt(
            prompt, enhance, aspect, resolution, width, height, pipe.tokenizer
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    yield {
        pending: prepared | sampling,
        status: "Generating image…",
        generate_button: gr.update(value="Generating…", interactive=False),
    }


def gpu_duration(request, progress=PROGRESS):
    # H200 panel timings at 1600/2048 leave room for startup and VAE decoding.
    area_scale = (request["width"] * request["height"] / 896**2) ** 2
    forwards = request["steps"] * (1 if request["guidance"] == 1 else 2)
    return 20 + math.ceil(0.025 * forwards * area_scale)


@spaces.GPU(duration=gpu_duration)
def generate(request: dict, progress=PROGRESS) -> str:
    """Generate a PNG with embedded settings from a prepared prompt/canvas request."""
    try:
        prompt = validate_prompt(request["prompt"], pipe.tokenizer)
        width, height = validate_dimensions(request["width"], request["height"])
        sampling = sampling_settings(
            request["seed"], False, request["steps"], request["guidance"]
        )
    except (ValueError, KeyError, TypeError) as error:
        raise gr.Error(str(error)) from error
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=sampling["steps"],
        guidance_scale=sampling["guidance"],
        generator=torch.Generator(device="cpu").manual_seed(sampling["seed"]),
        progress_callback=lambda current, total: progress(
            (current, total), desc="Generating image"
        ),
    ).images[0]
    record = {
        "prompt": prompt,
        "width": width,
        "height": height,
        **sampling,
        "model": MODEL_ID,
        "sampler": "Euler",
        "notice": request.get("notice", ""),
    }
    return save_image(image, record, RESULT_DIR)


def display_updates(record, records):
    return {
        selected: record,
        output: record["image"],
        used_prompt: record["prompt"],
        recipe: gr.update(value=recipe_text(record), visible=True),
        result_notice: gr.update(
            value=record.get("notice", ""), visible=bool(record.get("notice"))
        ),
        history_gallery: history_html(records, record["image"]),
        reuse_button: gr.update(interactive=True),
        delete_button: gr.update(
            visible=record["kind"] == "Generated", interactive=True
        ),
    }


def commit_result(path, records):
    record = read_record(path, RESULT_DIR)
    updated = append_history(records, record, RESULT_DIR)
    return {
        **display_updates(record, updated),
        history: updated,
        busy: False,
        status: "",
        progress_display: "",
        **interaction_updates(False),
        seed: gr.update(value=record["seed"], interactive=True),
        width: gr.update(value=record["width"], interactive=True),
        height: gr.update(value=record["height"], interactive=True),
        canvas_label: f"{record['width']} × {record['height']} pixels",
    }


def restore_record(record, records):
    ratio, size = canvas_selection(record["width"], record["height"])
    return {
        **display_updates(record, records),
        prompt: record["prompt"],
        enhance: False,
        aspect: ratio,
        resolution: gr.update(value=size, visible=ratio != "Custom"),
        custom_canvas: gr.update(visible=ratio == "Custom"),
        width: record["width"],
        height: record["height"],
        seed: record["seed"],
        randomize_seed: False,
        steps: record["steps"],
        guidance: record["guidance"],
        canvas_label: f"{record['width']} × {record['height']} pixels",
        status: "",
    }


def select_example(is_busy, records, event: gr.SelectData):
    if is_busy:
        raise gr.Error(
            "Wait for the current generation before selecting another image."
        )
    index = event.index
    if type(index) is not int or not 0 <= index < len(SHOWCASE):
        raise gr.Error("Select an available example.")
    item = SHOWCASE[index]
    record = item | {"image": str(SHOWCASE_DIR / item["file"]), "kind": "Example"}
    return restore_record(record, records)


def select_history(is_busy, records, event: gr.SelectData):
    if is_busy:
        raise gr.Error(
            "Wait for the current generation before selecting another image."
        )
    index = event.index
    if type(index) is not int or not 0 <= index < len(records):
        raise gr.Error("Select an available generation.")
    return restore_record(records[index], records)


def delete_selected(record, is_busy, records):
    if is_busy:
        raise gr.Error("Wait for the current generation before deleting an image.")
    index = next(
        (
            index
            for index, item in enumerate(records)
            if record is not None and item["image"] == record["image"]
        ),
        None,
    )
    if index is None:
        raise gr.Error("Select an image from your generation history to delete.")
    updated = records[:index] + records[index + 1 :]
    if updated:
        changes = restore_record(updated[min(index, len(updated) - 1)], updated)
    else:
        changes = {
            selected: None,
            output: None,
            used_prompt: "",
            recipe: gr.update(value="", visible=False),
            result_notice: gr.update(value="", visible=False),
            history_gallery: history_html([]),
            reuse_button: gr.update(interactive=False),
            delete_button: gr.update(visible=False, interactive=False),
            status: "",
        }
    delete_results([records[index]], RESULT_DIR)
    return {**changes, history: updated}


def reuse_prompt(record, is_busy):
    if is_busy or record is None:
        return {prompt: gr.skip()}
    ratio, size = canvas_selection(record["width"], record["height"])
    return {
        prompt: record["prompt"],
        enhance: False,
        aspect: ratio,
        resolution: gr.update(value=size, visible=ratio != "Custom"),
        custom_canvas: gr.update(visible=ratio == "Custom"),
        width: record["width"],
        height: record["height"],
        canvas_label: f"{record['width']} × {record['height']} pixels",
    }


def failed_request(record):
    return {
        busy: False,
        status: "The request failed. You can try again with the same settings.",
        progress_display: "",
        reuse_button: gr.update(interactive=record is not None),
        delete_button: gr.update(interactive=record is not None),
        **interaction_updates(False),
    }


def clear_history(records):
    delete_results(records, RESULT_DIR)


with gr.Blocks(title="Inko", delete_cache=(3600, 3600)) as demo:
    history = gr.State([], delete_callback=clear_history)
    selected = gr.State(None)
    busy = gr.State(False)
    pending = gr.JSON(visible=False)
    completed = gr.File(visible=False, type="filepath")
    gr.HTML(
        '<header class="inko-header"><div><h1>Inko<span class="ink-dot">.</span></h1>'
        "<p>A 0.5B diffusion transformer for paintings.</p></div>"
        '<nav aria-label="Project links"><a href="https://huggingface.co/kaupane/Inko" '
        'target="_blank" rel="noopener noreferrer">Model ↗</a>'
        '<a href="https://github.com/Mtrya/inko" target="_blank" rel="noopener noreferrer">Code ↗</a>'
        "</nav></header>",
        container=False,
        apply_default_css=False,
        elem_id="inko-brand",
    )
    showcase = gr.HTML(
        carousel_html(),
        js_on_load=(HERE / "gallery.js").read_text(),
        elem_id="showcase-carousel",
        apply_default_css=False,
    )
    with gr.Row(equal_height=False, elem_id="workspace"):
        with gr.Column(scale=5, min_width=300, elem_id="prompt-panel"):
            prompt = gr.Textbox(
                label="Your prompt",
                lines=5,
                max_lines=12,
                placeholder="Describe your image…",
            )
            enhance = gr.Checkbox(
                value=True,
                label="Enhance prompt with DeepSeek",
                info="Adds visual detail and chooses Auto canvas settings. Your prompt is sent to DeepSeek.",
            )
            aspect = gr.Radio(
                ["Auto", *CANVASES["Standard"], "Custom"],
                value="Auto",
                label="Aspect ratio",
            )
            resolution = gr.Radio(
                ["Auto", *CANVASES],
                value="Auto",
                label="Resolution",
                info="Standard ≈ 896² pixels · Large ≈ 1280² pixels",
            )
            with gr.Row(visible=False) as custom_canvas:
                width = gr.Slider(480, 1920, value=896, step=16, label="Width (pixels)")
                height = gr.Slider(
                    480, 1920, value=896, step=16, label="Height (pixels)"
                )
            canvas_label = gr.Markdown(
                "Auto canvas · chosen during enhancement", elem_classes=["canvas-note"]
            )
            generate_button = gr.Button(
                "Generate", variant="primary", elem_id="generate-button"
            )
            status = gr.Markdown("", elem_id="generation-status")
            progress_display = gr.HTML(
                "", elem_id="generation-progress", apply_default_css=False
            )
            with gr.Accordion("Generation settings", open=False):
                randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
                seed = gr.Number(
                    value=42, minimum=0, maximum=2**31 - 1, precision=0, label="Seed"
                )
                steps = gr.Slider(
                    10,
                    50,
                    value=pipe.config["num_inference_steps"],
                    step=1,
                    label="Steps",
                )
                guidance = gr.Slider(
                    0, 10, value=pipe.config["guidance_scale"], step=0.1, label="CFG"
                )
        with gr.Column(scale=7, min_width=300, elem_id="result-panel"):
            output = gr.Image(
                label="Selected painting",
                show_label=False,
                type="filepath",
                format="png",
                interactive=False,
                height="58vh",
                buttons=["download", "fullscreen"],
                placeholder="Your painting will appear here.",
                elem_id="result-viewer",
            )
            recipe = gr.Markdown("", visible=False, elem_classes=["result-recipe"])
            result_notice = gr.Markdown(
                "", visible=False, elem_classes=["result-notice"]
            )
            history_gallery = gr.HTML(
                history_html([]),
                js_on_load=(HERE / "gallery.js").read_text(),
                elem_id="result-history",
                apply_default_css=False,
            )
            delete_button = gr.Button(
                "Delete image", size="sm", visible=False, elem_id="delete-image"
            )
            used_prompt = gr.Textbox(
                label="Prompt used",
                lines=4,
                max_lines=8,
                interactive=False,
                buttons=["copy"],
            )
            reuse_button = gr.Button("Use this prompt", interactive=False, size="sm")

    input_controls = [
        prompt,
        enhance,
        aspect,
        resolution,
        width,
        height,
        randomize_seed,
        seed,
        steps,
        guidance,
        generate_button,
    ]
    request_inputs = [
        prompt,
        enhance,
        aspect,
        resolution,
        width,
        height,
        seed,
        randomize_seed,
        steps,
        guidance,
    ]
    result_outputs = [
        selected,
        output,
        used_prompt,
        recipe,
        result_notice,
        history_gallery,
        reuse_button,
        delete_button,
    ]
    restore_outputs = [
        *result_outputs,
        *input_controls,
        custom_canvas,
        canvas_label,
        status,
    ]
    busy_outputs = [
        busy,
        status,
        progress_display,
        reuse_button,
        delete_button,
        *input_controls,
    ]
    gr.on(
        triggers=[
            aspect.input,
            resolution.input,
            width.input,
            height.input,
            enhance.input,
        ],
        fn=canvas_ui,
        inputs=[aspect, resolution, width, height, enhance],
        outputs=[custom_canvas, resolution, canvas_label],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    showcase.select(
        select_example,
        inputs=[busy, history],
        outputs=restore_outputs,
        queue=False,
        api_visibility="private",
    )
    history_gallery.select(
        select_history,
        inputs=[busy, history],
        outputs=restore_outputs,
        queue=False,
        api_visibility="private",
    )
    delete_button.click(
        delete_selected,
        inputs=[selected, busy, history],
        outputs=[*restore_outputs, history],
        queue=False,
        api_visibility="private",
    )
    reuse_button.click(
        reuse_prompt,
        inputs=[selected, busy],
        outputs=[
            prompt,
            enhance,
            aspect,
            resolution,
            custom_canvas,
            width,
            height,
            canvas_label,
        ],
        queue=False,
        api_visibility="private",
    )
    preparation = gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=prepare_request,
        inputs=request_inputs,
        outputs=[pending, *busy_outputs],
        api_name="prepare_prompt",
        trigger_mode="once",
        show_progress="full",
        show_progress_on=progress_display,
    )
    generation = preparation.success(
        generate,
        inputs=pending,
        outputs=completed,
        api_name="generate",
        show_progress="full",
        show_progress_on=progress_display,
    )
    committed = generation.success(
        commit_result,
        inputs=[completed, history],
        outputs=list(
            dict.fromkeys(
                [
                    *result_outputs,
                    history,
                    *busy_outputs,
                    width,
                    height,
                    seed,
                    canvas_label,
                ]
            )
        ),
        api_visibility="private",
        show_progress="hidden",
    )
    for event in (preparation, generation, committed):
        event.failure(
            failed_request,
            inputs=selected,
            outputs=busy_outputs,
            api_visibility="private",
            show_progress="hidden",
        )


def launch():
    colors = {
        "body_background_fill": "#f4f0e8",
        "body_text_color": "#322d28",
        "body_text_color_subdued": "#766b60",
        "block_background_fill": "#fffcf7",
        "block_border_color": "#ded6c9",
        "block_label_text_color": "#62584e",
        "block_label_background_fill": "transparent",
        "block_title_text_color": "#322d28",
        "input_background_fill": "#fffdf9",
        "input_border_color": "#ded6c9",
        "input_placeholder_color": "#94897b",
        "button_primary_background_fill": "#a94f3c",
        "button_primary_background_fill_hover": "#91412f",
        "button_primary_text_color": "#fffaf3",
        "button_secondary_background_fill": "#eee7db",
        "button_secondary_background_fill_hover": "#e4d9c9",
        "button_secondary_text_color": "#493d33",
    }
    vermilion = gr.themes.Color(
        c50="#fbf0eb",
        c100="#f3dfd5",
        c200="#e6bcaa",
        c300="#d69378",
        c400="#c56d50",
        c500="#b05a42",
        c600="#a94f3c",
        c700="#91412f",
        c800="#753529",
        c900="#5b2d24",
        c950="#361a15",
        name="vermilion",
    )
    theme = gr.themes.Base(
        primary_hue=vermilion,
        neutral_hue="stone",
        font=[
            gr.themes.Font(name)
            for name in ("Inter", "ui-sans-serif", "system-ui", "sans-serif")
        ],
    ).set(**colors, **{f"{key}_dark": value for key, value in colors.items()})
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        theme=theme,
        css=(HERE / "style.css").read_text(),
        footer_links=["api"],
    )
