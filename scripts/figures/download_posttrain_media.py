import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import swanlab
from PIL import Image

from download_posttrain_swanlab import RUNS


TARGETS = {
    "posttrain-110": None,
    "posttrain-600": None,
    "posttrain_specialized": None,
    "selection-e0-art": [1000],
    "selection-e0-nft": [800],
    "selection-e23-full-3e-3": [700],
    "selection-e23-lora-r64-1e-3": [1000],
}
OUTPUT = Path(__file__).resolve().parent / "data" / "posttrain" / "media"


def request_json(run, keys, step):
    for attempt in range(3):
        try:
            return run.medias(keys, step=step) if step is not None else run.medias(keys, all=True)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def download(session, task):
    url, path = task
    if path.exists():
        return path, "kept"
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(response.content)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return path, "downloaded"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def collect(api, session, name, steps):
    project, run_id = RUNS[name]
    run = api.run(f"{project}/{run_id}")
    keys = sorted(run.series(metric_type="MEDIA").json()["keys"])
    responses = [request_json(run, keys, None)] if steps is None else [
        request_json(run, keys, step) for step in steps
    ]
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    tasks = []
    for response in responses:
        for series in response["list"]:
            scene = series["key"].rsplit("/", 1)[-1]
            for point in series.get("metrics", []):
                for item in point.get("items", []):
                    path = directory / f"{point['index']:05d}-{scene}.jpg"
                    tasks.append((item["url"], path))
                    manifest.append(
                        {
                            "step": point["index"],
                            "scene": scene,
                            "file": path.name,
                            "caption": item.get("caption", ""),
                        }
                    )
    counts = {"downloaded": 0, "kept": 0}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _, status in executor.map(lambda task: download(session, task), tasks):
            counts[status] += 1
    (directory / "manifest.json").write_text(
        json.dumps({"run": run.url, "images": manifest}, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"{name}: {len(tasks)} panels | {counts} | {directory}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    api = swanlab.Api()
    with requests.Session() as session:
        for name, steps in TARGETS.items():
            collect(api, session, name, steps)


if __name__ == "__main__":
    main()
