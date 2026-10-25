import csv
import time
from pathlib import Path

import swanlab


RUNS = {
    "posttrain-110": ("mtrya/inko", "famios0bq9f0pp0zqa47e"),
    "posttrain-600": ("mtrya/inko", "gqontt9jhrhrzgulbkekm"),
    "posttrain_specialized": ("mtrya/inko", "m3jjy8wgkggzkfo20qkpe"),
    "selection-e0-nft": ("mtrya/inko-posttrain", "gnihnyye2m4slrgqas01p"),
    "selection-e0-art": ("mtrya/inko-posttrain", "fk4ggn2tmm83vkjtzjob4"),
    "selection-e1-anchor-on": ("mtrya/inko-posttrain", "ijr732ocoiex3syh6vg0u"),
    "selection-e23-full-1e-3": ("mtrya/inko-posttrain", "hq8vuyqop2s68nky3qk80"),
    "selection-e23-full-3e-3": ("mtrya/inko-posttrain", "0186mifk71hetfo2pengb"),
    "selection-e23-full-3e-4": ("mtrya/inko-posttrain", "h7si19qjv1ugvtamw2oh4"),
    "selection-e23-lora-3e-3": ("mtrya/inko-posttrain", "d925d5ao19b93zkkt139c"),
    "selection-e23-lora-3e-4": ("mtrya/inko-posttrain", "0e27zat6iljpz5jzex08g"),
    "selection-e23-lora-r64-1e-3": ("mtrya/inko", "uxifb4d0brv2bz6igsdqn"),
}
CHUNK_SIZE = 12
OUTPUT = Path(__file__).resolve().parent / "data" / "posttrain"


def fetch(run, keys):
    for attempt in range(3):
        try:
            return run.metrics(keys, all=True, ignore_timestamp=False)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def download(api, name, project, run_id):
    path = OUTPUT / f"{name}-scalars.csv"
    if path.exists():
        print(f"{name}: keeping {path}")
        return
    run = api.run(f"{project}/{run_id}")
    keys = sorted(run.series().json()["keys"])
    rows = 0
    covered = set()
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "step", "value", "timestamp"])
        for start in range(0, len(keys), CHUNK_SIZE):
            chunk = keys[start : start + CHUNK_SIZE]
            response = fetch(run, chunk)
            returned = {item["key"]: item.get("metrics", []) for item in response["list"]}
            for key in chunk:
                points = returned.get(key, [])
                if points:
                    covered.add(key)
                for point in points:
                    writer.writerow(
                        [key, point.get("step"), point.get("value"), point.get("timestamp")]
                    )
                    rows += 1
    temporary.replace(path)
    missing = sorted(set(keys) - covered)
    print(f"{name}: {run.url} | {len(keys)} series | {rows} points | {path}")
    if missing:
        print(f"  {len(missing)} series without points: {missing}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    api = swanlab.Api()
    for name, (project, run_id) in RUNS.items():
        download(api, name, project, run_id)


if __name__ == "__main__":
    main()
