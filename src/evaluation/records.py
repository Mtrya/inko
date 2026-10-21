"""Prompt catalogs and image records shared by generation and scoring."""

import json
from pathlib import Path
from urllib.parse import quote


def prompt_key(row):
    return tuple(row[key] for key in ("benchmark", "split", "prompt_id", "variant"))


def sample_key(row):
    return (row["model"], *prompt_key(row), row["draw"])


def path_component(value):
    # Native IDs may include slashes; they must remain one directory component.
    return "_" + quote(str(value), safe="")


def read_jsonl(path, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r+b" if repair_tail else "rb") as stream:
        while line := stream.readline():
            end = stream.tell()
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if not repair_tail or line.endswith(b"\n") or stream.read(1):
                    raise
                stream.truncate(end - len(line))
                break
            if repair_tail and not line.endswith(b"\n"):
                stream.write(b"\n")
    return rows


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def load_catalog(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        key = prompt_key(row)
        if any(not isinstance(v, str) or not v for v in key) or key in seen:
            raise ValueError(
                "catalog requires unique benchmark/split/prompt/variant keys"
            )
        seen.add(key)
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("empty prompt text")
        if not isinstance(row["metadata"], dict):
            raise ValueError("prompt metadata must be an object")
        seeds = row["seeds"]
        if not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
            raise ValueError("each draw requires an explicit nonnegative seed")
    if not rows:
        raise ValueError("empty prompt catalog")
    return rows
