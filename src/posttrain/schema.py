"""Strict decoding for explicit posttrain configuration and wire records."""

import json
import math
import re
from dataclasses import is_dataclass
from types import UnionType
from typing import get_args, get_origin, get_type_hints


def decode(kind, value, location="config"):
    if kind is object:
        # Opaque structured rubric/diagnostic JSON stays data, never a reward.
        try:
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{location}: expected finite JSON data") from exc
        return value
    origin, args = get_origin(kind), get_args(kind)
    if is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected table")
        hints = get_type_hints(kind)
        if set(value) != set(hints):
            raise ValueError(
                f"{location}: missing={sorted(hints.keys() - value.keys())}, unknown={sorted(value.keys() - hints.keys())}"
            )
        return kind(
            **{k: decode(t, value[k], f"{location}.{k}") for k, t in hints.items()}
        )
    if origin is UnionType and type(None) in args:
        return (
            None
            if value is None
            else decode(next(t for t in args if t is not type(None)), value, location)
        )
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{location}: expected array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], v, location) for v in value)
        if len(value) != len(args):
            raise ValueError(f"{location}: incorrect array length")
        return tuple(decode(t, v, location) for t, v in zip(args, value, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected table")
        return {
            decode(args[0], k, location): decode(args[1], v, location)
            for k, v in value.items()
        }
    if kind is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{location}: expected finite number")
        return float(value)
    if type(value) is not kind:
        raise ValueError(f"{location}: expected {kind.__name__}")
    if kind is str and "${" in value:
        raise ValueError(f"{location}: environment templates are unsupported")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, label="digest"):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        f"{label}: invalid digest",
    )


def name(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value),
        "expected safe nonempty identifier",
    )
