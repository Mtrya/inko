"""Single bounded HTTP transaction, isolated so DNS cannot defeat the deadline."""

import json
import sys
import urllib.error
import urllib.request


def main():
    request = json.load(sys.stdin)
    payload = request["payload"]
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    wire = urllib.request.Request(
        request["endpoint"].rstrip("/") + request["route"],
        data=data,
        headers={"Content-Type": "application/json", **request.get("headers", {})},
    )
    try:
        with opener.open(wire, timeout=request["timeout"]) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("score response too large")
            value = json.loads(raw)
        print(json.dumps(value, allow_nan=False))
    except urllib.error.HTTPError as exc:
        print(
            json.dumps(
                {"status": exc.code, "error": exc.read(4096).decode(errors="replace")}
            )
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - error crosses a process boundary
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
