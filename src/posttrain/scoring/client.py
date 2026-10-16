"""One collection client for remote training and evaluation scoring."""

import json
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..schema import require
from .contracts import image_payload


class HTTPError(OSError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"scoring HTTP {status}: {message}")


def http_json(endpoint, route, timeout, payload=None, *, token=None):
    # A subprocess bounds DNS, headers and a trickling response by elapsed time.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.posttrain.scoring.transport"],
            input=json.dumps(
                {
                    "endpoint": endpoint,
                    "route": route,
                    "timeout": timeout,
                    "payload": payload,
                    "headers": {"Authorization": f"Bearer {token}"} if token else {},
                },
                allow_nan=False,
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("absolute HTTP request deadline exceeded") from exc
    if result.returncode == 2:
        error = json.loads(result.stdout)
        raise HTTPError(error["status"], error["error"])
    if result.returncode:
        raise OSError(f"score transport failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def group_metadata(group_id, prompt, draws):
    group = {"id": group_id, "prompt": prompt["text"], "draws": draws}
    if "rubric" in prompt:
        group["rubric"] = prompt["rubric"]
        if prompt.get("reference"):
            group["reference"] = image_payload(Path(prompt["reference"]).read_bytes())
    return group


class ScoreClient:
    def __init__(self, config, deployment):
        self.config = config
        self.endpoint, self.token = deployment["endpoint"], deployment.get("token")
        require(
            self.endpoint.startswith(("http://", "https://")),
            "scoring endpoint required",
        )

    def call(self, route, payload=None):
        for attempt in range(self.config.attempts):
            try:
                return http_json(
                    self.endpoint, route, self.config.timeout, payload, token=self.token
                )
            except (OSError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.status not in (429, 503):
                    raise
                if attempt + 1 == self.config.attempts:
                    raise
                time.sleep(min(2**attempt, 5))

    def info(self):
        return self.call("/info")

    def open(self, collection_id, channels, seed, groups):
        return self.call(
            "/open",
            {
                "collection_id": collection_id,
                "channels": list(channels),
                "seed": seed,
                "groups": groups,
            },
        )

    def submit(self, collection_id, sample_id, group_id, draw_index, image):
        return self.call(
            "/submit",
            {
                "collection_id": collection_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "draw_index": draw_index,
                "image": image,
            },
        )

    def read(self, collection_id, after=0):
        result = self.call("/read", {"collection_id": collection_id, "after": after})
        return result

    def release(self, collection_id):
        return self.call("/release", {"collection_id": collection_id})


class Submissions:
    """Bound images awaiting transport while the learner produces its next batch."""

    def __init__(self, client, collection_id):
        self.client, self.collection_id = client, collection_id
        self.pool = ThreadPoolExecutor(max_workers=client.config.max_pending)
        self.pending = deque()
        self.wait_seconds = 0.0

    def wait(self, future):
        started = time.monotonic()
        try:
            return future.result()
        finally:
            self.wait_seconds += time.monotonic() - started

    def submit(self, sample_id, group_id, draw_index, image):
        if len(self.pending) >= self.client.config.max_pending:
            self.wait(self.pending.popleft())
        self.pending.append(
            self.pool.submit(
                self.client.submit,
                self.collection_id,
                sample_id,
                group_id,
                draw_index,
                image,
            )
        )

    def close(self):
        try:
            for future in self.pending:
                self.wait(future)
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
