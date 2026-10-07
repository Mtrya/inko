from dataclasses import replace

import pytest
import torch

from src.posttrain.records import ScoreRequest, ScoreResult, file_hash, join_scores
from src.posttrain.scoring import acquire, float32_head_inputs


def test_retry_preserves_image_and_failure_is_not_zero(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"immutable test input")
    request = ScoreRequest("sample", str(path), file_hash(path), "prompt")

    class Backend:
        def __init__(self):
            self.calls = 0

        def score(self, requests):
            assert requests == [request]
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return [{"value": 0.75}]

    result = acquire(Backend(), [request], scorer="s", scorer_identity="v", attempts=2)[
        0
    ]
    assert result.status == "ok" and result.attempts == 2
    failed = acquire(Backend(), [request], scorer="s", scorer_identity="v", attempts=1)[
        0
    ]
    assert failed.status == "failed" and failed.values == {}
    with pytest.raises(ValueError, match="forbidden"):
        join_scores([request], [failed], {"s": "v"}, require_complete=True)


def test_join_uses_image_prompt_and_scorer_identity(tmp_path):
    request = ScoreRequest("a", "unused", "imagehash", "caption")
    result = ScoreResult("a", request.key("v"), "s", "v", "ok", {"value": 1.0}, 1, None)
    assert (
        join_scores([request], [result], {"s": "v"}, require_complete=True)["a"]["s"]
        == result
    )
    for changed in (
        replace(request, image_sha256="different"),
        replace(request, prompt="different"),
    ):
        with pytest.raises(ValueError, match="different"):
            join_scores([changed], [result], {"s": "v"}, require_complete=True)
    with pytest.raises(ValueError, match="duplicate"):
        join_scores([request], [result, result], {"s": "v"}, require_complete=False)
    with pytest.raises(ValueError, match="nonfinite"):
        replace(result, values={"value": float("nan")})


def test_hps_head_boundary_preserves_fp32_projection():
    head = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2.0, 3.0]]))
    hidden = torch.tensor([[4.0, 5.0]], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        head(hidden)
    head.register_forward_pre_hook(float32_head_inputs)
    assert head(hidden).dtype == torch.float32
    assert head(hidden).item() == 23.0
