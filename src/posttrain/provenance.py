"""Content identity shared by training checkpoint admission and export."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import tree_identity
from .records import file_hash
from .schema import require
from .scoring.lifecycle import isolated_environment

REPO = Path(__file__).resolve().parents[2]


def training_contract(config, assets, prompts):
    from .initialization import boundary_manifest

    boundary = boundary_manifest(config, assets)
    for checkpoint in (config.initialization, config.teacher):
        path = Path(assets["checkpoints"][checkpoint.asset])
        require(file_hash(path) == checkpoint.sha256, "initial checkpoint bytes differ")
        require(
            file_hash(path.with_name("transformer_config.json"))
            == checkpoint.architecture_sha256,
            "initial architecture bytes differ",
        )
    native = assets["learner"].get("runtime_files", {})
    require(
        assets["device_type"] != "npu" or bool(native),
        "NPU training requires native runtime inventory",
    )
    runtime = subprocess.run(
        [sys.executable, "-m", "src.posttrain.runtime"],
        env=isolated_environment(assets["learner"]["python_paths"]),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(
        json.dumps(
            {
                "config": asdict(config),
                "boundary": boundary,
                "prompts": prompts,
                "device_type": assets["device_type"],
                "runtime": json.loads(runtime.stdout),
                "native_runtime": {k: file_hash(v) for k, v in sorted(native.items())},
                "text_encoder": tree_identity(assets["text_encoder"]),
                "vae": tree_identity(assets["vae"]),
                "source": {
                    str(p.relative_to(REPO)): file_hash(p)
                    for p in sorted((REPO / "src").rglob("*.py"))
                },
            }
        )
    )
