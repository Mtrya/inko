"""Exact-source capability receipts for bounded implementation qualification."""

import json
import tarfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from src.posttrain.provenance import REPO
from src.posttrain.records import file_hash, identity
from src.posttrain.schema import decode, digest, name, require
from src.posttrain.stages import rounds_before
from src.posttrain.training_config import load_training_config

CAPABILITIES = frozenset(
    {
        "student.full",
        "student.lora.attention",
        "student.lora.attention_mlp",
        "fake.full",
        "fake.teacher_qv",
        "fake.selected_prediction",
        "fake.terminal",
        "fake.q1",
        "fake.q_multiple",
        "optimizer.student.adamw",
        "optimizer.student.muon_adamw",
        "optimizer.fake.adamw",
        "optimizer.fake.muon_adamw",
        "lr.student.constant",
        "lr.student.cosine",
        "lr.fake.constant",
        "lr.fake.cosine",
        "lr.warmup",
        "dmd.teacher_residual",
        "dmd.clean_difference",
        "dmd.score_kl",
        "sampling.euler",
        "sampling.renoise",
        "sampling.cfg",
        "sampling.shape_shift",
        "sampling.no_shift",
        "sampling.selected_uniform",
        "sampling.selected_nonuniform",
        "execution.checkpointing",
        "execution.no_checkpointing",
        "rewards.hps_only",
        "rewards.vqa_only",
        "rewards.mixture",
        "rewards.raw",
        "rewards.standardized",
        "rewards.mean_one_exp",
        "rewards.paper_max_exp",
        "joint.dmd.full",
        "joint.dmd.lora",
        "joint.nft.full",
        "joint.nft.lora",
        "joint.both.full",
        "joint.both.lora",
        "nft.clean_adaptive",
        "nft.velocity_mse",
        "nft.group",
        "nft.round",
        "nft.uniform",
        "nft.grid",
        "nft.multiple_updates",
        "nft.multiple_times",
        "old.copy",
        "old.constant",
        "old.linear_ramp",
        "reference.nft",
        "reference.joint",
        "integration.full",
        "integration.lora",
        "recovery.boundary.full",
        "recovery.boundary.lora",
        "recovery.mid_nft.full",
        "recovery.mid_nft.lora",
        "export.live",
        "export.ema",
        "scorer.substitution",
        *(f"cold.{bits}" for bits in ("001", "010", "011", "100", "101", "110")),
        *(
            f"cold.{bits}.{mode}"
            for bits in ("000", "111")
            for mode in ("full", "lora")
        ),
    }
)


def source_files():
    return {
        str(p.relative_to(REPO)): file_hash(p)
        for p in sorted((REPO / "src").rglob("*.py"))
    }


def qualification_files():
    """Current runner and mechanism bytes reviewed separately from core runs."""
    paths = [
        *(REPO / "scripts" / "posttrain").glob("*.py"),
        *(REPO / "tests").glob("test_posttrain*.py"),
    ]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


@dataclass(frozen=True)
class EvidencePin:
    archive: str
    member: str
    sha256: str


@dataclass(frozen=True)
class Case:
    id: str
    config: str
    config_sha256: str
    steps: int
    capabilities: tuple[str, ...]
    execution: str
    resume: str
    compare: str
    exports: tuple[str, ...]
    evidence: tuple[EvidencePin, ...]


@dataclass(frozen=True)
class Matrix:
    source_sha256: str
    evaluation: str
    evaluation_sha256: str
    requirements: tuple[str, ...]
    cases: tuple[Case, ...]


def configured_capabilities(config, steps):
    """Only branches reached before the explicit stop, excluding artifact claims."""
    c, mode = config, config.student.mode
    result = {
        "student.full" if mode == "full" else f"student.lora.{c.student.targets}",
        f"optimizer.student.{c.student_optimizer.kind}",
        f"lr.student.{c.student_optimizer.schedule}",
        f"sampling.{c.collection.rule}",
        (
            "sampling.shape_shift"
            if c.collection.shift == "shape"
            else "sampling.no_shift"
        ),
        (
            "sampling.selected_uniform"
            if len(set(c.collection.selected_pmf)) == 1
            else "sampling.selected_nonuniform"
        ),
        (
            "execution.checkpointing"
            if c.execution.activation_checkpointing
            else "execution.no_checkpointing"
        ),
    }
    cold = min(steps, c.stages.cold_steps)
    joint = max(0, steps - c.stages.cold_steps)
    if c.collection.guidance != 1 or c.objective.teacher_guidance != 1:
        result.add("sampling.cfg")
    dmd = cold > 0 or (joint > 0 and c.joint.dmd_coefficient > 0)
    if dmd:
        result.update(
            {
                f"fake.{c.fake.strategy}",
                f"fake.{c.fake.target}",
                "fake.q1" if c.fake.updates == 1 else "fake.q_multiple",
                f"dmd.{c.objective.estimator}",
                f"optimizer.fake.{c.fake_optimizer.kind}",
                f"lr.fake.{c.fake_optimizer.schedule}",
            }
        )
    if c.student_optimizer.warmup_steps or (dmd and c.fake_optimizer.warmup_steps):
        result.add("lr.warmup")
    if cold:
        bits = "".join(
            str(int(on))
            for on in (
                c.objective.mode == "rtdm",
                c.dyna_dg.enabled,
                c.dyna_rs.enabled,
            )
        )
        result.add(f"cold.{bits}" + (f".{mode}" if bits in {"000", "111"} else ""))
    scored = (cold and c.objective.mode == "rtdm") or (
        joint and c.joint.nft_coefficient > 0
    )
    if scored:
        channels = {ch.channel for ch in c.rewards.channels}
        if channels == {"hpsv3.mu"}:
            result.add("rewards.hps_only")
        elif channels == {"vqascore.probability"}:
            result.add("rewards.vqa_only")
        elif len(channels) > 1:
            result.add("rewards.mixture")
        for channel in c.rewards.channels:
            result.add(
                "rewards.raw"
                if channel.standardization == "none"
                else "rewards.standardized"
            )
        if cold and c.objective.mode == "rtdm":
            result.add(f"rewards.{c.rewards.weighting}")
    if joint:
        dmd_on, nft_on = c.joint.dmd_coefficient > 0, c.joint.nft_coefficient > 0
        control = "both" if dmd_on and nft_on else "dmd" if dmd_on else "nft"
        result.add(f"joint.{control}.{mode}")
        if nft_on:
            result.update(
                {
                    f"nft.{c.nft.parameterization}",
                    f"nft.{c.nft.advantage_scale}",
                    f"nft.{c.nft.time_distribution}",
                    f"old.{c.old_policy.strategy}",
                }
            )
            if c.nft.updates_per_round > 1:
                result.add("nft.multiple_updates")
            if c.nft.time_draws > 1:
                result.add("nft.multiple_times")
            if c.joint.reference_coefficient > 0:
                result.add("reference.joint" if dmd_on else "reference.nft")
        if cold:
            result.add(f"integration.{mode}")
    return result


def validate_matrix(payload, *, repo=REPO):
    matrix = decode(Matrix, payload)
    digest(matrix.source_sha256)
    require(
        identity(source_files()) == matrix.source_sha256,
        "qualification source hash differs",
    )
    from src.posttrain.config import load_config

    digest(matrix.evaluation_sha256)
    evaluation = Path(repo) / matrix.evaluation
    require(
        file_hash(evaluation) == matrix.evaluation_sha256,
        "evaluation config hash differs",
    )
    load_config(evaluation)
    require(
        bool(matrix.cases) and bool(matrix.requirements), "empty qualification matrix"
    )
    require(
        len(set(matrix.requirements)) == len(matrix.requirements)
        and set(matrix.requirements) <= CAPABILITIES,
        "unknown or duplicate required capability",
    )
    require(
        len({c.id for c in matrix.cases}) == len(matrix.cases),
        "duplicate qualification case",
    )
    assigned = set()
    for case in matrix.cases:
        name(case.id)
        digest(case.config_sha256)
        path = Path(repo) / case.config
        require(
            file_hash(path) == case.config_sha256, "qualification config hash differs"
        )
        config = load_training_config(path)
        require(0 < case.steps <= config.training.steps, "invalid planned step count")
        rounds_before(config, case.steps)
        require(case.execution in {"run", "reuse", "resume"}, "unknown case execution")
        require(
            bool(case.capabilities)
            and len(set(case.capabilities)) == len(case.capabilities)
            and set(case.capabilities) <= CAPABILITIES,
            "unknown or duplicate case capability",
        )
        require(
            case.execution != "run" or not case.resume,
            "fresh case cannot declare a resume artifact",
        )
        require(
            case.execution != "resume" or (bool(case.resume) and bool(case.compare)),
            "resume qualification requires starting and comparison artifacts",
        )
        require(
            bool(case.evidence) if case.execution == "reuse" else not case.evidence,
            "reused cases require pinned evidence; executing cases cannot claim old receipts",
        )
        require(
            bool(case.exports)
            and len(set(case.exports)) == len(case.exports)
            and set(case.exports) <= {"live", *(v.name for v in config.training.ema)},
            "invalid qualified export views",
        )
        allowed = configured_capabilities(config, case.steps) | {
            f"export.{'live' if w == 'live' else 'ema'}" for w in case.exports
        }
        if case.execution == "resume":
            allowed.add(f"recovery.boundary.{config.student.mode}")
        if case.execution == "reuse":
            allowed.add(f"recovery.mid_nft.{config.student.mode}")
        if config.requires_scores and any(
            spec.id not in {"hpsv3", "vqascore"} and spec.structured_fields
            for spec in config.scorers
        ):
            allowed.add("scorer.substitution")
        require(
            set(case.capabilities) <= allowed,
            "capability is not reached by case config",
        )
        for pin in case.evidence:
            name(pin.archive)
            digest(pin.sha256)
            require(
                ".." not in Path(pin.member).parts
                and not Path(pin.member).is_absolute(),
                "unsafe evidence member",
            )
        assigned.update(case.capabilities)
    require(
        set(matrix.requirements) <= assigned, "required capability has no assigned case"
    )
    return matrix


class Evidence:
    """Read pinned named archive members, never expand a raw evidence tree."""

    def __init__(self, archives):
        self.archives = archives
        self.opened = {}
        self.stack = ExitStack()

    def archive(self, key):
        if key not in self.opened:
            require(key in self.archives, "missing evidence archive")
            spec = self.archives[key]
            require(set(spec) == {"path", "sha256"}, "invalid archive fields")
            digest(spec["sha256"])
            require(
                file_hash(spec["path"]) == spec["sha256"],
                "evidence archive hash differs",
            )
            self.opened[key] = self.stack.enter_context(
                tarfile.open(spec["path"])  # noqa: SIM115 - ExitStack owns cached archives
            )
        return self.opened[key]

    def read(self, key, member):
        tar = self.archive(key)
        try:
            stream = tar.extractfile(member)
            require(stream is not None, "evidence member is not a file")
            return stream.read()
        except KeyError as exc:
            raise ValueError(f"missing evidence member: {member}") from exc

    def pinned(self, pin):
        from hashlib import sha256

        raw = self.read(pin.archive, pin.member)
        require(sha256(raw).hexdigest() == pin.sha256, "evidence member hash differs")
        return json.loads(raw)

    def close(self):
        self.stack.close()


def validate_reuse(case, config, evidence):
    """Admit current whole-run evidence, exports and interruption witnesses."""
    objects = {pin.member: evidence.pinned(pin) for pin in case.evidence}
    manifests = [
        (pin, objects[pin.member])
        for pin in case.evidence
        if pin.member.endswith("/manifest.json") and "/checkpoints/" not in pin.member
    ]
    require(len(manifests) == 1, "reuse requires one run manifest")
    pin, contract = manifests[0]
    require(contract["source"] == source_files(), "reused execution source differs")
    require(
        contract["config"] == json.loads(json.dumps(asdict(config))),
        "reused config differs",
    )
    prefix = pin.member.removesuffix("manifest.json")
    tar = evidence.archive(pin.archive)
    names = tar.getnames()
    completions = {
        member.rsplit("-rank-", 1)[-1].removesuffix(".json"): json.loads(
            evidence.read(pin.archive, member)
        )
        for member in names
        if member.startswith(prefix + "completion/")
        and member.endswith(".json")
        and json.loads(evidence.read(pin.archive, member))["step"] == case.steps
    }
    require(
        set(completions)
        == {f"{r:03d}" for r in range(len(config.execution.learner_devices))}
        and all(row["frozen_unchanged"] for row in completions.values()),
        "missing complete learner evidence",
    )
    checkpoint_member = prefix + f"checkpoints/step-{case.steps:06d}/manifest.json"
    checkpoint = json.loads(evidence.read(pin.archive, checkpoint_member))
    require(
        checkpoint["contract"] == contract and checkpoint["step"] == case.steps,
        "reused checkpoint differs",
    )
    expected_views = set(case.exports)
    exports = [
        obj for member, obj in objects.items() if member.endswith("/export.json")
    ]
    require(
        {e["weights"] for e in exports} == expected_views,
        "missing reused export receipt",
    )
    from hashlib import sha256

    checkpoint_hash = sha256(evidence.read(pin.archive, checkpoint_member)).hexdigest()
    require(
        all(
            e["reload_exact"]
            and e["source_checkpoint"] == checkpoint_hash
            and e["contract_sha256"] == identity(contract)
            for e in exports
        ),
        "export receipt/checkpoint identity differs",
    )
    recovered = any(cap.startswith("recovery.mid_nft.") for cap in case.capabilities)
    if recovered:
        comparisons = [
            o
            for member, o in objects.items()
            if member.endswith("-recovery-comparison.json")
        ]
        kills = [o for member, o in objects.items() if member.endswith("-kill.json")]
        require(
            len(comparisons) == len(kills) == 1
            and comparisons[0]["all_role_optimizer_clock_reward_tensors_exact"]
            and comparisons[0]["all_12_rank_rng_cursor_states_exact"]
            and comparisons[0]["old_terminal_conditioning_images_noise_exact"]
            and comparisons[0]["recollected_samples"]
            == config.collection.prompts_per_round * config.collection.draws_per_prompt
            and kills[0]["rank"] == 0
            and kills[0]["exit_code"] != 0
            and kills[0]["intermediate_checkpoint_absent"]
            and comparisons[0]["control_manifest"] == checkpoint_hash,
            "missing successful interrupted-recovery witnesses",
        )
    return {
        "case": case.id,
        "execution": "reused_exact_source",
        "config_sha256": case.config_sha256,
        "source_sha256": identity(contract["source"]),
        "capabilities": list(case.capabilities),
        "step": case.steps,
        "learner_count": len(completions),
        "exports": exports,
        "pins": [asdict(p) for p in case.evidence],
        "limit": "Pinned local metadata/receipts; original large checkpoint/image/latent bytes remain remote",
    }


def coverage_report(matrix, results, *, reviews=()):
    """Absence is an open gate; proposed assignments never count as execution."""
    by_id = {result["case"]: result for result in results}
    require(len(by_id) == len(results), "duplicate case result")
    known = {case.id: case for case in matrix.cases}
    require(set(by_id) <= known.keys(), "unknown case result")
    covered = set()
    for key, result in by_id.items():
        case = known[key]
        require(
            result["config_sha256"] == case.config_sha256
            and result["source_sha256"] == matrix.source_sha256
            and result["step"] == case.steps
            and set(result["capabilities"]) == set(case.capabilities),
            "case result identity differs",
        )
        covered.update(result["capabilities"])
    qualification_sha256 = identity(qualification_files())
    passed_reviews = set()
    for review in reviews:
        require(review["kind"] in {"native", "independent"}, "unknown review kind")
        require(
            review["source_sha256"] == matrix.source_sha256
            and review["qualification_sha256"] == qualification_sha256
            and review["matrix_sha256"] == identity(asdict(matrix))
            and review["results_sha256"] == identity(results),
            "review identities differ",
        )
        require(
            file_hash(review["output"]) == review["output_sha256"],
            "review output hash differs",
        )
        require(
            Path(review["output"]).read_text().strip(), "empty review is not a pass"
        )
        require(review["exit_code"] == 0, "failed reviewer is not a pass")
        if review["verdict"] == "pass":
            passed_reviews.add(review["kind"])
    missing = sorted(set(matrix.requirements) - covered)
    family_missing = sorted(CAPABILITIES - set(matrix.requirements))
    return {
        "source_sha256": matrix.source_sha256,
        "qualification_sha256": qualification_sha256,
        "matrix_sha256": identity(asdict(matrix)),
        "covered": sorted(covered),
        "missing_capabilities": missing,
        "missing_cases": sorted(known.keys() - by_id.keys()),
        "missing_reviews": sorted({"native", "independent"} - passed_reviews),
        "missing_family_requirements": family_missing,
        "qualified": not missing
        and not family_missing
        and not (known.keys() - by_id.keys())
        and passed_reviews == {"native", "independent"},
        "cases": list(results),
        "interpretation": "Implementation execution/recovery/protocol coverage, never quality or variant selection",
    }
