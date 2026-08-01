"""Fail-closed selection for the two-scene architecture screening stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from avgaussianv2.benchmark.artifacts import load_generation, repository_identity
from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationResult,
    SCENE_SAMPLE_COUNTS,
    verify_evaluation,
)
from avgaussianv2.benchmark.lre_orchestration import load_lre_run_manifest


SCHEMA = "avgaussianv2.architecture-screening-selection"
VERSION = 2
SCENES = ("scene1_opera", "Scene7playing")
QUALITY_OBJECTIVES = (
    "audio_total",
    "paper_mag",
    "paper_env",
    "paper_dpam",
    "waveform_l1",
)
SPATIAL_OBJECTIVES = (
    "paper_lre_db",
    "lre_error_db",
    "ild_error_db",
    "ipd_error_rad",
)
OBJECTIVES = (*QUALITY_OBJECTIVES, *SPATIAL_OBJECTIVES)
FAIRNESS_FIELDS = (
    "seed",
    "index_sha256",
    "planned_updates",
    "completed_updates",
    "checkpoint_step",
    "batch_size",
    "visual_initialization_sha256",
)
POSTPROCESSING_ONLY_PATHS = frozenset(
    {
        "avgaussianv2/benchmark/architecture_selection.py",
        "avgaussianv2/cli/benchmark_architecture_select.py",
        "docs/2026-07-31-post-fix-two-gpu-experiment-plan.zh-CN.md",
        "scripts/generate_lre_ablation_configs.py",
        "tests/test_architecture_selection.py",
        "tests/test_lre_ablation_configs.py",
    }
)
CAUSAL_RELATIVE_IMPROVEMENT_MIN = 0.001
SPATIAL_RELATIVE_DEGRADATION_MAX = 0.20
EXPLORATORY_PAPER_LRE_ABSOLUTE_DEGRADATION_MAX_DB = 1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_preparation(run_dir: Path) -> Mapping[str, object]:
    """Load and authenticate the immutable preparation used by one run."""
    protocol = Path(run_dir) / "protocol"
    _, files, _, _ = load_generation(
        protocol / "immutable",
        schema="avgaussianv2.cam38-production-preparation",
    )
    data = files.get("preparation.json")
    if data is None or (protocol / "preparation.json").read_bytes() != data:
        raise ValueError(f"architecture preparation is missing or differs: {run_dir}")
    try:
        preparation = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"architecture preparation is invalid: {run_dir}") from error
    if not isinstance(preparation, Mapping):
        raise ValueError(f"architecture preparation must be a mapping: {run_dir}")
    try:
        continuation = json.loads(
            (Path(run_dir) / "continuation_identity.json").read_text()
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"architecture continuation identity is invalid: {run_dir}") from error
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("schema")
        != "avgaussianv2.lre-loss-continuation-identity"
        or continuation.get("version") != 1
        or continuation.get("continuation_id") != Path(run_dir).name
    ):
        raise ValueError(f"architecture continuation identity mismatches: {run_dir}")
    authenticated = dict(preparation)
    authenticated["_continuation_identity"] = dict(continuation)
    return authenticated


def _verify_postprocessing_only_revision(
    experiment_repository: Mapping[str, object],
    current_repository: Mapping[str, object],
) -> None:
    """Permit historical results only across an audited post-processing-only diff."""
    if (
        experiment_repository.get("clean") is not True
        or current_repository.get("clean") is not True
        or experiment_repository.get("root") != current_repository.get("root")
    ):
        raise ValueError("historical architecture repository identity is not clean/aligned")
    root = Path(str(current_repository["root"])).resolve()
    old = str(experiment_repository.get("commit", ""))
    new = str(current_repository.get("commit", ""))
    if len(old) != 40 or len(new) != 40 or old == new:
        raise ValueError("historical architecture revisions are invalid")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", old, new],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("experiment revision is not an ancestor of selector revision")
    changed = subprocess.run(
        ["git", "diff", "--name-only", old, new, "--"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = frozenset(line for line in changed.stdout.splitlines() if line)
    unexpected = sorted(paths - POSTPROCESSING_ONLY_PATHS)
    if not paths or unexpected:
        raise ValueError(
            "historical architecture revision is not post-processing-only: "
            + ", ".join(unexpected or ["empty diff"])
        )


def _mean(result: BenchmarkEvaluationResult, metric: str) -> float:
    try:
        value = float(result.summary[metric]["mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"architecture evaluation lacks required metric {metric}: "
            f"{result.identity.scene_id}/{result.identity.system_name}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"architecture metric is nonfinite: {metric}")
    if result.metric_directions.get(metric) != "lower_is_better":
        raise ValueError(f"architecture objective has invalid direction: {metric}")
    return value


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    objectives: tuple[str, ...] = OBJECTIVES,
) -> bool:
    return all(left[name] <= right[name] for name in objectives) and any(
        left[name] < right[name] for name in objectives
    )


def _dpam_protocol(result: BenchmarkEvaluationResult) -> Mapping[str, object]:
    registry = result.metric_protocol.get("extra_metric_registry")
    record = registry.get("paper_dpam") if isinstance(registry, Mapping) else None
    protocol = record.get("protocol") if isinstance(record, Mapping) else None
    if (
        not isinstance(protocol, Mapping)
        or record.get("direction") != "lower_is_better"
        or record.get("modality") != "audio"
    ):
        raise ValueError(
            "architecture DPAM protocol is missing: "
            f"{result.identity.scene_id}/{result.identity.system_name}"
        )
    return protocol


def select_architecture_winners(
    manifest_path: Path,
    run_root: Path,
    output_path: Path,
    *,
    evaluation_loader: Callable[[Path], BenchmarkEvaluationResult] = (
        verify_evaluation
    ),
    preparation_loader: Callable[[Path], Mapping[str, object]] = _load_preparation,
    repository_identity_getter: Callable[[], Mapping[str, object]] = (
        repository_identity
    ),
    revision_validator: Callable[
        [Mapping[str, object], Mapping[str, object]], None
    ] = _verify_postprocessing_only_revision,
    allow_postprocessing_revision: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    run_root = Path(run_root).resolve()
    manifest = load_lre_run_manifest(manifest_path)
    if manifest["stage"] != "architecture":
        raise ValueError("architecture selection requires an architecture manifest")
    current_repository = dict(repository_identity_getter())
    historical_revision = manifest["repository"] != current_repository
    if historical_revision:
        if not allow_postprocessing_revision:
            raise ValueError("architecture manifest repository identity is not current")
        revision_validator(manifest["repository"], current_repository)

    runs: dict[tuple[str, str], Mapping[str, object]] = {}
    for run in manifest["runs"]:
        key = (str(run["scene"]), str(run["system"]))
        if (
            key in runs
            or int(run["seed"]) != 42
            or float(run["lambda_lre"]) != 0.0
            or int(run["max_steps"]) != 5_000
            or list(run["report_steps"]) != [5_000]
        ):
            raise ValueError(f"invalid architecture screening run: {run['run_id']}")
        runs[key] = run
    systems = tuple(sorted({system for _, system in runs}))
    if "audio_only" not in systems or set(runs) != {
        (scene, system) for scene in SCENES for system in systems
    }:
        raise ValueError("architecture screening coverage must be rectangular")

    main: dict[tuple[str, str], BenchmarkEvaluationResult] = {}
    causal: dict[tuple[str, str], tuple[BenchmarkEvaluationResult, ...]] = {}
    preparations: dict[tuple[str, str], Mapping[str, object]] = {}
    global_dpam_protocol = None
    for scene in SCENES:
        expected_ids = None
        for system in systems:
            run = runs[(scene, system)]
            run_dir = run_root / str(run["continuation_id"])
            root = run_dir / "evaluations"
            preparation = preparation_loader(run_dir)
            continuation = preparation.get("_continuation_identity")
            if (
                preparation.get("scene_id") != scene
                or not isinstance(continuation, Mapping)
                or continuation.get("repository") != manifest["repository"]
                or continuation.get("scene") != scene
                or continuation.get("system") != system
                or continuation.get("seed") != 42
                or float(continuation.get("lambda_lre", float("nan"))) != 0.0
            ):
                raise ValueError(
                    f"architecture preparation identity mismatch: {scene}/{system}"
                )
            preparations[(scene, system)] = preparation
            result = evaluation_loader(root / "step_005000")
            if (
                result.identity.scene_id != scene
                or result.identity.system_name != system
                or result.identity.reporting_step != 5_000
                or result.count != SCENE_SAMPLE_COUNTS[scene]
                or result.identity.expected_sample_count != SCENE_SAMPLE_COUNTS[scene]
            ):
                raise ValueError(f"architecture evaluation identity mismatch: {scene}/{system}")
            for metric in OBJECTIVES:
                _mean(result, metric)
            if expected_ids is None:
                expected_ids = result.identity.expected_sample_ids
            elif expected_ids != result.identity.expected_sample_ids:
                raise ValueError(f"architecture sample IDs differ within scene: {scene}")
            protocol = _dpam_protocol(result)
            if global_dpam_protocol is None:
                global_dpam_protocol = protocol
            elif global_dpam_protocol != protocol:
                raise ValueError("architecture DPAM protocols differ")
            main[(scene, system)] = result

            branches = []
            branch_names = list(run.get("evaluation_systems", [system]))[1:]
            if branch_names and (
                len(branch_names) != 2 or not str(branch_names[0]).endswith("_no_rgbd")
            ):
                raise ValueError(f"invalid architecture causal branch set: {system}")
            for branch in branch_names:
                counterfactual = evaluation_loader(
                    root / str(branch) / "step_005000"
                )
                if (
                    counterfactual.identity.scene_id != scene
                    or counterfactual.identity.system_name != branch
                    or counterfactual.identity.reporting_step != 5_000
                    or counterfactual.identity.expected_sample_ids
                    != result.identity.expected_sample_ids
                    or counterfactual.provenance.get("checkpoint_sha256")
                    != result.provenance.get("checkpoint_sha256")
                    or counterfactual.provenance.get("model_initialization_sha256")
                    != result.provenance.get("model_initialization_sha256")
                ):
                    raise ValueError(
                        f"architecture causal evaluation mismatch: {scene}/{branch}"
                    )
                _mean(counterfactual, "audio_total")
                branches.append(counterfactual)
            causal[(scene, system)] = tuple(branches)

        control = main[(scene, "audio_only")]
        control_preparation = preparations[(scene, "audio_only")]
        for system in systems:
            result = main[(scene, system)]
            for field in FAIRNESS_FIELDS:
                if (
                    field not in result.provenance
                    or field not in control.provenance
                    or result.provenance[field] != control.provenance[field]
                ):
                    raise ValueError(
                        f"architecture fairness mismatch: {scene}/{system}/{field}"
                    )
            preparation = preparations[(scene, system)]
            native = preparation.get("native_contracts")
            control_native = control_preparation.get("native_contracts")
            if (
                not isinstance(native, Mapping)
                or set(native) != {"audiogs", "ftgspp"}
                or native != control_native
            ):
                raise ValueError(
                    f"architecture native fairness mismatch: {scene}/{system}/native_contracts"
                )
            for kind, record in native.items():
                if (
                    not isinstance(record, Mapping)
                    or not isinstance(record.get("path"), str)
                    or not all(
                        isinstance(record.get(field), str)
                        and len(record[field]) == 64
                        for field in ("checkpoint_sha256", "manifest_sha256")
                    )
                ):
                    raise ValueError(
                        f"architecture native contract is incomplete: {scene}/{system}/{kind}"
                    )
            runtime = preparation.get("runtime")
            control_runtime = control_preparation.get("runtime")
            if not isinstance(runtime, Mapping) or not isinstance(
                control_runtime, Mapping
            ):
                raise ValueError(
                    f"architecture preparation runtime is missing: {scene}/{system}"
                )
            if (
                runtime.get("dataset_identity_sha256")
                != control_runtime.get("dataset_identity_sha256")
            ):
                raise ValueError(
                    f"architecture dataset fairness mismatch: {scene}/{system}"
                )
            if (
                result.provenance["seed"] != 42
                or result.provenance["completed_updates"] != 5_000
                or result.provenance["checkpoint_step"] != 5_000
                or result.provenance["batch_size"] != 1
            ):
                raise ValueError(f"architecture fixed protocol mismatch: {scene}/{system}")

    candidates = []
    macro: dict[str, dict[str, float]] = {}
    eligible = []
    for system in systems:
        per_scene = {
            scene: {metric: _mean(main[(scene, system)], metric) for metric in OBJECTIVES}
            for scene in SCENES
        }
        macro[system] = {
            metric: sum(per_scene[scene][metric] for scene in SCENES) / len(SCENES)
            for metric in OBJECTIVES
        }
        branches = [causal[(scene, system)] for scene in SCENES]
        causal_gate = True
        causal_evidence = []
        if any(branches):
            if any(len(values) < 2 for values in branches):
                raise ValueError(f"conditioned architecture lacks two causal branches: {system}")
            no_rgbd_relative_improvement = []
            alternate_delta = []
            for scene, values in zip(SCENES, branches, strict=True):
                main_audio = _mean(main[(scene, system)], "audio_total")
                no_rgbd_audio = _mean(values[0], "audio_total")
                alternate_audio = _mean(values[1], "audio_total")
                first = no_rgbd_audio - main_audio
                second = alternate_audio - main_audio
                relative = first / max(abs(no_rgbd_audio), 1e-12)
                no_rgbd_relative_improvement.append(relative)
                alternate_delta.append(second)
                causal_evidence.append(
                    {
                        "scene": scene,
                        "no_rgbd_minus_main_audio_total": first,
                        "main_relative_improvement_over_no_rgbd": relative,
                        "alternate_minus_main_audio_total": second,
                        "alternate_system": values[1].identity.system_name,
                    }
                )
            causal_gate = (
                sum(no_rgbd_relative_improvement)
                / len(no_rgbd_relative_improvement)
                >= CAUSAL_RELATIVE_IMPROVEMENT_MIN
                and not all(delta < 0.0 for delta in alternate_delta)
            )
        elif system not in {"audio_only", "plain_unet"}:
            raise ValueError(f"conditioned architecture lacks causal evaluations: {system}")
        dominated_by_audio_only = (
            system != "audio_only"
            and _dominates(
                macro["audio_only"], macro[system], QUALITY_OBJECTIVES
            )
        )
        spatial_relative_degradation = {
            metric: (
                macro[system][metric] - macro["audio_only"][metric]
            )
            / max(abs(macro["audio_only"][metric]), 1e-12)
            for metric in SPATIAL_OBJECTIVES
        }
        spatial_gate = all(
            value <= SPATIAL_RELATIVE_DEGRADATION_MAX
            for value in spatial_relative_degradation.values()
        )
        passed = causal_gate and spatial_gate and not dominated_by_audio_only
        if passed:
            eligible.append(system)
        candidates.append(
            {
                "system": system,
                "passed": passed,
                "causal_gate_passed": causal_gate,
                "dominated_by_audio_only": dominated_by_audio_only,
                "spatial_guardrail_passed": spatial_gate,
                "spatial_relative_degradation_vs_audio_only": (
                    spatial_relative_degradation
                ),
                "per_scene": per_scene,
                "scene_macro": macro[system],
                "causal_evidence": causal_evidence,
                "inputs": [
                    {
                        "scene": scene,
                        "content_sha256": main[(scene, system)].content_sha256,
                        "causal_content_sha256": [
                            result.content_sha256 for result in causal[(scene, system)]
                        ],
                    }
                    for scene in SCENES
                ],
            }
        )

    pareto = [
        system
        for system in eligible
        if not any(
            other != system and _dominates(macro[other], macro[system])
            for other in eligible
        )
    ]
    ranked = sorted(
        pareto,
        key=lambda system: (
            *(macro[system][metric] for metric in OBJECTIVES),
            system,
        ),
    )
    selected = ranked[:2]
    if not selected:
        raise ValueError("no architecture passes the pre-registered gates")
    exploratory = []
    if not any(system not in {"audio_only", "plain_unet"} for system in selected):
        rescue_pool = [
            candidate
            for candidate in candidates
            if candidate["system"] not in {"audio_only", "plain_unet"}
            and candidate["causal_gate_passed"]
            and not candidate["dominated_by_audio_only"]
            and (
                candidate["scene_macro"]["paper_lre_db"]
                - macro["audio_only"]["paper_lre_db"]
                <= EXPLORATORY_PAPER_LRE_ABSOLUTE_DEGRADATION_MAX_DB
            )
        ]
        rescue_pool.sort(
            key=lambda candidate: (
                candidate["scene_macro"]["paper_lre_db"]
                - macro["audio_only"]["paper_lre_db"],
                *(candidate["scene_macro"][metric] for metric in QUALITY_OBJECTIVES),
                candidate["system"],
            )
        )
        if rescue_pool and len(selected) < 2:
            exploratory = [str(rescue_pool[0]["system"])]
    screening_systems = [*selected, *exploratory]
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "repository": current_repository,
        "experiment_repository": manifest["repository"],
        "postprocessing_revision_only": historical_revision,
        "fairness_policy": {
            "shared_fields": list(FAIRNESS_FIELDS),
            "shared_preparation_fields": [
                "native_contracts",
                "runtime.dataset_identity_sha256",
            ],
            "architecture_specific_initialization_fields": [
                "audio_initialization_sha256",
                "model_initialization_sha256",
            ],
        },
        "objective_order": list(OBJECTIVES),
        "selection_method": "gates_then_pareto_then_lexicographic",
        "causal_relative_improvement_min": CAUSAL_RELATIVE_IMPROVEMENT_MIN,
        "spatial_relative_degradation_max": SPATIAL_RELATIVE_DEGRADATION_MAX,
        "comparison_scope": "main_updates_matched_not_total_compute_matched",
        "conditioned_warmup_updates": 2_000,
        "selected_systems": selected,
        "exploratory_systems": exploratory,
        "screening_systems": screening_systems,
        "exploratory_policy": {
            "post_hoc": True,
            "purpose": "test_whether_lre_loss_can_rescue_best_conditioned_candidate",
            "activated_only_without_conditioned_confirmatory_survivor": True,
            "paper_lre_absolute_degradation_max_db": (
                EXPLORATORY_PAPER_LRE_ABSOLUTE_DEGRADATION_MAX_DB
            ),
            "ranking": [
                "paper_lre_absolute_degradation_vs_audio_only",
                *QUALITY_OBJECTIVES,
                "stable_system_id",
            ],
        },
        "pareto_systems": sorted(pareto),
        "candidates": candidates,
    }
    output_path = Path(output_path).resolve()
    if output_path.is_file():
        try:
            existing = json.loads(output_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("cannot verify existing architecture selection") from error
        if existing != result:
            raise FileExistsError("architecture selection output already differs")
    else:
        _atomic_json(output_path, result)
    return result


__all__ = ["SCHEMA", "select_architecture_winners"]
