from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from avgaussianv2.benchmark.architecture_ablation import (
    validate_strategy_only_delta,
)
from avgaussianv2.benchmark.cross_attention_ablation import (
    cross_attention_variant,
    validate_backend_only_delta,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_RUN_ROOT = (ROOT / "runs/cam38_strict").resolve()
FILM_EVALUATION_SYSTEMS = (
    "joint_conditioned",
    "joint_conditioned_no_rgbd",
    "joint_conditioned_wrong_camera",
)


def _weight_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "")


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_architecture_source(
    *,
    system: str,
    canonical_config: Path,
    architecture_config: Path,
    evaluation_systems: tuple[str, ...],
) -> None:
    """Reject architecture inputs that change anything beyond the tested axis."""
    if system in {"audio_only", "joint_conditioned"}:
        if architecture_config.read_bytes() != canonical_config.read_bytes():
            raise ValueError(
                f"{system} must use the byte-identical canonical scene config"
            )
        expected_evaluations = (
            ("audio_only",) if system == "audio_only" else FILM_EVALUATION_SYSTEMS
        )
    elif system == "plain_unet":
        validate_strategy_only_delta(
            canonical_config,
            architecture_config,
            expected_strategy="plain_unet",
        )
        expected_evaluations = ("plain_unet",)
    else:
        expected_backends = {
            "cross_attention": "cross_attention_tokens",
            "cross_attention_masks": "cross_attention_masks",
            "query_dependent_p1": "query_dependent_p1",
        }
        try:
            expected_backend = expected_backends[system]
        except KeyError as error:
            raise ValueError(f"unsupported architecture system: {system}") from error
        validate_backend_only_delta(
            canonical_config,
            architecture_config,
            expected_backend=expected_backend,
        )
        expected_evaluations = cross_attention_variant(
            expected_backend
        ).evaluation_systems
    if evaluation_systems != expected_evaluations:
        raise ValueError(
            f"architecture {system} evaluation systems must be "
            f"{list(expected_evaluations)}"
        )


def _rebase_paths(
    config: dict[str, Any],
    *,
    source_directory: Path,
    destination_directory: Path,
    strict_run_root: Path | None,
) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("base config paths must be a mapping")
    for name, raw_path in paths.items():
        if raw_path is None:
            continue
        if not isinstance(raw_path, str):
            raise ValueError(f"base config paths.{name} must be a string or null")
        path = Path(raw_path)
        if path.is_absolute():
            continue
        absolute = (source_directory / path).resolve()
        if strict_run_root is not None:
            try:
                relative = absolute.relative_to(DEFAULT_STRICT_RUN_ROOT)
            except ValueError:
                pass
            else:
                absolute = strict_run_root.resolve() / relative
        paths[name] = os.path.relpath(absolute, destination_directory.resolve())


def _selection_weights(
    manifest: dict[str, Any],
    *,
    stage: str,
    winners_path: Path | None,
    screening_manifest_path: Path,
) -> tuple[float, ...]:
    screening = manifest["screening"]
    available = tuple(float(value) for value in screening["lambda_lre"])
    if len(set(available)) != len(available) or 0.0 not in available:
        raise ValueError("screening.lambda_lre must be unique and include 0.0")
    if any(value < 0 for value in available):
        raise ValueError("screening.lambda_lre must be nonnegative")
    if stage == "architecture":
        architecture = manifest["architecture_screening"]
        if float(architecture["lambda_lre"]) != 0.0:
            raise ValueError("architecture screening requires lambda_lre=0")
        if winners_path is not None:
            raise ValueError("--winners is not valid for architecture screening")
        return (0.0,)
    if stage == "smoke":
        smoke = manifest["smoke"]
        weights = tuple(float(value) for value in smoke["lambda_lre"])
        if not weights or len(set(weights)) != len(weights) or any(
            value <= 0 for value in weights
        ):
            raise ValueError("smoke.lambda_lre must contain unique nonzero weights")
        if winners_path is not None:
            raise ValueError("--winners is not valid for smoke")
        return tuple(sorted(weights))
    if stage == "screening":
        if winners_path is not None:
            raise ValueError("--winners is only valid after screening")
        return tuple(sorted(available))
    if winners_path is None:
        raise ValueError(f"{stage} requires --winners")
    winners_path = winners_path.resolve()
    selection = json.loads(winners_path.read_text())
    expected_fields = {
        "schema",
        "version",
        "source_screening_manifest_sha256",
        "selected_lambda_lre",
    }
    if not isinstance(selection, dict) or not expected_fields.issubset(selection):
        raise ValueError("screening winner fields mismatch")
    if (
        selection["schema"] != "avgaussianv2.lre-loss-screening-selection"
        or selection["version"] != 1
    ):
        raise ValueError("unsupported screening winner file")
    source_digest = selection["source_screening_manifest_sha256"]
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise ValueError("source_screening_manifest_sha256 must be a SHA-256 digest")
    if (
        not screening_manifest_path.is_file()
        or source_digest != _sha256(screening_manifest_path)
    ):
        raise ValueError("screening winner file does not bind the generated manifest")
    selected = tuple(float(value) for value in selection["selected_lambda_lre"])
    maximum_count = int(screening["keep_nonzero_candidates_max"])
    if len(selected) > maximum_count or len(set(selected)) != len(selected):
        raise ValueError(
            f"winner file may contain at most {maximum_count} unique nonzero weights"
        )
    if any(value == 0.0 or value not in available for value in selected):
        raise ValueError("winner weights must be nonzero screening candidates")
    return (0.0, *sorted(selected))


def generate(
    manifest_path: Path,
    output_dir: Path,
    *,
    stage: str = "screening",
    winners_path: Path | None = None,
    systems: tuple[str, ...] | None = None,
    strict_run_root: Path | None = None,
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = _load_mapping(manifest_path)
    if (
        manifest.get("schema") != "avgaussianv2.lre-loss-ablation"
        or manifest.get("version") != 1
    ):
        raise ValueError("unsupported LRE ablation manifest")
    if stage not in {
        "smoke",
        "architecture",
        "screening",
        "confirmation",
        "robustness",
    }:
        raise ValueError("unsupported LRE experiment stage")
    fixed = manifest["fixed_loss"]
    stage_config = (
        manifest["architecture_screening"]
        if stage == "architecture"
        else manifest[stage]
    )
    seeds = (
        (int(stage_config["seed"]),)
        if stage == "architecture"
        else tuple(int(seed) for seed in stage_config["seeds"])
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{stage}.seeds must be nonempty and unique")
    weights = _selection_weights(
        manifest,
        stage=stage,
        winners_path=winners_path,
        screening_manifest_path=output_dir.resolve() / "screening/manifest.json",
    )
    systems = tuple(systems or ())
    if not systems or len(set(systems)) != len(systems):
        raise ValueError(
            "loss-search systems must be explicitly supplied from architecture survivors"
        )
    report_steps = tuple(int(step) for step in stage_config["report_steps"])
    if (
        not report_steps
        or tuple(sorted(set(report_steps))) != report_steps
        or report_steps[0] <= 0
    ):
        raise ValueError(f"{stage}.report_steps must be sorted, unique, and positive")
    max_steps = report_steps[-1]
    stage_dir = output_dir.resolve() / stage
    config_dir = output_dir.resolve() / "configs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    catalog = manifest.get("architecture_configs")
    if not isinstance(catalog, dict):
        raise ValueError("architecture_configs must be a mapping")
    configs = []
    runs = []
    for system in systems:
        specification = catalog.get(system)
        if not isinstance(specification, dict):
            raise ValueError(f"unknown architecture survivor system: {system}")
        training_mode = specification.get("training_mode")
        if training_mode not in {"audio_only", "joint_conditioned"}:
            raise ValueError(f"invalid training mode for architecture {system}")
        evaluation_systems = specification.get("evaluation_systems")
        if (
            not isinstance(evaluation_systems, list)
            or not evaluation_systems
            or evaluation_systems[0] != system
            or len(set(evaluation_systems)) != len(evaluation_systems)
            or any(not isinstance(value, str) or not value for value in evaluation_systems)
        ):
            raise ValueError(f"invalid evaluation systems for architecture {system}")
        scenes = stage_config.get("scenes", manifest.get("scenes"))
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("scenes must be a nonempty list")
        for scene in scenes:
            relative = specification.get(scene)
            if not isinstance(relative, str):
                raise ValueError(f"architecture {system} has no config for {scene}")
            base_path = (manifest_path.parent / relative).resolve()
            canonical_relative = catalog["audio_only"].get(scene)
            if not isinstance(canonical_relative, str):
                raise ValueError(f"audio_only has no canonical config for {scene}")
            canonical_path = (manifest_path.parent / canonical_relative).resolve()
            _validate_architecture_source(
                system=system,
                canonical_config=canonical_path,
                architecture_config=base_path,
                evaluation_systems=tuple(evaluation_systems),
            )
            base = _load_mapping(base_path)
            for seed in seeds:
                for weight in weights:
                    derived = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
                    if not isinstance(derived.get("train"), dict) or not isinstance(
                        derived.get("benchmark"), dict
                    ):
                        raise ValueError("base config requires train and benchmark mappings")
                    derived["train"]["seed"] = int(seed)
                    derived["train"]["lambda_lre"] = float(weight)
                    derived["train"]["lre_scale_db"] = float(fixed["lre_scale_db"])
                    derived["train"]["lre_epsilon"] = float(fixed["lre_epsilon"])
                    derived["train"]["lre_smooth_l1_beta"] = float(
                        fixed["lre_smooth_l1_beta"]
                    )
                    derived["benchmark"]["seed"] = int(seed)
                    continuation_id = (
                        ("smoke__" if stage == "smoke" else "")
                        + f"{system}__{scene}__seed{seed}"
                        f"__lre{_weight_slug(weight)}"
                    )
                    config_id = continuation_id
                    destination = config_dir / f"{config_id}.yaml"
                    _rebase_paths(
                        derived,
                        source_directory=base_path.parent,
                        destination_directory=destination.parent,
                        strict_run_root=strict_run_root,
                    )
                    data = yaml.safe_dump(derived, sort_keys=False).encode()
                    if destination.exists():
                        if destination.read_bytes() != data:
                            raise ValueError(
                                f"continuation config changed across stages: {config_id}"
                            )
                    else:
                        destination.write_bytes(data)
                    configs.append(
                        {
                            "config_id": config_id,
                            "system": system,
                            "training_mode": training_mode,
                            "scene": scene,
                            "seed": seed,
                            "lambda_lre": weight,
                            "base_config": str(base_path),
                            "config": str(destination.resolve()),
                            "config_sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                    run_id = (
                        continuation_id
                        if stage == "smoke"
                        else f"{stage}__{continuation_id}"
                    )
                    control_run_id = (
                        None
                        if weight == 0.0 or stage == "smoke"
                        else (
                            f"{stage}__{system}__{scene}__seed{seed}"
                            f"__lre{_weight_slug(0.0)}"
                        )
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "continuation_id": continuation_id,
                            "stage": stage,
                            "config_id": config_id,
                            "scene": scene,
                            "system": system,
                            "training_mode": training_mode,
                            "seed": seed,
                            "lambda_lre": weight,
                            "control_run_id": control_run_id,
                            "evaluation_systems": (
                                list(evaluation_systems)
                                if stage in {"smoke", "architecture"}
                                else [system]
                            ),
                            "report_steps": list(report_steps),
                            "max_steps": max_steps,
                            "stop_after_step": (
                                max_steps if max_steps < 30_000 else None
                            ),
                        }
                    )
    result = {
        "schema": "avgaussianv2.lre-loss-run-manifest",
        "version": 1,
        "stage": stage,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "winners": str(winners_path.resolve()) if winners_path is not None else None,
        "configs": configs,
        "runs": runs,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiments/lre_loss_ablation.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "configs/generated/lre_loss_ablation",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "architecture",
            "screening",
            "confirmation",
            "robustness",
        ),
        default="screening",
    )
    parser.add_argument(
        "--winners",
        type=Path,
        help="screening selection JSON; required for confirmation/robustness",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="architecture survivor system; repeat for multiple systems",
    )
    parser.add_argument(
        "--strict-run-root",
        type=Path,
        help=(
            "override the canonical runs/cam38_strict asset root in generated "
            "config paths"
        ),
    )
    args = parser.parse_args()
    result = generate(
        args.manifest,
        args.output_dir,
        stage=args.stage,
        winners_path=args.winners,
        systems=tuple(args.systems or ()),
        strict_run_root=args.strict_run_root,
    )
    print(
        json.dumps(
            {
                "configs": len(result["configs"]),
                "runs": len(result["runs"]),
                "stage": result["stage"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
