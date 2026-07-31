from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def _weight_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "")


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebase_paths(
    config: dict[str, Any],
    *,
    source_directory: Path,
    destination_directory: Path,
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
    if not isinstance(selection, dict) or set(selection) != expected_fields:
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
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = _load_mapping(manifest_path)
    if (
        manifest.get("schema") != "avgaussianv2.lre-loss-ablation"
        or manifest.get("version") != 1
    ):
        raise ValueError("unsupported LRE ablation manifest")
    if stage not in {"screening", "confirmation", "robustness"}:
        raise ValueError("stage must be screening, confirmation, or robustness")
    fixed = manifest["fixed_loss"]
    stage_config = manifest[stage]
    seeds = tuple(int(seed) for seed in stage_config["seeds"])
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
    stage_dir.mkdir(parents=True, exist_ok=True)
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
        scenes = manifest.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("scenes must be a nonempty list")
        for scene in scenes:
            relative = specification.get(scene)
            if not isinstance(relative, str):
                raise ValueError(f"architecture {system} has no config for {scene}")
            base_path = (manifest_path.parent / relative).resolve()
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
                    derived["train"]["joint_steps"] = max_steps
                    derived["benchmark"]["seed"] = int(seed)
                    derived["benchmark"]["continuation_updates"] = max_steps
                    derived["benchmark"]["report_steps"] = list(report_steps)
                    config_id = (
                        f"{stage}__{system}__{scene}__seed{seed}"
                        f"__lre{_weight_slug(weight)}"
                    )
                    destination = stage_dir / f"{config_id}.yaml"
                    _rebase_paths(
                        derived,
                        source_directory=base_path.parent,
                        destination_directory=destination.parent,
                    )
                    data = yaml.safe_dump(derived, sort_keys=False).encode()
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
                    run_id = config_id
                    control_run_id = (
                        None
                        if weight == 0.0
                        else (
                            f"{stage}__{system}__{scene}__seed{seed}"
                            f"__lre{_weight_slug(0.0)}"
                        )
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "stage": stage,
                            "config_id": config_id,
                            "scene": scene,
                            "system": system,
                            "training_mode": training_mode,
                            "seed": seed,
                            "lambda_lre": weight,
                            "control_run_id": control_run_id,
                            "report_steps": list(report_steps),
                            "max_steps": max_steps,
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
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "configs/experiments/lre_loss_ablation.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "configs/generated/lre_loss_ablation",
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "confirmation", "robustness"),
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
    args = parser.parse_args()
    result = generate(
        args.manifest,
        args.output_dir,
        stage=args.stage,
        winners_path=args.winners,
        systems=tuple(args.systems or ()),
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
