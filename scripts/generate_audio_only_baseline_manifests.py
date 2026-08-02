from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "avgaussianv2.audio-only-baseline-manifests"
RUN_MANIFEST_SCHEMA = "avgaussianv2.lre-loss-run-manifest"
SCENES = ("scene1_opera", "Scene7playing")
SOURCE_SEED = 42
ROBUSTNESS_SEEDS = (17, 73)
SEED_TOKEN = re.compile(r"__seed42__")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read source manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("source manifest must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return value


def _validate_repository(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"root", "commit", "clean"}
        or not isinstance(value.get("root"), str)
        or not Path(value["root"]).is_absolute()
        or not isinstance(value.get("commit"), str)
        or len(value["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in value["commit"])
        or value.get("clean") is not True
    ):
        raise ValueError(
            "source manifest repository identity must be clean and complete"
        )
    return dict(value)


def _leaf_differences(
    left: object, right: object, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_leaf_differences(left[key], right[key], path))
        return differences
    return set() if left == right else {prefix}


def _source_audio_only_records(
    manifest: Mapping[str, object],
) -> list[tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]]:
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("stage") != "screening"
        or not isinstance(manifest.get("configs"), list)
        or not isinstance(manifest.get("runs"), list)
    ):
        raise ValueError("source must be a version-1 screening run manifest")

    config_by_id: dict[str, dict[str, Any]] = {}
    for raw_config in manifest["configs"]:
        if not isinstance(raw_config, dict):
            raise ValueError("source config records must be mappings")
        config_id = raw_config.get("config_id")
        if not isinstance(config_id, str) or config_id in config_by_id:
            raise ValueError("source config IDs must be unique strings")
        config_by_id[config_id] = raw_config

    selected: list[tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]] = []
    for raw_run in manifest["runs"]:
        if not isinstance(raw_run, dict):
            raise ValueError("source run records must be mappings")
        if not (
            raw_run.get("system") == "audio_only"
            and raw_run.get("training_mode") == "audio_only"
            and raw_run.get("seed") == SOURCE_SEED
            and float(raw_run.get("lambda_lre", -1.0)) == 0.0
        ):
            continue
        config_id = raw_run.get("config_id")
        config = config_by_id.get(config_id) if isinstance(config_id, str) else None
        if config is None:
            raise ValueError(f"source run has no config record: {config_id!r}")
        for field in ("scene", "system", "training_mode", "seed", "lambda_lre"):
            if raw_run.get(field) != config.get(field):
                raise ValueError(f"source run/config mismatch for {config_id}: {field}")
        config_path_raw = config.get("config")
        config_digest = config.get("config_sha256")
        if not isinstance(config_path_raw, str) or not isinstance(config_digest, str):
            raise ValueError(f"source config path/hash missing: {config_id}")
        config_path = Path(config_path_raw).resolve()
        if not config_path.is_file() or _sha256(config_path) != config_digest:
            raise ValueError(f"source config hash mismatch: {config_id}")
        config_value = _load_yaml(config_path)
        train = config_value.get("train")
        benchmark = config_value.get("benchmark")
        if (
            not isinstance(train, dict)
            or not isinstance(benchmark, dict)
            or train.get("seed") != SOURCE_SEED
            or benchmark.get("seed") != SOURCE_SEED
            or float(train.get("lambda_lre", -1.0)) != 0.0
            or train.get("joint_steps") != 30_000
            or benchmark.get("continuation_updates") != 30_000
            or benchmark.get("report_steps") != [5_000, 10_000, 30_000]
        ):
            raise ValueError(
                f"source config is not the fixed 30k Audio-only protocol: {config_id}"
            )
        selected.append((raw_run, config, config_path, config_value))

    if {record[0].get("scene") for record in selected} != set(SCENES) or len(
        selected
    ) != 2:
        raise ValueError(
            "source must contain exactly one seed42 Audio-only/lambda=0 run per scene"
        )
    return sorted(selected, key=lambda record: SCENES.index(str(record[0]["scene"])))


def _config_for_seed(
    source_record: dict[str, Any],
    source_path: Path,
    source_value: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if seed == SOURCE_SEED:
        return copy.deepcopy(source_record)
    source_id = source_record["config_id"]
    if not isinstance(source_id, str) or len(SEED_TOKEN.findall(source_id)) != 1:
        raise ValueError(f"source config ID has no unique seed42 token: {source_id!r}")
    config_id = SEED_TOKEN.sub(f"__seed{seed}__", source_id)
    destination = source_path.parent / f"{config_id}.yaml"
    derived = copy.deepcopy(source_value)
    derived["train"]["seed"] = seed
    derived["benchmark"]["seed"] = seed
    if _leaf_differences(source_value, derived) != {
        ("train", "seed"),
        ("benchmark", "seed"),
    }:
        raise AssertionError("derived Audio-only config changed outside the seed axis")
    data = yaml.safe_dump(derived, sort_keys=False).encode()
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f"refusing to overwrite changed derived config: {destination}")
    _atomic_write(destination, data)
    record = copy.deepcopy(source_record)
    record.update(
        {
            "config_id": config_id,
            "config": str(destination.resolve()),
            "config_sha256": hashlib.sha256(data).hexdigest(),
            "seed": seed,
        }
    )
    return record


def _run_for_seed(
    source_run: dict[str, Any],
    config_record: dict[str, Any],
    *,
    seed: int,
    stage: str,
) -> dict[str, Any]:
    config_id = config_record["config_id"]
    report_steps = [5_000, 10_000, 30_000] if seed == SOURCE_SEED else [30_000]
    run = copy.deepcopy(source_run)
    run.update(
        {
            "run_id": f"audio_only_baseline_{stage}__{config_id}",
            "continuation_id": config_id,
            "stage": stage,
            "config_id": config_id,
            "seed": seed,
            "report_steps": report_steps,
            "max_steps": 30_000,
            "stop_after_step": None,
        }
    )
    return run


def generate(source_manifest: Path, output_dir: Path) -> dict[str, object]:
    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    source = _load_json(source_manifest)
    repository = _validate_repository(source.get("repository"))
    records = _source_audio_only_records(source)
    generator_sha256 = _sha256(Path(__file__).resolve())

    manifests: dict[str, dict[str, object]] = {}
    for stage, seeds in (
        ("confirmation", (SOURCE_SEED,)),
        ("robustness", ROBUSTNESS_SEEDS),
    ):
        configs: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for source_run, source_config, source_path, source_value in records:
            for seed in seeds:
                config = _config_for_seed(
                    source_config, source_path, source_value, seed=seed
                )
                configs.append(config)
                runs.append(_run_for_seed(source_run, config, seed=seed, stage=stage))
        manifest: dict[str, object] = {
            "schema": RUN_MANIFEST_SCHEMA,
            "version": 1,
            "stage": stage,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "repository": repository,
            "generator": {
                "schema": SCHEMA,
                "version": 1,
                "script": str(Path(__file__).resolve()),
                "script_sha256": generator_sha256,
                "only_derived_axis": "train.seed == benchmark.seed",
            },
            "configs": configs,
            "runs": runs,
        }
        path = output_dir / stage / "manifest.json"
        _atomic_write(path, _json_bytes(manifest))
        manifests[stage] = {
            "path": str(path),
            "sha256": _sha256(path),
            "run_count": len(runs),
        }

    index: dict[str, object] = {
        "schema": SCHEMA,
        "version": 1,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "repository": repository,
        "generator_script_sha256": generator_sha256,
        "manifests": manifests,
    }
    _atomic_write(output_dir / "index.json", _json_bytes(index))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate update-matched Audio-only final baseline manifests."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source_manifest, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
