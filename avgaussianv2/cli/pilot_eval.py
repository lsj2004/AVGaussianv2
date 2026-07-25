"""Full-heldout evaluation jobs for the Scene 1 pilot.

The module deliberately keeps production runtime imports behind ``run_evaluation`` so
``python -m ... --help`` never imports configured upstream Python.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from avgaussianv2.config import load_project_config_bytes
from avgaussianv2.experiment.checkpoint import (
    hash_index_manifest,
    inspect_pilot_checkpoint,
    validate_pilot_resume_model,
)
from avgaussianv2.experiment.contracts import EvaluationResult, Variant
from avgaussianv2.experiment.report import (
    EvaluationArtifactProvenance,
    EvaluationProvenance,
    build_evaluation_manifest_sha256,
    build_evaluation_run_id,
    _validate_evaluation,
)


MAX_INPUT_BYTES = 16 * 1024 * 1024
JOB_SCHEMA = "avgaussianv2.pilot-evaluation-job"
JOB_VERSION = 1
_CONDITIONS = {
    "baseline_imported": False,
    "joint_conditioned_on": True,
    "joint_conditioned_off": False,
    "frozen_visual_on": True,
    "condition_off": False,
}
_JOB_FIELDS = {
    "schema", "version", "scene_id", "camera", "config_sha256",
    "source_hashes", "condition_specs", "train_length", "eval_length",
    "runtime_identity", "shared_manifest", "variant", "systems",
}
_SYSTEM_FIELDS = {
    "metrics_per_sample_path", "metrics_per_sample_sha256",
    "metrics_summary_path", "metrics_summary_sha256", "manifest_sha256",
    "system_name", "condition_enabled", "count", "evaluation_indices",
    "evaluation_indices_hash", "checkpoint", "evaluation_run_id",
}
_CHECKPOINT_FIELDS = {
    "scene_id", "checkpoint_path", "checkpoint_sha256",
    "checkpoint_generation", "run_fingerprint", "evaluation_indices_hash",
    "condition_enabled", "evaluation_run_id", "compatibility",
    "variant_indices", "pilot_config",
}


def _json_text_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _json_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _json_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _json_digest(value: object, name: str) -> str:
    result = _json_text_value(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _json_absolute_path(value: object, name: str) -> Path:
    result = Path(_json_text_value(value, name))
    if not result.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if Path(os.path.abspath(result)) != result:
        raise ValueError(f"{name} must be lexically normalized")
    return result


@dataclass(frozen=True)
class EvaluationSpec:
    system_name: str
    condition_enabled: bool


@dataclass(frozen=True)
class EvaluationJobResult:
    manifest_path: Path
    artifacts: tuple[EvaluationArtifactProvenance, ...]
    evaluations: tuple[EvaluationResult, ...]


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


def parse_evaluation_spec(value: str) -> EvaluationSpec:
    """Parse the explicit ``SYSTEM:on|off`` interface and enforce Task 8's matrix."""
    try:
        name, condition_text = value.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("system must be SYSTEM:on or SYSTEM:off") from error
    if condition_text not in {"on", "off"}:
        raise ValueError("condition must be on or off")
    condition = condition_text == "on"
    if name not in _CONDITIONS:
        raise ValueError(f"unsupported pilot system: {name!r}")
    if _CONDITIONS[name] is not condition:
        raise ValueError(f"condition mismatch for {name}")
    return EvaluationSpec(name, condition)


def _read_regular(path: Path, limit: int = MAX_INPUT_BYTES) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a non-symlink regular file: {path}")
    if before.st_size > limit:
        raise ValueError(f"input exceeds {limit} byte limit: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        blocks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
            blocks.append(block)
            total += len(block)
            if total > limit:
                break
        data = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) > limit:
        raise ValueError(f"input exceeds {limit} byte limit: {path}")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"input changed while reading: {path}")
    return data


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError(f"artifact must be a single-link regular file: {path}")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()


def _snapshot(path: Path) -> SourceSnapshot:
    original = path.lstat()
    if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
        raise ValueError(f"source must be a non-symlink regular file: {path}")
    resolved = path.resolve(strict=True)
    metadata = resolved.lstat()
    return SourceSnapshot(
        resolved,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        _sha256_file(resolved),
    )


def _verify_snapshot(snapshot: SourceSnapshot) -> None:
    if _snapshot(snapshot.path) != snapshot:
        raise ValueError(f"source changed during evaluation: {snapshot.path}")


def _atomic_json(path: Path, value: object) -> None:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _strict_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(
            _read_regular(path).decode(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _preflight_output(output: Path, *, resume: bool, overwrite: bool) -> None:
    current = Path(output.anchor)
    for part in output.parts[1:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"output parent is unsafe: {current}")
    if output.exists():
        metadata = output.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"output must be a non-symlink directory: {output}")
        entries = [item for item in output.iterdir() if item.name != ".evaluation.lock"]
        if entries and not (resume or overwrite):
            raise FileExistsError(f"evaluation output is nonempty: {output}")
    else:
        output.mkdir(parents=True)


def _verify_existing(
    output: Path,
    expected_specs: Sequence[EvaluationSpec],
    *,
    config_path: Path | None = None,
    shared_manifest: Path | None = None,
    variant: Variant | None = None,
) -> EvaluationJobResult:
    raw = _strict_json(output / "evaluation_manifest.json")
    if set(raw) != _JOB_FIELDS:
        raise ValueError("evaluation manifest fields mismatch")
    if raw.get("schema") != JOB_SCHEMA or raw.get("version") != JOB_VERSION:
        raise ValueError("evaluation manifest schema/version mismatch")
    scene_id = _json_text_value(raw["scene_id"], "scene_id")
    config_sha256 = _json_digest(raw["config_sha256"], "config_sha256")
    if not isinstance(raw["camera"], list) or any(
        not isinstance(camera, str) or not camera for camera in raw["camera"]
    ):
        raise TypeError("camera must be a list of nonempty strings")
    if not isinstance(raw["systems"], list):
        raise TypeError("systems must be a list")
    if not isinstance(raw["condition_specs"], list):
        raise TypeError("condition_specs must be a list")
    train_length = _json_int(raw["train_length"], "train_length", minimum=1)
    eval_length = _json_int(raw["eval_length"], "eval_length", minimum=1)
    source_fields = {
        "project_config_sha256", "dataset_manifest_sha256",
        "visual_checkpoint_sha256", "audio_checkpoint_sha256",
        "camera_mapping_sha256",
    }
    if not isinstance(raw["source_hashes"], Mapping) or set(
        raw["source_hashes"]
    ) != source_fields:
        raise ValueError("source_hashes fields mismatch")
    for name in source_fields:
        _json_digest(raw["source_hashes"][name], f"source_hashes.{name}")
    runtime_identity = raw["runtime_identity"]
    if (
        not isinstance(runtime_identity, Mapping)
        or set(runtime_identity) != {"model_class", "model_format_version"}
        or any(
            not isinstance(runtime_identity[name], str) or not runtime_identity[name]
            for name in runtime_identity
        )
    ):
        raise ValueError("evaluation runtime identity is invalid")
    if (
        isinstance(raw["train_length"], bool)
        or not isinstance(raw["train_length"], int)
        or train_length <= 0
        or isinstance(raw["eval_length"], bool)
        or not isinstance(raw["eval_length"], int)
        or eval_length <= 0
    ):
        raise ValueError("evaluation dataset lengths are invalid")
    expected_baseline_path: Path | None = None
    if config_path is not None:
        config_bytes = _read_regular(config_path)
        config = load_project_config_bytes(config_bytes, base_dir=config_path.parent)
        if scene_id != config.scene.scene_id:
            raise ValueError("evaluation scene identity mismatch")
        if config_sha256 != hashlib.sha256(config_bytes).hexdigest():
            raise ValueError("evaluation config hash mismatch")
        if raw["camera"] != list(config.scene.eval_cameras):
            raise ValueError("evaluation camera identity mismatch")
        actual_sources = {
            "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "dataset_manifest_sha256": _sha256_file(
                config.paths.manifest.resolve(strict=True)
            ),
            "visual_checkpoint_sha256": _sha256_file(
                config.paths.visual_checkpoint.resolve(strict=True)
            ),
            "audio_checkpoint_sha256": _sha256_file(
                config.paths.audio_checkpoint.resolve(strict=True)
            ),
            "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
        }
        if raw["source_hashes"] != actual_sources:
            raise ValueError("evaluation source hashes mismatch")
        expected_baseline_path = config.paths.visual_checkpoint.resolve(strict=True)
    expected_shared = (
        None
        if shared_manifest is None
        else {
            "path": str(shared_manifest.resolve(strict=True)),
            "sha256": _sha256_file(shared_manifest.resolve(strict=True)),
        }
    )
    if raw["shared_manifest"] != expected_shared:
        raise ValueError("evaluation shared manifest identity mismatch")
    if raw["variant"] != (None if variant is None else variant.value):
        raise ValueError("evaluation variant identity mismatch")
    if shared_manifest is not None:
        shared = _strict_json(shared_manifest)
        if (
            raw["scene_id"] != shared.get("scene_id")
            or raw["source_hashes"] != shared.get("source_hashes")
            or raw["runtime_identity"] != shared.get("runtime_identity")
            or {
                "train": raw["train_length"],
                "eval": raw["eval_length"],
            }
            != shared.get("dataset_lengths")
        ):
            raise ValueError("evaluation/shared runtime identity mismatch")
    recorded_items: list[EvaluationSpec] = []
    for position, item in enumerate(raw["systems"]):
        if not isinstance(item, Mapping):
            raise TypeError(f"systems[{position}] must be an object")
        recorded_items.append(
            EvaluationSpec(
                _json_text_value(
                    item.get("system_name"), f"systems[{position}].system_name"
                ),
                _json_bool(
                    item.get("condition_enabled"),
                    f"systems[{position}].condition_enabled",
                ),
            )
        )
    recorded = tuple(recorded_items)
    if recorded != tuple(expected_specs):
        raise ValueError("existing evaluation systems mismatch")
    if raw["condition_specs"] != [
        asdict(spec) for spec in expected_specs
    ]:
        raise ValueError("evaluation condition specs mismatch")
    artifacts: list[EvaluationArtifactProvenance] = []
    evaluations: list[EvaluationResult] = []
    for item in raw["systems"]:
        if not isinstance(item, Mapping) or set(item) != _SYSTEM_FIELDS:
            raise ValueError("evaluation system record fields mismatch")
        if not isinstance(item["condition_enabled"], bool):
            raise TypeError("evaluation condition_enabled must be boolean")
        system_name = _json_text_value(item["system_name"], "system_name")
        condition_enabled = _json_bool(
            item["condition_enabled"], f"{system_name}.condition_enabled"
        )
        count = _json_int(item["count"], f"{system_name}.count", minimum=1)
        _json_digest(
            item["metrics_per_sample_sha256"],
            f"{system_name}.metrics_per_sample_sha256",
        )
        _json_digest(
            item["metrics_summary_sha256"],
            f"{system_name}.metrics_summary_sha256",
        )
        _json_digest(item["manifest_sha256"], f"{system_name}.manifest_sha256")
        _json_digest(
            item["evaluation_indices_hash"],
            f"{system_name}.evaluation_indices_hash",
        )
        _json_digest(
            item["evaluation_run_id"], f"{system_name}.evaluation_run_id"
        )
        system_dir = output / system_name
        rows_path = system_dir / "metrics_per_sample.jsonl"
        summary_path = system_dir / "metrics_summary.json"
        if (
            _json_absolute_path(
                item["metrics_per_sample_path"],
                f"{system_name}.metrics_per_sample_path",
            )
            != rows_path.resolve()
            or _json_absolute_path(
                item["metrics_summary_path"],
                f"{system_name}.metrics_summary_path",
            )
            != summary_path.resolve()
        ):
            raise ValueError(f"{item['system_name']} artifact path mismatch")
        if _sha256_file(rows_path) != item["metrics_per_sample_sha256"]:
            raise ValueError(f"{item['system_name']} rows hash mismatch")
        if _sha256_file(summary_path) != item["metrics_summary_sha256"]:
            raise ValueError(f"{item['system_name']} summary hash mismatch")
        rows = tuple(json.loads(line) for line in _read_regular(rows_path).decode().splitlines())
        summary = dict(_strict_json(summary_path))
        if count != len(rows) or count != eval_length:
            raise ValueError(f"{item['system_name']} is not a full-heldout evaluation")
        if not isinstance(item["evaluation_indices"], list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in item["evaluation_indices"]
        ):
            raise TypeError(f"{system_name} evaluation_indices must be integers")
        expected_indices = tuple(range(eval_length))
        if tuple(item["evaluation_indices"]) != expected_indices:
            raise ValueError(f"{item['system_name']} evaluation indices mismatch")
        if item["evaluation_indices_hash"] != hash_index_manifest(
            list(expected_indices)
        ):
            raise ValueError(f"{item['system_name']} index hash mismatch")
        if not _finite_tree((rows, summary)):
            raise ValueError(f"{item['system_name']} contains non-finite values")
        provenance = _artifact_from_json(item, output)
        checkpoint = provenance.checkpoint
        if (
            checkpoint.condition_enabled != provenance.condition_enabled
            or checkpoint.evaluation_run_id != provenance.evaluation_run_id
            or checkpoint.evaluation_indices_hash
            != provenance.evaluation_indices_hash
            or checkpoint.scene_id != raw["scene_id"]
        ):
            raise ValueError(f"{item['system_name']} nested provenance mismatch")
        if _sha256_file(checkpoint.checkpoint_path) != checkpoint.checkpoint_sha256:
            raise ValueError(f"{item['system_name']} checkpoint hash mismatch")
        expected_run_id = build_evaluation_run_id(
            checkpoint_path=checkpoint.checkpoint_path,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            checkpoint_generation=checkpoint.checkpoint_generation,
            run_fingerprint=checkpoint.run_fingerprint,
            evaluation_indices_hash=checkpoint.evaluation_indices_hash,
        )
        if expected_run_id != provenance.evaluation_run_id:
            raise ValueError(f"{item['system_name']} evaluation run ID mismatch")
        expected_manifest_sha = build_evaluation_manifest_sha256(
            metrics_per_sample_sha256=provenance.metrics_per_sample_sha256,
            metrics_summary_sha256=provenance.metrics_summary_sha256,
            system_name=provenance.system_name,
            condition_enabled=provenance.condition_enabled,
            count=provenance.count,
            evaluation_indices=provenance.evaluation_indices,
            evaluation_indices_hash=provenance.evaluation_indices_hash,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            checkpoint_generation=checkpoint.checkpoint_generation,
            evaluation_run_id=provenance.evaluation_run_id,
        )
        if expected_manifest_sha != provenance.manifest_sha256:
            raise ValueError(f"{item['system_name']} provenance manifest hash mismatch")
        if provenance.system_name != "baseline_imported":
            state = inspect_pilot_checkpoint(
                checkpoint.checkpoint_path,
                expected_compatibility=checkpoint.compatibility,
                indices=checkpoint.variant_indices,
                expected_run_fingerprint=checkpoint.run_fingerprint,
                active_resume=False,
            )
            if (
                state.checkpoint_kind != "best"
                or state.generation != checkpoint.checkpoint_generation
            ):
                raise ValueError(f"{item['system_name']} best checkpoint mismatch")
        else:
            expected_inputs = {
                "kind": "imported_visual_baseline",
                "scene_id": raw["scene_id"],
                "visual_checkpoint_sha256": raw["source_hashes"][
                    "visual_checkpoint_sha256"
                ],
                "audio_checkpoint_sha256": raw["source_hashes"][
                    "audio_checkpoint_sha256"
                ],
                "project_config_sha256": raw["config_sha256"],
                "runtime_identity": dict(runtime_identity),
            }
            expected_fingerprint = {
                "algorithm": "avgaussianv2-imported-baseline-v1",
                "sha256": hash_index_manifest(expected_inputs),
                "inputs": expected_inputs,
            }
            if (
                checkpoint.checkpoint_generation != 0
                or checkpoint.compatibility is not None
                or checkpoint.variant_indices is not None
                or checkpoint.pilot_config is not None
                or checkpoint.run_fingerprint != expected_fingerprint
                or (
                    expected_baseline_path is not None
                    and checkpoint.checkpoint_path != expected_baseline_path
                )
            ):
                raise ValueError("baseline imported-artifact identity mismatch")
        evaluation = EvaluationResult(
            system_name, count, rows, summary
        )
        _validate_evaluation(evaluation, system_name)
        for row in evaluation.rows:
            if row["scene_id"] != scene_id:
                raise ValueError(f"{system_name} row scene identity mismatch")
            if row["camera"] not in raw["camera"]:
                raise ValueError(f"{system_name} row camera identity mismatch")
        artifacts.append(provenance)
        evaluations.append(evaluation)
    return EvaluationJobResult(
        (output / "evaluation_manifest.json").resolve(),
        tuple(artifacts),
        tuple(evaluations),
    )


def _clean_owned_partial(output: Path, specs: Sequence[EvaluationSpec]) -> None:
    output_fd = os.open(
        output,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    system_fds: dict[str, tuple[int, tuple[str, ...]]] = {}
    root_temps: list[str] = []
    try:
        allowed = {".evaluation.lock", *(spec.system_name for spec in specs)}
        for name in os.listdir(output_fd):
            if name in allowed:
                continue
            if name.startswith(".evaluation_manifest.") and name.endswith(".tmp"):
                metadata = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError(f"partial staging file is unsafe: {name}")
                root_temps.append(name)
                continue
            raise ValueError(f"resume refuses unrelated evaluation output: {name}")
        for spec in specs:
            name = spec.system_name
            try:
                before = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"partial system output is unsafe: {name}")
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=output_fd,
            )
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(descriptor)
                raise ValueError(f"partial system output changed: {name}")
            children = tuple(os.listdir(descriptor))
            try:
                for child in children:
                    known = child in {
                        ".metrics-publication.lock",
                        "metrics_per_sample.jsonl",
                        "metrics_summary.json",
                    } or (
                        child.startswith(
                            (
                                ".metrics_per_sample.jsonl.",
                                ".metrics_summary.json.",
                            )
                        )
                        and child.endswith((".tmp", ".backup", ".restore"))
                    )
                    metadata = os.stat(
                        child, dir_fd=descriptor, follow_symlinks=False
                    )
                    if (
                        not known
                        or not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_nlink != 1
                    ):
                        raise ValueError(
                            f"partial system file is unsafe: {name}/{child}"
                        )
            except BaseException:
                os.close(descriptor)
                raise
            system_fds[name] = (descriptor, children)
        # No removal begins until every owned entry has passed validation.
        for name in root_temps:
            os.unlink(name, dir_fd=output_fd)
        for name, (descriptor, children) in system_fds.items():
            for child in children:
                os.unlink(child, dir_fd=descriptor)
            os.fsync(descriptor)
            os.close(descriptor)
            system_fds[name] = (-1, ())
            os.rmdir(name, dir_fd=output_fd)
        os.fsync(output_fd)
    finally:
        for descriptor, _ in system_fds.values():
            if descriptor >= 0:
                os.close(descriptor)
        os.close(output_fd)


def _clean_overwrite(output: Path) -> None:
    entries = [item for item in output.iterdir() if item.name != ".evaluation.lock"]
    for item in entries:
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"overwrite refuses unsafe symlink: {item}")
    for item in entries:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return False


def _artifact_from_json(
    item: Mapping[str, object], output: Path
) -> EvaluationArtifactProvenance:
    checkpoint = item["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    if set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("evaluation checkpoint provenance fields mismatch")
    condition = _json_bool(
        checkpoint["condition_enabled"], "checkpoint.condition_enabled"
    )
    scene_id = _json_text_value(checkpoint["scene_id"], "checkpoint.scene_id")
    checkpoint_path = _json_absolute_path(
        checkpoint["checkpoint_path"], "checkpoint.checkpoint_path"
    )
    checkpoint_sha = _json_digest(
        checkpoint["checkpoint_sha256"], "checkpoint.checkpoint_sha256"
    )
    generation = _json_int(
        checkpoint["checkpoint_generation"],
        "checkpoint.checkpoint_generation",
    )
    indices_hash = _json_digest(
        checkpoint["evaluation_indices_hash"],
        "checkpoint.evaluation_indices_hash",
    )
    run_id = _json_digest(
        checkpoint["evaluation_run_id"], "checkpoint.evaluation_run_id"
    )
    if not isinstance(checkpoint["run_fingerprint"], Mapping):
        raise TypeError("checkpoint.run_fingerprint must be an object")
    compatibility = checkpoint.get("compatibility")
    pilot_config = checkpoint.get("pilot_config")
    variant_indices = checkpoint.get("variant_indices")
    if compatibility is not None:
        from avgaussianv2.experiment.checkpoint import PilotCompatibility
        from avgaussianv2.experiment.contracts import PilotConfig, VariantIndices

        compatibility = PilotCompatibility.from_mapping(compatibility)
        if not isinstance(pilot_config, Mapping):
            raise TypeError("checkpoint.pilot_config must be an object")
        pilot_config = PilotConfig(**pilot_config)
        if not isinstance(variant_indices, Mapping) or set(variant_indices) != {
            "warmup", "joint"
        }:
            raise TypeError("checkpoint.variant_indices must be an exact object")
        for name in ("warmup", "joint"):
            if not isinstance(variant_indices[name], list) or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in variant_indices[name]
            ):
                raise TypeError(f"checkpoint.variant_indices.{name} is invalid")
        variant_indices = VariantIndices(
            warmup=tuple(variant_indices["warmup"]),
            joint=tuple(variant_indices["joint"]),
        )
    provenance = EvaluationProvenance(
        scene_id=scene_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_generation=generation,
        run_fingerprint=dict(checkpoint["run_fingerprint"]),
        evaluation_indices_hash=indices_hash,
        condition_enabled=condition,
        evaluation_run_id=run_id,
        compatibility=compatibility,
        variant_indices=variant_indices,
        pilot_config=pilot_config,
    )
    name = _json_text_value(item["system_name"], "system_name")
    return EvaluationArtifactProvenance(
        metrics_per_sample_path=(output / name / "metrics_per_sample.jsonl").resolve(),
        metrics_per_sample_sha256=_json_digest(
            item["metrics_per_sample_sha256"], "metrics_per_sample_sha256"
        ),
        metrics_summary_path=(output / name / "metrics_summary.json").resolve(),
        metrics_summary_sha256=_json_digest(
            item["metrics_summary_sha256"], "metrics_summary_sha256"
        ),
        manifest_sha256=_json_digest(item["manifest_sha256"], "manifest_sha256"),
        system_name=name,
        condition_enabled=_json_bool(item["condition_enabled"], "condition_enabled"),
        count=_json_int(item["count"], "count", minimum=1),
        evaluation_indices=tuple(item["evaluation_indices"]),
        evaluation_indices_hash=_json_digest(
            item["evaluation_indices_hash"], "evaluation_indices_hash"
        ),
        checkpoint=provenance,
        evaluation_run_id=_json_digest(item["evaluation_run_id"], "evaluation_run_id"),
    )


def run_evaluation(
    config_path: str | Path,
    output_dir: str | Path,
    systems: Sequence[EvaluationSpec | str],
    *,
    checkpoint: str | Path | None = None,
    manifest: str | Path | None = None,
    variant: Variant | str | None = None,
    device: str = "cuda:0",
    resume: bool = False,
    overwrite: bool = False,
    trust_upstream_artifacts: bool = False,
    runtime_factory: Callable[..., Any] | None = None,
    evaluator_factory: Callable[..., Any] | None = None,
) -> EvaluationJobResult:
    """Build one runtime, optionally restore one best checkpoint, and evaluate."""
    specs = tuple(
        parse_evaluation_spec(item) if isinstance(item, str) else item
        for item in systems
    )
    if not specs or len({item.system_name for item in specs}) != len(specs):
        raise ValueError("evaluation systems must be nonempty and unique")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    for item in specs:
        if _CONDITIONS.get(item.system_name) is not item.condition_enabled:
            raise ValueError(f"invalid condition matrix for {item.system_name}")
    if checkpoint is None:
        if specs != (EvaluationSpec("baseline_imported", False),):
            raise ValueError("no checkpoint permits baseline_imported:off only")
        if manifest is not None or variant is not None:
            raise ValueError("baseline evaluation does not accept manifest/variant")
    else:
        if any(item.system_name == "baseline_imported" for item in specs):
            raise ValueError("checkpoint evaluation cannot import baseline")
        if manifest is None or variant is None:
            raise ValueError("checkpoint evaluation requires manifest and variant")
        resolved_variant = Variant(variant)
        allowed = {
            Variant.JOINT_CONDITIONED: {
                "joint_conditioned_on",
                "joint_conditioned_off",
            },
            Variant.FROZEN_VISUAL: {"frozen_visual_on"},
            Variant.CONDITION_OFF: {"condition_off"},
        }[resolved_variant]
        if {item.system_name for item in specs} != allowed:
            raise ValueError(f"systems do not match variant {resolved_variant.value}")

    if runtime_factory is None and not trust_upstream_artifacts:
        raise PermissionError(
            "production evaluation requires --trust-upstream-artifacts"
        )
    if device != "cpu":
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("device must be cpu or cuda:<nonnegative-index>")
        import torch

        index = int(device[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {device}")

    config_source = Path(config_path)
    config_snapshot = _snapshot(config_source)
    config_bytes = _read_regular(config_snapshot.path)
    config = load_project_config_bytes(config_bytes, base_dir=config_source.parent)
    checkpoint_state = None
    compatibility = variant_indices = pilot_config = None
    checkpoint_path: Path
    checkpoint_sha: str
    checkpoint_generation: int
    fingerprint: Mapping[str, object]
    baseline_audio_sha: str | None = None
    dataset_snapshot = _snapshot(config.paths.manifest)
    visual_snapshot = _snapshot(config.paths.visual_checkpoint)
    audio_snapshot = _snapshot(config.paths.audio_checkpoint)
    shared_snapshot = None if manifest is None else _snapshot(Path(manifest))
    source_snapshots = (
        config_snapshot, dataset_snapshot, visual_snapshot, audio_snapshot,
        *((shared_snapshot,) if shared_snapshot is not None else ()),
    )
    config = replace(
        config,
        paths=replace(
            config.paths,
            manifest=dataset_snapshot.path,
            visual_checkpoint=visual_snapshot.path,
            audio_checkpoint=audio_snapshot.path,
        ),
    )
    if checkpoint is None:
        checkpoint_path = visual_snapshot.path
        checkpoint_sha = visual_snapshot.sha256
        baseline_audio_sha = audio_snapshot.sha256
        source_hashes = {
            "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "dataset_manifest_sha256": dataset_snapshot.sha256,
            "visual_checkpoint_sha256": checkpoint_sha,
            "audio_checkpoint_sha256": baseline_audio_sha,
            "camera_mapping_sha256": hash_index_manifest(
                config.scene.camera_mapping
            ),
        }
    else:
        from avgaussianv2.cli.pilot_worker import load_worker_manifest

        worker_manifest = load_worker_manifest(
            shared_snapshot.path, config_path=config_snapshot.path, config=config
        )
        resolved_variant = Variant(variant)
        compatibility = worker_manifest.compatibility[resolved_variant]
        variant_indices = worker_manifest.shared_indices.for_variant(
            resolved_variant
        )
        pilot_config = worker_manifest.pilot_config
        source_hashes = dict(worker_manifest.source_hashes)
        checkpoint_snapshot = _snapshot(Path(checkpoint))
        source_snapshots = (*source_snapshots, checkpoint_snapshot)
        checkpoint_path = checkpoint_snapshot.path
        checkpoint_sha = checkpoint_snapshot.sha256
        checkpoint_state = inspect_pilot_checkpoint(
            checkpoint_path,
            expected_compatibility=compatibility,
            indices=variant_indices,
            model=None,
            active_resume=False,
        )
        if checkpoint_state.checkpoint_kind != "best":
            raise ValueError("evaluation checkpoint must be kind best")
        checkpoint_generation = checkpoint_state.generation
        fingerprint = checkpoint_state.run_fingerprint
    output = Path(output_dir).absolute()
    _preflight_output(output, resume=resume, overwrite=overwrite)
    lock_fd = os.open(
        output / ".evaluation.lock",
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("evaluation output already has an active writer") from error
        if overwrite:
            _clean_overwrite(output)
        if resume:
            manifest_path = output / "evaluation_manifest.json"
            if manifest_path.exists():
                return _verify_existing(
                    output,
                    specs,
                    config_path=config_snapshot.path,
                    shared_manifest=(
                        None if shared_snapshot is None else shared_snapshot.path
                    ),
                    variant=(None if variant is None else Variant(variant)),
                )
            _clean_owned_partial(output, specs)

        if runtime_factory is None:
            from avgaussianv2.runtime import build_runtime

            bundle = build_runtime(
                config,
                device,
                trusted_upstream_artifacts=True,
                include_eval=True,
            )
        else:
            bundle = runtime_factory(config, device)
        if bundle.eval_samples is None or not bundle.eval_samples:
            raise ValueError("evaluation runtime requires a nonempty eval split")
        indices = tuple(range(len(bundle.eval_samples)))
        indices_hash = hash_index_manifest(list(indices))

        if checkpoint is None:
            if _sha256_file(checkpoint_path) != checkpoint_sha:
                raise ValueError("visual checkpoint changed while constructing runtime")
            if (
                _sha256_file(config.paths.audio_checkpoint.resolve(strict=True))
                != baseline_audio_sha
            ):
                raise ValueError("audio checkpoint changed while constructing runtime")
            checkpoint_generation = 0
            model_type = type(bundle.model)
            runtime_identity = {
                "model_class": f"{model_type.__module__}.{model_type.__qualname__}",
                "model_format_version": str(
                    getattr(bundle.model, "checkpoint_format_version", "state-dict-v1")
                ),
            }
            fingerprint_inputs = {
                "kind": "imported_visual_baseline",
                "scene_id": config.scene.scene_id,
                "visual_checkpoint_sha256": checkpoint_sha,
                "audio_checkpoint_sha256": baseline_audio_sha,
                "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "runtime_identity": runtime_identity,
            }
            fingerprint = {
                "algorithm": "avgaussianv2-imported-baseline-v1",
                "sha256": hash_index_manifest(fingerprint_inputs),
                "inputs": fingerprint_inputs,
            }
        else:
            if _sha256_file(checkpoint_path) != checkpoint_sha:
                raise ValueError("checkpoint changed while constructing runtime")
            validate_pilot_resume_model(checkpoint_state, bundle.model)
            bundle.model.load_state_dict(checkpoint_state.model_state_dict, strict=True)

        run_id = build_evaluation_run_id(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_generation=checkpoint_generation,
            run_fingerprint=fingerprint,
            evaluation_indices_hash=indices_hash,
        )
        if evaluator_factory is None:
            from avgaussianv2.experiment.evaluation import Evaluator

            evaluator_factory = Evaluator
        evaluator = evaluator_factory(bundle.model, bundle.audio_loss_fn, device)
        if getattr(evaluator, "model", bundle.model) is not bundle.model:
            raise ValueError("evaluator must share the single runtime model")

        system_records: list[dict[str, object]] = []
        evaluations: list[EvaluationResult] = []
        artifacts: list[EvaluationArtifactProvenance] = []
        for spec in specs:
            system_dir = output / spec.system_name
            if system_dir.exists():
                raise FileExistsError(f"system output already exists: {system_dir}")
            result = evaluator.evaluate(
                bundle.eval_samples,
                indices,
                spec.system_name,
                spec.condition_enabled,
                system_dir,
            )
            for snapshot in source_snapshots:
                _verify_snapshot(snapshot)
            if result.count != len(indices) or not _finite_tree(asdict(result)):
                raise ValueError("evaluator did not produce a finite full-heldout result")
            rows_path = system_dir / "metrics_per_sample.jsonl"
            summary_path = system_dir / "metrics_summary.json"
            rows_sha = _sha256_file(rows_path)
            summary_sha = _sha256_file(summary_path)
            manifest_sha = build_evaluation_manifest_sha256(
                metrics_per_sample_sha256=rows_sha,
                metrics_summary_sha256=summary_sha,
                system_name=spec.system_name,
                condition_enabled=spec.condition_enabled,
                count=result.count,
                evaluation_indices=indices,
                evaluation_indices_hash=indices_hash,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_generation=checkpoint_generation,
                evaluation_run_id=run_id,
            )
            provenance = EvaluationProvenance(
                scene_id=config.scene.scene_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_generation=checkpoint_generation,
                run_fingerprint=fingerprint,
                evaluation_indices_hash=indices_hash,
                condition_enabled=spec.condition_enabled,
                evaluation_run_id=run_id,
                compatibility=compatibility,
                variant_indices=variant_indices,
                pilot_config=pilot_config,
            )
            artifact = EvaluationArtifactProvenance(
                metrics_per_sample_path=rows_path.resolve(),
                metrics_per_sample_sha256=rows_sha,
                metrics_summary_path=summary_path.resolve(),
                metrics_summary_sha256=summary_sha,
                manifest_sha256=manifest_sha,
                system_name=spec.system_name,
                condition_enabled=spec.condition_enabled,
                count=result.count,
                evaluation_indices=indices,
                evaluation_indices_hash=indices_hash,
                checkpoint=provenance,
                evaluation_run_id=run_id,
            )
            checkpoint_json = asdict(provenance)
            for key in ("checkpoint_path",):
                checkpoint_json[key] = str(checkpoint_json[key])
            record = {
                **asdict(artifact),
                "metrics_per_sample_path": str(rows_path.resolve()),
                "metrics_summary_path": str(summary_path.resolve()),
                "checkpoint": checkpoint_json,
            }
            system_records.append(record)
            artifacts.append(artifact)
            evaluations.append(result)
        job = {
            "schema": JOB_SCHEMA,
            "version": JOB_VERSION,
            "scene_id": config.scene.scene_id,
            "camera": list(config.scene.eval_cameras),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_hashes": source_hashes,
            "condition_specs": [
                {
                    "system_name": spec.system_name,
                    "condition_enabled": spec.condition_enabled,
                }
                for spec in specs
            ],
            "train_length": len(bundle.train_samples),
            "eval_length": len(bundle.eval_samples),
            "runtime_identity": {
                "model_class": (
                    f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}"
                ),
                "model_format_version": str(
                    getattr(bundle.model, "checkpoint_format_version", "state-dict-v1")
                ),
            },
            "shared_manifest": (
                None
                if shared_snapshot is None
                else {
                    "path": str(shared_snapshot.path),
                    "sha256": shared_snapshot.sha256,
                }
            ),
            "variant": None if variant is None else Variant(variant).value,
            "systems": system_records,
        }
        for snapshot in source_snapshots:
            _verify_snapshot(snapshot)
        manifest_path = output / "evaluation_manifest.json"
        _atomic_json(manifest_path, job)
        return EvaluationJobResult(
            manifest_path.resolve(), tuple(artifacts), tuple(evaluations)
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a complete Scene 1 pilot split")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--variant", choices=tuple(item.value for item in Variant))
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        type=parse_evaluation_spec,
        help="repeatable SYSTEM:on|off specification",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_evaluation(
        args.config,
        args.output_dir,
        args.system,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        variant=args.variant,
        device=args.device,
        resume=args.resume,
        overwrite=args.overwrite,
        trust_upstream_artifacts=args.trust_upstream_artifacts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
