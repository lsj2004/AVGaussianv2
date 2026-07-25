"""Immutable native AudioGS/FreeTimeGS++ training contracts."""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import stat
import subprocess
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from avgaussianv2.benchmark.artifacts import canonical_json, publish_generation
from avgaussianv2.benchmark.assets import (
    EXPECTED,
    PROTOCOL,
    audit_audiogs_conversion,
    audit_ftgspp_flow_cache,
    audit_ftgspp_seed_record,
    audit_ftgspp_upstream_config,
    audit_initialization_provenance,
    audit_protocol_config,
)
from avgaussianv2.benchmark.training import TEST_CAMERA, TRAIN_CAMERAS
from avgaussianv2.benchmark.output import BenchmarkOutputError, BenchmarkOutputReadLock

SCHEMA = "avgaussianv2.cam38-native-training-contract"
VERSION = 1
AUDIO_SEED_SCHEMA = "avgaussianv2.audiogs-native-seed-record"
AUDIO_SEED_FIELDS = {
    "schema",
    "version",
    "scene_id",
    "upstream_scene",
    "seed",
    "batch_size",
    "max_epochs",
    "test_viewpoint",
    "train_cameras",
}
MODEL_KINDS = {"audiogs", "ftgspp"}
_TOP_FIELDS = {
    "schema",
    "version",
    "model_kind",
    "scene_id",
    "protocol",
    "split",
    "seed",
    "budget",
    "inputs",
    "upstream",
    "checkpoint",
    "completion",
    "derived_initialization",
}


class NativeContractError(RuntimeError):
    pass


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_fd(fd: int, label: str) -> bytes:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
    ):
        raise NativeContractError(f"{label} must be an owned single-link regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise NativeContractError(f"{label} changed while being snapshotted")
    return b"".join(chunks)


def _hash_fd(fd: int, label: str) -> str:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
    ):
        raise NativeContractError(f"{label} must be an owned single-link regular file")
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise NativeContractError(f"{label} changed while being hashed")
    return digest.hexdigest()


def _open_absolute_regular(path: Path, label: str) -> int:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise NativeContractError(f"{label} path must be absolute")
    directory = os.open(
        "/",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        return os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
    except OSError as error:
        raise NativeContractError(f"cannot pin {label}: {error}") from error
    finally:
        os.close(directory)


def _stable_bytes(path: Path, label: str) -> bytes:
    descriptor = _open_absolute_regular(path, label)
    try:
        data = _read_fd(descriptor, label)
        _verify_retained_path(path, descriptor, label)
        return data
    finally:
        os.close(descriptor)


def _verify_retained_path(path: Path, fd: int, label: str) -> None:
    retained = os.fstat(fd)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise NativeContractError(f"{label} disappeared while pinned") from error
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        retained.st_dev,
        retained.st_ino,
    ):
        raise NativeContractError(f"{label} path identity changed while pinned")


def _snapshot_record(
    path: Path, snapshot: str, files: dict[str, bytes], label: str
) -> dict[str, str]:
    data = _stable_bytes(path, label)
    files[snapshot] = data
    return {
        "path": str(Path(os.path.abspath(path))),
        "sha256": _digest_bytes(data),
        "snapshot": snapshot,
    }


def _safe_regular(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise NativeContractError(f"cannot inspect {label}: {error}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
    ):
        raise NativeContractError(f"{label} must be an owned single-link regular file")


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise NativeContractError(f"{label} fields mismatch")
    return value


def _json(path: Path, label: str) -> Mapping[str, Any]:
    _safe_regular(path, label)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise NativeContractError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise NativeContractError(f"{label} must contain a JSON object")
    return value


def _torch_pickle_ops_fd(
    fd: int,
) -> tuple[dict[str, int], set[str], set[str]]:
    """Inspect Torch's pickle opcode stream without executing pickle globals."""
    try:
        with os.fdopen(os.dup(fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
            names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(names) != 1:
                raise NativeContractError(
                    "native checkpoint must contain exactly one Torch data.pkl"
                )
            info = archive.getinfo(names[0])
            if info.file_size > 256 * 1024 * 1024:
                raise NativeContractError(
                    "native checkpoint pickle is unreasonably large"
                )
            data = archive.read(names[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise NativeContractError(
            f"native checkpoint is not a safe Torch zip archive: {error}"
        ) from error
    integers: dict[str, int] = {}
    strings: set[str] = set()
    globals_: set[str] = set()
    pending_key: str | None = None
    try:
        for opcode, argument, _ in pickletools.genops(data):
            if opcode.name == "GLOBAL" and isinstance(argument, str):
                globals_.add(argument)
                pending_key = None
            elif opcode.name in {
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "UNICODE",
                "STRING",
            } and isinstance(argument, str):
                strings.add(argument)
                pending_key = argument
            elif opcode.name in {"BININT", "BININT1", "BININT2", "INT"}:
                if pending_key is not None and isinstance(argument, int):
                    integers[pending_key] = argument
                pending_key = None
            elif opcode.name not in {"MEMOIZE", "BINPUT", "LONG_BINPUT"}:
                pending_key = None
    except (ValueError, pickletools.Error) as error:
        raise NativeContractError(
            f"native checkpoint pickle opcode stream is invalid: {error}"
        ) from error
    return integers, strings, globals_


def inspect_native_checkpoint(
    path: str | Path,
    *,
    model_kind: str,
    scene_id: str,
) -> dict[str, object]:
    checkpoint = Path(path)
    descriptor = _open_absolute_regular(checkpoint, "native checkpoint")
    try:
        _hash_fd(descriptor, "native checkpoint")
        integers, strings, globals_ = _torch_pickle_ops_fd(descriptor)
        _verify_retained_path(checkpoint, descriptor, "native checkpoint")
    finally:
        os.close(descriptor)
    return _checkpoint_metadata(
        integers,
        strings,
        globals_,
        model_kind=model_kind,
        scene_id=scene_id,
    )


def _checkpoint_metadata(
    integers: Mapping[str, int],
    strings: set[str],
    globals_: set[str],
    *,
    model_kind: str,
    scene_id: str,
) -> dict[str, object]:
    if model_kind == "audiogs":
        expected_updates = 2_318 if scene_id == "scene1_opera" else 6_954
        required = {
            "epoch": 60,
            "iter": expected_updates,
            "max_epoch": 61,
            "test_viewpoint": 39,
        }
        if any(integers.get(name) != value for name, value in required.items()):
            raise NativeContractError(
                "AudioGS checkpoint epoch/update/viewpoint metadata mismatch"
            )
        if "audio_3dgs_mono_diff_gs_only" not in strings:
            raise NativeContractError(
                "AudioGS checkpoint model class metadata mismatch"
            )
        model_class = "Audio3DGSMonoDiffGSOnly"
        completion = {
            "epoch_zero_based": 60,
            "max_epochs": 61,
            "resolved_updates": expected_updates,
            "test_viewpoint": 39,
        }
    elif model_kind == "ftgspp":
        required_parameters = {
            "means",
            "scales",
            "quats",
            "opacities",
            "sh_0",
            "sh_n",
            "times",
            "durations",
            "marginal_gates",
        }
        expected_globals = {
            "ftgspp.models.gaussians Gaussians",
            "torch._utils _rebuild_parameter",
            "torch._utils _rebuild_tensor_v2",
            "torch FloatStorage",
            "collections OrderedDict",
            "__builtin__ set",
        }
        if (
            globals_ != expected_globals
            or not {
                *required_parameters,
                "velocity_model",
                "max_duration",
                "sh_degree",
            }.issubset(strings)
            or integers.get("sh_degree") != 3
        ):
            raise NativeContractError(
                "FreeTimeGS++ Gaussians checkpoint schema mismatch"
            )
        model_class = "ftgspp.models.gaussians.Gaussians"
        completion = {"iterations": 30_000}
    else:
        raise NativeContractError(f"unsupported native model kind: {model_kind!r}")
    state_names = sorted(
        value
        for value in strings
        if "." in value
        or value
        in {
            "means",
            "scales",
            "quats",
            "opacities",
            "sh_0",
            "sh_n",
            "times",
            "durations",
            "marginal_gates",
        }
    )
    return {
        "model_class": model_class,
        "metadata": completion,
        "state_schema_sha256": hashlib.sha256(
            json.dumps(state_names, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _git_identity(root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise NativeContractError(
                f"cannot resolve upstream Git identity: {error}"
            ) from error

    status = git("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "root": str(root.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _source_files(kind: str, root: Path) -> tuple[Path, ...]:
    paths = (
        (
            root / "configs" / "audio_3dgs_replaynvas_viewpoint.yaml",
            root / "tools" / "train_audio_3dgs_viewpoint.py",
            *sorted(root.joinpath("libs").rglob("*.py")),
        )
        if kind == "audiogs"
        else tuple(sorted(root.joinpath("ftgspp").rglob("*.py")))
    )
    for path in paths:
        _safe_regular(path, "upstream source audit")
    return paths


def _record_set_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_audiogs_seed_record(
    path: str | Path, *, scene_id: str, upstream_scene: str
) -> dict[str, object]:
    if scene_id not in EXPECTED or upstream_scene != EXPECTED[scene_id]["audio_scene"]:
        raise NativeContractError("AudioGS seed record scene mapping mismatch")
    record: dict[str, object] = {
        "schema": AUDIO_SEED_SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "upstream_scene": upstream_scene,
        "seed": 42,
        "batch_size": 1,
        "max_epochs": 61,
        "test_viewpoint": 39,
        "train_cameras": list(TRAIN_CAMERAS),
    }
    _atomic_json(Path(path), record)
    return record


def _audit_audiogs_seed_record(
    path: str | Path, *, expected_scene: str
) -> dict[str, object]:
    return _audit_audiogs_seed_mapping(
        _json(Path(path), "AudioGS seed record"), expected_scene=expected_scene
    )


def _audit_audiogs_seed_mapping(
    value: object, *, expected_scene: str
) -> dict[str, object]:
    record = _exact(
        value,
        AUDIO_SEED_FIELDS,
        "AudioGS seed record",
    )
    expected = {
        "schema": AUDIO_SEED_SCHEMA,
        "version": 1,
        "scene_id": expected_scene,
        "upstream_scene": EXPECTED[expected_scene]["audio_scene"],
        "seed": 42,
        "batch_size": 1,
        "max_epochs": 61,
        "test_viewpoint": 39,
        "train_cameras": list(TRAIN_CAMERAS),
    }
    if dict(record) != expected:
        raise NativeContractError("AudioGS seed record protocol mismatch")
    return expected


def _flow_inventory(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*.npz")):
        descriptor = _open_absolute_regular(path, "FTGS++ flow artifact")
        try:
            digest = _hash_fd(descriptor, "FTGS++ flow artifact")
            size = os.fstat(descriptor).st_size
            _verify_retained_path(path, descriptor, "FTGS++ flow artifact")
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": digest,
                    "size": size,
                }
            )
        finally:
            os.close(descriptor)
    return {"root": str(root.resolve()), "files": files}


def finalize_native_contract(
    *,
    model_kind: str,
    config_path: str | Path,
    provenance_path: str | Path,
    checkpoint_path: str | Path,
    upstream_root: str | Path,
    output_path: str | Path,
    conversion_manifest: str | Path | None = None,
    rendered_config: str | Path | None = None,
    sampled_scene_root: str | Path | None = None,
    train_log: str | Path | None = None,
    seed_records: Sequence[str | Path] = (),
) -> dict[str, object]:
    if model_kind not in MODEL_KINDS:
        raise NativeContractError("model_kind must be audiogs or ftgspp")
    config = Path(config_path)
    provenance = Path(provenance_path)
    checkpoint = Path(checkpoint_path)
    upstream = Path(upstream_root)
    raw = audit_protocol_config(config)
    scene_id = raw["scene"]["id"]
    audit_initialization_provenance(provenance, expected_scene=scene_id)
    configured_checkpoint = Path(
        raw["paths"][
            "audio_checkpoint" if model_kind == "audiogs" else "visual_checkpoint"
        ]
    )
    if not configured_checkpoint.is_absolute():
        configured_checkpoint = config.parent / configured_checkpoint
    configured_checkpoint = Path(os.path.abspath(configured_checkpoint))
    configured_root = Path(
        raw["paths"][
            "audio_upstream_root" if model_kind == "audiogs" else "visual_upstream_root"
        ]
    )
    if Path(os.path.abspath(checkpoint)) != configured_checkpoint or Path(
        os.path.abspath(upstream)
    ) != Path(os.path.abspath(configured_root)):
        raise NativeContractError(
            "native checkpoint/upstream path differs from protocol"
        )
    checkpoint_fd = _open_absolute_regular(checkpoint, "native checkpoint")
    try:
        checkpoint_sha256 = _hash_fd(checkpoint_fd, "native checkpoint")
        checkpoint_metadata = _checkpoint_metadata(
            *_torch_pickle_ops_fd(checkpoint_fd),
            model_kind=model_kind,
            scene_id=scene_id,
        )
        _verify_retained_path(checkpoint, checkpoint_fd, "native checkpoint")
    finally:
        os.close(checkpoint_fd)
    expected = EXPECTED[scene_id]
    seed_records = tuple(Path(path) for path in seed_records)
    files: dict[str, bytes] = {}
    source_audits = [
        _snapshot_record(
            path, f"source_{index:02d}.snapshot", files, "upstream source audit"
        )
        for index, path in enumerate(_source_files(model_kind, upstream))
    ]
    inputs: dict[str, object] = {
        "protocol_config": _snapshot_record(
            config, "protocol_config.snapshot", files, "protocol config"
        ),
        "provenance": _snapshot_record(
            provenance, "provenance.snapshot", files, "provenance"
        ),
        "seed_records": [
            _snapshot_record(path, f"seed_{index:02d}.snapshot", files, "seed record")
            for index, path in enumerate(seed_records)
        ],
        "source_audits": source_audits,
    }
    if model_kind == "audiogs":
        if conversion_manifest is None or len(seed_records) != 1:
            raise NativeContractError(
                "AudioGS requires conversion manifest and one launch seed record"
            )
        seed_audit = _audit_audiogs_seed_record(
            seed_records[0], expected_scene=scene_id
        )
        conversion = Path(conversion_manifest)
        payload = _json(conversion, "AudioGS conversion manifest")
        audit = audit_audiogs_conversion(
            payload,
            expected_scene=expected["audio_scene"],
            expected_clips=expected["audio_clips"],
            epochs=61,
            expected_audio_root=payload["audio_root"],
            expected_cameras_npz=payload["cameras_npz"],
            expected_output_root=payload["output_root"],
        )
        inputs["conversion_manifest"] = _snapshot_record(
            conversion,
            "conversion_manifest.snapshot",
            files,
            "AudioGS conversion manifest",
        )
        completion = {
            "conversion_audit": audit,
            "seed_audit": seed_audit,
            **checkpoint_metadata["metadata"],
        }
        budget = {"epochs": 61, "resolved_updates": audit["resolved_updates"]}
    else:
        if (
            rendered_config is None
            or sampled_scene_root is None
            or train_log is None
            or len(seed_records) != 3
        ):
            raise NativeContractError(
                "FTGS++ requires rendered config, sampled root, train log, and 3 seed records"
            )
        rendered = Path(rendered_config)
        config_audit = audit_ftgspp_upstream_config(
            rendered,
            protocol_config=config,
            repo_root=config.resolve().parent.parent.parent,
            ftgspp_root=upstream,
            sampled_scene_root=sampled_scene_root,
        )
        seed_audits = [
            audit_ftgspp_seed_record(path, expected_scene=scene_id)
            for path in seed_records
        ]
        stages = {
            tuple(
                audit["argv"][audit["argv"].index(option) + 1]
                for option in ("--from", "--to")
            )
            if "--from" in audit["argv"]
            else ("flow", "flow")
            for audit in seed_audits
        }
        if stages != {("extract", "prep"), ("points", "train"), ("flow", "flow")}:
            raise NativeContractError(
                "FTGS++ seed records do not cover prep/flow/train"
            )
        flow_root = Path(config_audit["namespaces"][-1])
        flow_audit = audit_ftgspp_flow_cache(
            flow_root,
            frame_count=raw["benchmark"]["expected_test_samples"],
            keyframe_stride=10,
        )
        log = Path(train_log)
        _safe_regular(log, "FTGS++ train.log")
        log_text = log.read_text(encoding="utf-8")
        if "Starting training" not in log_text or "Done training" not in log_text:
            raise NativeContractError("FTGS++ train.log lacks completion evidence")
        inputs["rendered_config"] = _snapshot_record(
            rendered, "rendered_config.snapshot", files, "FTGS++ rendered config"
        )
        inputs["train_log"] = _snapshot_record(
            log, "train_log.snapshot", files, "FTGS++ train log"
        )
        inputs["sampled_scene_root"] = str(Path(sampled_scene_root).resolve())
        flow_inventory = _flow_inventory(flow_root)
        flow_inventory_bytes = canonical_json(flow_inventory)
        files["flow_inventory.snapshot"] = flow_inventory_bytes
        inputs["flow_inventory"] = {
            "snapshot": "flow_inventory.snapshot",
            "sha256": _digest_bytes(flow_inventory_bytes),
        }
        completion = {
            "config_audit": config_audit,
            "flow_audit": flow_audit,
            "seed_audits": seed_audits,
            "train_log_markers": ["Starting training", "Done training"],
            **checkpoint_metadata["metadata"],
        }
        budget = {"iterations": 30_000}
    contract: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "model_kind": model_kind,
        "scene_id": scene_id,
        "protocol": PROTOCOL,
        "split": {
            "train_cameras": list(TRAIN_CAMERAS),
            "test_camera": TEST_CAMERA,
        },
        "seed": 42,
        "budget": budget,
        "inputs": inputs,
        "upstream": {
            **_git_identity(upstream),
            "source_sha256": _record_set_sha256(source_audits),
        },
        "checkpoint": {
            "path": str(Path(os.path.abspath(checkpoint))),
            "sha256": checkpoint_sha256,
            **checkpoint_metadata,
        },
        "completion": completion,
    }
    provenance_bytes = files[inputs["provenance"]["snapshot"]]
    audio_basis = (
        files[inputs["conversion_manifest"]["snapshot"]]
        if model_kind == "audiogs"
        else canonical_json({"model_kind": "ftgspp", "audio": "not_applicable"})
    )
    contract["derived_initialization"] = {
        "visual_initialization_sha256": _digest_bytes(provenance_bytes),
        "audio_initialization_sha256": _digest_bytes(audio_basis),
        "model_initialization_sha256": _digest_bytes(
            canonical_json(
                {
                    "model_kind": model_kind,
                    "model_class": checkpoint_metadata["model_class"],
                    "state_schema_sha256": checkpoint_metadata["state_schema_sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_sha256": contract["upstream"]["source_sha256"],
                }
            )
        ),
    }
    files["contract.json"] = canonical_json(contract)
    try:
        _, manifest_digest = publish_generation(
            Path(output_path),
            schema=SCHEMA,
            files=files,
            identity={"scene_id": scene_id, "model_kind": model_kind},
            overwrite=True,
        )
    except Exception as error:
        raise NativeContractError(f"cannot publish native contract: {error}") from error
    return {**contract, "_manifest_sha256": manifest_digest}


def _openat_directory(parent_fd: int, name: str, label: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise NativeContractError(f"cannot open {label}: {error}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        os.close(descriptor)
        raise NativeContractError(f"{label} must be an owned directory")
    return descriptor


def _readat_regular(parent_fd: int, name: str, label: str) -> bytes:
    if not name or Path(name).name != name:
        raise NativeContractError(f"unsafe {label} filename")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise NativeContractError(f"cannot open {label}: {error}") from error
    try:
        data = _read_fd(descriptor, label)
        retained = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
            retained.st_dev,
            retained.st_ino,
        ):
            raise NativeContractError(f"{label} identity changed while pinned")
        return data
    finally:
        os.close(descriptor)


def _load_native_generation(
    pinned: Path,
) -> tuple[dict[str, bytes], str, Mapping[str, object]]:
    root_fd = os.open(
        pinned,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if set(os.listdir(root_fd)) != {
            ".benchmark.lock",
            "current.json",
            "generations",
        }:
            raise NativeContractError(
                "native contract output contains unexpected entries"
            )
        pointer_data = _readat_regular(
            root_fd, "current.json", "native current pointer"
        )
        pointer = json.loads(pointer_data)
        if (
            not isinstance(pointer, Mapping)
            or set(pointer) != {"schema", "version", "generation", "manifest_sha256"}
            or pointer["schema"] != f"{SCHEMA}.current"
            or pointer["version"] != 1
        ):
            raise NativeContractError("native current pointer schema mismatch")
        relative = pointer["generation"]
        if (
            not isinstance(relative, str)
            or Path(relative).parts != ("generations", Path(relative).name)
            or not Path(relative).name.startswith("generation-")
        ):
            raise NativeContractError("native generation pointer is unsafe")
        generations_fd = _openat_directory(root_fd, "generations", "native generations")
        try:
            generation_names = os.listdir(generations_fd)
            if not generation_names or any(
                not name.startswith("generation-") for name in generation_names
            ):
                raise NativeContractError("native generations contain unsafe entries")
            generation_fd = _openat_directory(
                generations_fd, Path(relative).name, "native generation"
            )
        finally:
            os.close(generations_fd)
        try:
            manifest_data = _readat_regular(
                generation_fd, "manifest.json", "native manifest"
            )
            if _digest_bytes(manifest_data) != pointer["manifest_sha256"]:
                raise NativeContractError("native manifest hash mismatch")
            manifest = json.loads(manifest_data)
            if (
                not isinstance(manifest, Mapping)
                or set(manifest) != {"schema", "version", "identity", "sha256"}
                or manifest["schema"] != f"{SCHEMA}.generation"
                or manifest["version"] != 1
                or not isinstance(manifest["identity"], Mapping)
                or not isinstance(manifest["sha256"], Mapping)
            ):
                raise NativeContractError("native generation manifest schema mismatch")
            files = {}
            for name, digest in manifest["sha256"].items():
                if not isinstance(name, str) or not isinstance(digest, str):
                    raise NativeContractError("native manifest file entry is invalid")
                data = _readat_regular(generation_fd, name, f"native snapshot {name}")
                if _digest_bytes(data) != digest:
                    raise NativeContractError(f"native snapshot hash mismatch: {name}")
                files[name] = data
            if set(os.listdir(generation_fd)) != {*files, "manifest.json"}:
                raise NativeContractError("native generation contains unexpected files")
        finally:
            os.close(generation_fd)
    except (OSError, ValueError) as error:
        if isinstance(error, NativeContractError):
            raise
        raise NativeContractError(f"cannot load native generation: {error}") from error
    finally:
        os.close(root_fd)
    return files, pointer["manifest_sha256"], manifest["identity"]


def verify_native_contract(
    path: str | Path,
    *,
    expected_scene: str | None = None,
    expected_model_kind: str | None = None,
) -> dict[str, object]:
    try:
        with BenchmarkOutputReadLock(Path(path)) as pinned:
            files, manifest_sha256, generation_identity = _load_native_generation(
                pinned
            )
            try:
                value = json.loads(files["contract.json"])
            except (KeyError, ValueError) as error:
                raise NativeContractError(
                    f"cannot parse snapshotted native contract: {error}"
                ) from error
            contract = _verify_native_snapshot(
                value,
                files,
                expected_scene=expected_scene,
                expected_model_kind=expected_model_kind,
            )
            if dict(generation_identity) != {
                "scene_id": contract["scene_id"],
                "model_kind": contract["model_kind"],
            }:
                raise NativeContractError("native generation identity mismatch")
    except BenchmarkOutputError as error:
        raise NativeContractError(f"unsafe native contract output: {error}") from error
    return {**contract, "_manifest_sha256": manifest_sha256}


def _verify_native_snapshot(
    value: object,
    files: Mapping[str, bytes],
    *,
    expected_scene: str | None,
    expected_model_kind: str | None,
) -> dict[str, object]:
    contract = _exact(value, _TOP_FIELDS, "native contract")
    if (
        contract["schema"] != SCHEMA
        or contract["version"] != VERSION
        or contract["model_kind"] not in MODEL_KINDS
        or contract["scene_id"] not in EXPECTED
        or contract["protocol"] != PROTOCOL
        or contract["split"]
        != {"train_cameras": list(TRAIN_CAMERAS), "test_camera": TEST_CAMERA}
        or contract["seed"] != 42
    ):
        raise NativeContractError("native contract protocol mismatch")
    if expected_scene is not None and contract["scene_id"] != expected_scene:
        raise NativeContractError("native contract scene mismatch")
    if (
        expected_model_kind is not None
        and contract["model_kind"] != expected_model_kind
    ):
        raise NativeContractError("native contract model kind mismatch")
    inputs = contract["inputs"]
    common_inputs = {
        "protocol_config",
        "provenance",
        "seed_records",
        "source_audits",
    }
    expected_inputs = (
        common_inputs | {"conversion_manifest"}
        if contract["model_kind"] == "audiogs"
        else common_inputs
        | {
            "rendered_config",
            "train_log",
            "sampled_scene_root",
            "flow_inventory",
        }
    )
    inputs = _exact(inputs, expected_inputs, "native contract inputs")

    def snapshot(record_value: object, label: str) -> tuple[Mapping[str, Any], bytes]:
        record = _exact(record_value, {"path", "sha256", "snapshot"}, label)
        name = record["snapshot"]
        if not isinstance(name, str) or name not in files:
            raise NativeContractError(f"{label} snapshot is missing")
        data = files[name]
        if _digest_bytes(data) != record["sha256"]:
            raise NativeContractError(f"{label} snapshot hash mismatch")
        return record, data

    for name in ("protocol_config", "provenance"):
        snapshot(inputs[name], f"native {name}")
    _, config_bytes = snapshot(inputs["protocol_config"], "native protocol config")
    _, provenance_bytes = snapshot(inputs["provenance"], "native provenance")
    try:
        raw = yaml.safe_load(config_bytes)
        provenance_value = json.loads(provenance_bytes)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
        raise NativeContractError(
            f"native protocol snapshot parse failed: {error}"
        ) from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("scene", {}).get("id") != contract["scene_id"]
        or raw.get("scene", {}).get("train_cameras") != list(TRAIN_CAMERAS)
        or raw.get("scene", {}).get("eval_cameras") != [TEST_CAMERA]
        or raw.get("benchmark", {}).get("protocol") != PROTOCOL
        or raw.get("benchmark", {}).get("seed") != 42
    ):
        raise NativeContractError("native protocol config scene mismatch")
    expected_spec = EXPECTED[contract["scene_id"]]
    if raw["benchmark"].get("native_budgets") != {
        "audiogs_epochs": 61,
        "audiogs_batch_size": 1,
        "audiogs_resolved_updates": expected_spec["audio_updates"],
        "ftgspp_updates": 30_000,
        "ftgspp_batch_size": 1,
    }:
        raise NativeContractError("native protocol config budget mismatch")
    config_origin = Path(inputs["protocol_config"]["path"])
    configured_checkpoint = Path(
        raw["paths"][
            "audio_checkpoint"
            if contract["model_kind"] == "audiogs"
            else "visual_checkpoint"
        ]
    )
    if not configured_checkpoint.is_absolute():
        configured_checkpoint = Path(
            os.path.abspath(config_origin.parent / configured_checkpoint)
        )
    configured_upstream = Path(
        raw["paths"][
            "audio_upstream_root"
            if contract["model_kind"] == "audiogs"
            else "visual_upstream_root"
        ]
    )
    if (
        str(configured_checkpoint) != contract["checkpoint"]["path"]
        or str(configured_upstream) != contract["upstream"]["root"]
    ):
        raise NativeContractError("native protocol snapshot path binding mismatch")
    if (
        not isinstance(provenance_value, Mapping)
        or set(provenance_value) != {"scene_id", "test_camera", "assets"}
        or provenance_value["scene_id"] != contract["scene_id"]
        or provenance_value["test_camera"] != TEST_CAMERA
        or not isinstance(provenance_value["assets"], list)
    ):
        raise NativeContractError("native provenance snapshot mismatch")
    for collection in ("seed_records", "source_audits"):
        if not isinstance(inputs.get(collection), list):
            raise NativeContractError(f"native {collection} must be a list")
        for record in inputs[collection]:
            snapshot(record, f"native {collection}")
    expected_source_paths = tuple(
        str(path) for path in _source_files(contract["model_kind"], configured_upstream)
    )
    if tuple(record["path"] for record in inputs["source_audits"]) != (
        expected_source_paths
    ):
        raise NativeContractError("native source audit path set mismatch")
    checkpoint = _exact(
        contract["checkpoint"],
        {"path", "sha256", "model_class", "metadata", "state_schema_sha256"},
        "native checkpoint",
    )
    checkpoint_path = Path(checkpoint["path"])
    upstream = _exact(
        contract["upstream"],
        {
            "root",
            "commit",
            "tree",
            "tracked_worktree_status_sha256",
            "source_sha256",
        },
        "native upstream",
    )
    if upstream["source_sha256"] != _record_set_sha256(inputs["source_audits"]):
        raise NativeContractError("native upstream source aggregate mismatch")
    derived = _exact(
        contract["derived_initialization"],
        {
            "visual_initialization_sha256",
            "audio_initialization_sha256",
            "model_initialization_sha256",
        },
        "native derived initialization",
    )
    if contract["model_kind"] == "audiogs":
        completion = _exact(
            contract["completion"],
            {
                "conversion_audit",
                "seed_audit",
                "epoch_zero_based",
                "max_epochs",
                "resolved_updates",
                "test_viewpoint",
            },
            "native AudioGS completion",
        )
        if contract["budget"] != {
            "epochs": 61,
            "resolved_updates": 2_318
            if contract["scene_id"] == "scene1_opera"
            else 6_954,
        }:
            raise NativeContractError("native AudioGS budget mismatch")
        record = _exact(
            inputs.get("conversion_manifest"),
            {"path", "sha256", "snapshot"},
            "AudioGS conversion manifest",
        )
        _, conversion_bytes = snapshot(record, "AudioGS conversion manifest")
        if len(inputs["seed_records"]) != 1:
            raise NativeContractError(
                "AudioGS native contract seed record count mismatch"
            )
        _, seed_bytes = snapshot(inputs["seed_records"][0], "AudioGS seed record")
        seed_value = json.loads(seed_bytes)
        seed_audit = _audit_audiogs_seed_mapping(
            seed_value, expected_scene=contract["scene_id"]
        )
        if seed_audit != completion["seed_audit"]:
            raise NativeContractError("AudioGS seed audit mismatch")
        payload = json.loads(conversion_bytes)
        expected = EXPECTED[contract["scene_id"]]
        conversion_audit = audit_audiogs_conversion(
            payload,
            expected_scene=expected["audio_scene"],
            expected_clips=expected["audio_clips"],
            epochs=61,
            expected_audio_root=payload["audio_root"],
            expected_cameras_npz=payload["cameras_npz"],
            expected_output_root=payload["output_root"],
        )
        if conversion_audit != completion["conversion_audit"]:
            raise NativeContractError("AudioGS conversion audit mismatch")
        if completion != {
            "conversion_audit": conversion_audit,
            "seed_audit": seed_audit,
            **checkpoint["metadata"],
        }:
            raise NativeContractError("AudioGS completion metadata mismatch")
    else:
        completion = _exact(
            contract["completion"],
            {
                "config_audit",
                "flow_audit",
                "seed_audits",
                "train_log_markers",
                "iterations",
            },
            "native FreeTimeGS++ completion",
        )
        if contract["budget"] != {"iterations": 30_000}:
            raise NativeContractError("native FreeTimeGS++ budget mismatch")
        for name in ("rendered_config", "train_log"):
            snapshot(inputs.get(name), f"FTGS++ {name}")
        if not isinstance(inputs.get("sampled_scene_root"), str):
            raise NativeContractError("FTGS++ sampled scene root is missing")
        _, rendered_bytes = snapshot(
            inputs["rendered_config"], "FTGS++ rendered config"
        )
        try:
            rendered_value = tomllib.loads(rendered_bytes.decode())
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise NativeContractError(
                f"FTGS++ rendered config snapshot is invalid: {error}"
            ) from error
        data_config = rendered_value.get("data", {})
        init_config = rendered_value.get("init", {})
        train_config = rendered_value.get("train", {})
        if (
            data_config.get("frames")
            != {
                "start": 0,
                "stop": raw["benchmark"]["expected_test_samples"],
            }
            or data_config.get("eval_cameras") != [37]
            or data_config.get("train_cameras") != {"start": 0, "stop": 38}
            or init_config.get("temporal_flow_cameras") != {"start": 0, "stop": 38}
            or init_config.get("keyframe_stride") != 10
            or init_config.get("temporal_motion_adapted") is not True
            or train_config.get("iterations") != 30_000
            or train_config.get("batch_size") != 1
        ):
            raise NativeContractError("FTGS++ rendered config protocol mismatch")
        if len(inputs["seed_records"]) != 3:
            raise NativeContractError("FTGS++ seed record count mismatch")
        seed_audits = [
            audit_ftgspp_seed_record(
                json.loads(snapshot(record, "FTGS++ seed record")[1]),
                expected_scene=contract["scene_id"],
            )
            for record in inputs["seed_records"]
        ]
        if seed_audits != completion["seed_audits"]:
            raise NativeContractError("FTGS++ seed audit mismatch")
        inventory_record = _exact(
            inputs["flow_inventory"],
            {"snapshot", "sha256"},
            "FTGS++ flow inventory",
        )
        inventory_name = inventory_record["snapshot"]
        if (
            not isinstance(inventory_name, str)
            or inventory_name not in files
            or _digest_bytes(files[inventory_name]) != inventory_record["sha256"]
        ):
            raise NativeContractError("FTGS++ flow inventory hash mismatch")
        inventory = json.loads(files[inventory_name])
        if (
            not isinstance(inventory, Mapping)
            or set(inventory) != {"root", "files"}
            or not isinstance(inventory["files"], list)
        ):
            raise NativeContractError("FTGS++ flow inventory schema mismatch")
        relative_paths = []
        for entry in inventory["files"]:
            entry = _exact(
                entry, {"path", "sha256", "size"}, "FTGS++ flow inventory entry"
            )
            relative = entry["path"]
            parts = Path(relative).parts if isinstance(relative, str) else ()
            if (
                len(parts) != 2
                or not parts[0].startswith("f")
                or not parts[1].startswith("c")
                or not parts[1].endswith(".npz")
                or not isinstance(entry["size"], int)
                or entry["size"] <= 0
            ):
                raise NativeContractError("FTGS++ flow inventory path mismatch")
            digest = entry["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise NativeContractError("FTGS++ flow inventory sha256 mismatch")
            relative_paths.append(relative)
        flow_audit = _exact(
            completion["flow_audit"],
            {"pairs", "cameras", "files"},
            "FTGS++ flow audit",
        )
        if (
            len(set(relative_paths)) != len(relative_paths)
            or flow_audit["cameras"] != 38
            or flow_audit["files"] != len(relative_paths)
            or flow_audit["pairs"]
            != len({Path(relative).parent.name for relative in relative_paths})
        ):
            raise NativeContractError("FTGS++ flow inventory/audit mismatch")
        _, log_bytes = snapshot(inputs["train_log"], "FTGS++ train log")
        log_text = log_bytes.decode()
        if "Starting training" not in log_text or "Done training" not in log_text:
            raise NativeContractError("FTGS++ train.log completion evidence mismatch")
        if completion["train_log_markers"] != ["Starting training", "Done training"]:
            raise NativeContractError("FTGS++ train.log marker contract mismatch")
        if completion["iterations"] != checkpoint["metadata"]["iterations"]:
            raise NativeContractError("FTGS++ completion iteration mismatch")
    audio_basis = (
        conversion_bytes
        if contract["model_kind"] == "audiogs"
        else canonical_json({"model_kind": "ftgspp", "audio": "not_applicable"})
    )
    expected_derived = {
        "visual_initialization_sha256": _digest_bytes(provenance_bytes),
        "audio_initialization_sha256": _digest_bytes(audio_basis),
        "model_initialization_sha256": _digest_bytes(
            canonical_json(
                {
                    "model_kind": contract["model_kind"],
                    "model_class": checkpoint["model_class"],
                    "state_schema_sha256": checkpoint["state_schema_sha256"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "source_sha256": upstream["source_sha256"],
                }
            )
        ),
    }
    if dict(derived) != expected_derived:
        raise NativeContractError("native derived initialization hash mismatch")
    checkpoint_fd = _open_absolute_regular(checkpoint_path, "native checkpoint")
    try:
        if _hash_fd(checkpoint_fd, "native checkpoint") != checkpoint["sha256"]:
            raise NativeContractError("native checkpoint hash mismatch")
        inspected = _checkpoint_metadata(
            *_torch_pickle_ops_fd(checkpoint_fd),
            model_kind=contract["model_kind"],
            scene_id=contract["scene_id"],
        )
        if {
            "model_class": checkpoint["model_class"],
            "metadata": checkpoint["metadata"],
            "state_schema_sha256": checkpoint["state_schema_sha256"],
        } != inspected:
            raise NativeContractError("native checkpoint metadata mismatch")
        _verify_retained_path(checkpoint_path, checkpoint_fd, "native checkpoint")
    finally:
        os.close(checkpoint_fd)
    return dict(contract)


__all__ = [
    "NativeContractError",
    "finalize_native_contract",
    "inspect_native_checkpoint",
    "verify_native_contract",
    "write_audiogs_seed_record",
]
