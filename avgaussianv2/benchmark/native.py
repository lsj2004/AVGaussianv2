"""Immutable native AudioGS/FreeTimeGS++ training contracts."""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
}


class NativeContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise NativeContractError(
            f"{label} must be an owned single-link regular file"
        )


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


def _torch_pickle_ops(path: Path) -> tuple[dict[str, int], set[str]]:
    """Inspect Torch's pickle opcode stream without executing pickle globals."""
    _safe_regular(path, "native checkpoint")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name for name in archive.namelist() if name.endswith("/data.pkl")
            ]
            if len(names) != 1:
                raise NativeContractError(
                    "native checkpoint must contain exactly one Torch data.pkl"
                )
            info = archive.getinfo(names[0])
            if info.file_size > 256 * 1024 * 1024:
                raise NativeContractError("native checkpoint pickle is unreasonably large")
            data = archive.read(names[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise NativeContractError(
            f"native checkpoint is not a safe Torch zip archive: {error}"
        ) from error
    integers: dict[str, int] = {}
    strings: set[str] = set()
    pending_key: str | None = None
    try:
        for opcode, argument, _ in pickletools.genops(data):
            if opcode.name in {
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
    return integers, strings


def inspect_native_checkpoint(
    path: str | Path,
    *,
    model_kind: str,
    scene_id: str,
) -> dict[str, object]:
    checkpoint = Path(path)
    integers, strings = _torch_pickle_ops(checkpoint)
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
            raise NativeContractError("AudioGS checkpoint model class metadata mismatch")
        model_class = "Audio3DGSMonoDiffGSOnly"
        completion = {
            "epoch_zero_based": 60,
            "max_epochs": 61,
            "resolved_updates": expected_updates,
            "test_viewpoint": 39,
        }
    elif model_kind == "ftgspp":
        if not (
            "ftgspp.models.gaussians" in strings
            and "Gaussians" in strings
            and {
                "means",
                "scales",
                "quats",
                "opacities",
                "sh_0",
                "sh_n",
                "times",
                "durations",
                "marginal_gates",
            }.issubset(strings)
        ):
            raise NativeContractError("FreeTimeGS++ Gaussians checkpoint schema mismatch")
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
    relative = (
        (
            "configs/audio_3dgs_replaynvas_viewpoint.yaml",
            "tools/train_audio_3dgs_viewpoint.py",
            "libs/trainers/Audio3DGSTrainer.py",
            "libs/models/audio_3dgs_mono_diff_gs_only.py",
        )
        if kind == "audiogs"
        else (
            "ftgspp/train/__init__.py",
            "ftgspp/train/train.py",
            "ftgspp/models/gaussians.py",
            "ftgspp/data/flow.py",
        )
    )
    paths = tuple(root / item for item in relative)
    for path in paths:
        _safe_regular(path, "upstream source audit")
    return paths


def _path_record(path: Path) -> dict[str, str]:
    _safe_regular(path, "native contract input")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _record_set_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
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
    record = _exact(
        _json(Path(path), "AudioGS seed record"),
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


def _flow_snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.npz")):
        _safe_regular(path, "FTGS++ flow artifact")
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


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
    ).resolve()
    configured_root = Path(
        raw["paths"][
            "audio_upstream_root" if model_kind == "audiogs" else "visual_upstream_root"
        ]
    ).resolve()
    if checkpoint.resolve() != configured_checkpoint or upstream.resolve() != configured_root:
        raise NativeContractError("native checkpoint/upstream path differs from protocol")
    checkpoint_metadata = inspect_native_checkpoint(
        checkpoint, model_kind=model_kind, scene_id=scene_id
    )
    expected = EXPECTED[scene_id]
    seed_records = tuple(Path(path) for path in seed_records)
    source_audits = [
        _path_record(path) for path in _source_files(model_kind, upstream)
    ]
    inputs: dict[str, object] = {
        "protocol_config": _path_record(config),
        "provenance": _path_record(provenance),
        "seed_records": [_path_record(path) for path in seed_records],
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
        inputs["conversion_manifest"] = _path_record(conversion)
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
            tuple(audit["argv"][audit["argv"].index(option) + 1] for option in ("--from", "--to"))
            if "--from" in audit["argv"]
            else ("flow", "flow")
            for audit in seed_audits
        }
        if stages != {("extract", "prep"), ("points", "train"), ("flow", "flow")}:
            raise NativeContractError("FTGS++ seed records do not cover prep/flow/train")
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
        inputs["rendered_config"] = _path_record(rendered)
        inputs["train_log"] = _path_record(log)
        inputs["sampled_scene_root"] = str(Path(sampled_scene_root).resolve())
        inputs["flow_snapshot_sha256"] = _flow_snapshot(flow_root)
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
            **_path_record(checkpoint),
            **checkpoint_metadata,
        },
        "completion": completion,
    }
    _atomic_json(Path(output_path), contract)
    return contract


def verify_native_contract(
    path: str | Path,
    *,
    expected_scene: str | None = None,
    expected_model_kind: str | None = None,
) -> dict[str, object]:
    contract_path = Path(path)
    contract = _exact(_json(contract_path, "native contract"), _TOP_FIELDS, "native contract")
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
            "flow_snapshot_sha256",
        }
    )
    inputs = _exact(inputs, expected_inputs, "native contract inputs")
    for name in ("protocol_config", "provenance"):
        record = _exact(inputs[name], {"path", "sha256"}, f"native {name}")
        _safe_regular(Path(record["path"]), f"native {name}")
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise NativeContractError(f"native {name} hash mismatch")
    raw = audit_protocol_config(inputs["protocol_config"]["path"])
    if raw["scene"]["id"] != contract["scene_id"]:
        raise NativeContractError("native protocol config scene mismatch")
    audit_initialization_provenance(
        inputs["provenance"]["path"], expected_scene=contract["scene_id"]
    )
    for collection in ("seed_records", "source_audits"):
        if not isinstance(inputs.get(collection), list):
            raise NativeContractError(f"native {collection} must be a list")
        for record in inputs[collection]:
            record = _exact(record, {"path", "sha256"}, f"native {collection}")
            _safe_regular(Path(record["path"]), f"native {collection}")
            if _sha256(Path(record["path"])) != record["sha256"]:
                raise NativeContractError(f"native {collection} hash mismatch")
    checkpoint = _exact(
        contract["checkpoint"],
        {"path", "sha256", "model_class", "metadata", "state_schema_sha256"},
        "native checkpoint",
    )
    checkpoint_path = Path(checkpoint["path"])
    _safe_regular(checkpoint_path, "native checkpoint")
    if _sha256(checkpoint_path) != checkpoint["sha256"]:
        raise NativeContractError("native checkpoint hash mismatch")
    inspected = inspect_native_checkpoint(
        checkpoint_path,
        model_kind=contract["model_kind"],
        scene_id=contract["scene_id"],
    )
    if {
        "model_class": checkpoint["model_class"],
        "metadata": checkpoint["metadata"],
        "state_schema_sha256": checkpoint["state_schema_sha256"],
    } != inspected:
        raise NativeContractError("native checkpoint metadata mismatch")
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
    git_identity = dict(upstream)
    source_sha256 = git_identity.pop("source_sha256")
    if _git_identity(Path(upstream["root"])) != git_identity:
        raise NativeContractError("native upstream Git identity changed")
    expected_sources = [
        _path_record(path)
        for path in _source_files(contract["model_kind"], Path(upstream["root"]))
    ]
    if inputs["source_audits"] != expected_sources:
        raise NativeContractError("native upstream source audit set mismatch")
    if source_sha256 != _record_set_sha256(expected_sources):
        raise NativeContractError("native upstream source aggregate mismatch")
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
            {"path", "sha256"},
            "AudioGS conversion manifest",
        )
        _safe_regular(Path(record["path"]), "AudioGS conversion manifest")
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise NativeContractError("AudioGS conversion manifest hash mismatch")
        if len(inputs["seed_records"]) != 1:
            raise NativeContractError("AudioGS native contract seed record count mismatch")
        seed_audit = _audit_audiogs_seed_record(
            inputs["seed_records"][0]["path"],
            expected_scene=contract["scene_id"],
        )
        if seed_audit != completion["seed_audit"]:
            raise NativeContractError("AudioGS seed audit mismatch")
        payload = _json(Path(record["path"]), "AudioGS conversion manifest")
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
            record = _exact(inputs.get(name), {"path", "sha256"}, f"FTGS++ {name}")
            _safe_regular(Path(record["path"]), f"FTGS++ {name}")
            if _sha256(Path(record["path"])) != record["sha256"]:
                raise NativeContractError(f"FTGS++ {name} hash mismatch")
        if not isinstance(inputs.get("sampled_scene_root"), str):
            raise NativeContractError("FTGS++ sampled scene root is missing")
        config_audit = audit_ftgspp_upstream_config(
            inputs["rendered_config"]["path"],
            protocol_config=inputs["protocol_config"]["path"],
            repo_root=Path(inputs["protocol_config"]["path"]).resolve().parent.parent.parent,
            ftgspp_root=upstream["root"],
            sampled_scene_root=inputs["sampled_scene_root"],
        )
        if config_audit != completion["config_audit"]:
            raise NativeContractError("FTGS++ rendered config audit mismatch")
        if len(inputs["seed_records"]) != 3:
            raise NativeContractError("FTGS++ seed record count mismatch")
        seed_audits = [
            audit_ftgspp_seed_record(
                record["path"], expected_scene=contract["scene_id"]
            )
            for record in inputs["seed_records"]
        ]
        if seed_audits != completion["seed_audits"]:
            raise NativeContractError("FTGS++ seed audit mismatch")
        flow_root = Path(completion["config_audit"]["namespaces"][-1])
        flow_audit = audit_ftgspp_flow_cache(
            flow_root,
            frame_count=raw["benchmark"]["expected_test_samples"],
            keyframe_stride=10,
        )
        if flow_audit != completion["flow_audit"]:
            raise NativeContractError("FTGS++ flow audit mismatch")
        if _flow_snapshot(flow_root) != inputs.get("flow_snapshot_sha256"):
            raise NativeContractError("FTGS++ flow snapshot hash mismatch")
        log_text = Path(inputs["train_log"]["path"]).read_text(encoding="utf-8")
        if "Starting training" not in log_text or "Done training" not in log_text:
            raise NativeContractError("FTGS++ train.log completion evidence mismatch")
        if completion["train_log_markers"] != ["Starting training", "Done training"]:
            raise NativeContractError("FTGS++ train.log marker contract mismatch")
        if completion["iterations"] != checkpoint["metadata"]["iterations"]:
            raise NativeContractError("FTGS++ completion iteration mismatch")
    return dict(contract)


__all__ = [
    "NativeContractError",
    "finalize_native_contract",
    "inspect_native_checkpoint",
    "verify_native_contract",
    "write_audiogs_seed_record",
]
