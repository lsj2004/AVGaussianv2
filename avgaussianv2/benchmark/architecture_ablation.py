"""Strict, reuse-first preparation for AudioGS rendering-architecture ablations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch import nn

from avgaussianv2.benchmark.artifacts import (
    canonical_json,
    load_generation,
    publish_generation,
)
from avgaussianv2.benchmark.assets import audit_protocol_config
from avgaussianv2.benchmark.native import verify_native_contract
from avgaussianv2.benchmark.production import (
    sha256_file,
    write_resolved_project_config,
)
from avgaussianv2.benchmark.runtime import (
    BenchmarkRuntime,
    build_production_runtime,
)
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    build_worker_manifest,
    hash_shared_indices,
)
from avgaussianv2.config import load_project_config


ABLATION_STRATEGIES = {
    "native_residual",
    "direct_conditioned_unet",
    "gated_native_residual",
}
_GATE_STATE_SUFFIX = "residual_gate_logit"
PREPARATION_SCHEMA = "avgaussianv2.audio-render-architecture-preparation"


def _load_yaml_mapping(path: Path) -> dict:
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load architecture config {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"architecture config {path} must be a mapping")
    return copy.deepcopy(dict(value))


def validate_strategy_only_delta(
    base_config: Path,
    derived_config: Path,
    *,
    expected_strategy: str,
    native_lre_anchor_strength: float | None = None,
) -> dict[str, str]:
    """Require a derived protocol to differ only by declared model fields."""
    if expected_strategy not in ABLATION_STRATEGIES:
        raise ValueError(f"unsupported ablation strategy {expected_strategy!r}")
    base = _load_yaml_mapping(base_config)
    derived = _load_yaml_mapping(derived_config)
    expected = copy.deepcopy(base)
    model = expected.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("base model config must be a mapping")
    model["audio_render_strategy"] = expected_strategy
    if native_lre_anchor_strength is not None:
        strength = float(native_lre_anchor_strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("native_lre_anchor_strength must be in [0,1]")
        model["native_lre_anchor_strength"] = strength
    if derived != expected:
        raise ValueError(
            "derived architecture config may change only the declared "
            "audio_render_strategy and native_lre_anchor_strength"
        )
    return {
        "strategy": expected_strategy,
        "base_config_sha256": hashlib.sha256(Path(base_config).read_bytes()).hexdigest(),
        "derived_config_sha256": hashlib.sha256(
            Path(derived_config).read_bytes()
        ).hexdigest(),
    }


def common_model_initialization_sha256(model: nn.Module) -> str:
    """Hash all initial state except the single architecture-C gate parameter."""
    digest = hashlib.sha256()
    selected = [
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if not name.endswith(_GATE_STATE_SUFFIX)
    ]
    if not selected:
        raise ValueError("architecture model has no common initialization state")
    for name, tensor in sorted(selected):
        value = tensor.detach().cpu().contiguous()
        metadata = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_aligned_worker_manifest(
    *,
    base_manifest: Mapping[str, object],
    compatibility: BenchmarkCompatibility,
    config: BenchmarkConfig,
    mode: BenchmarkMode | str = BenchmarkMode.JOINT_CONDITIONED,
) -> dict[str, object]:
    """Reuse a base worker's exact sequence while binding a derived runtime."""
    resolved_mode = BenchmarkMode(mode)
    expected_fields = {
        "schema",
        "version",
        "mode",
        "training",
        "compatibility",
        "shared_indices",
    }
    if set(base_manifest) != expected_fields:
        raise ValueError("base A worker manifest fields mismatch")
    if (
        base_manifest["schema"] != "avgaussianv2.cam38-benchmark-worker"
        or base_manifest["version"] != 1
        or base_manifest["mode"] != resolved_mode.value
    ):
        raise ValueError(
            f"base worker is not the strict {resolved_mode.value} run"
        )
    if compatibility.mode != resolved_mode.value:
        raise ValueError("derived compatibility mode differs from base worker")
    shared = tuple(base_manifest["shared_indices"])
    if len(shared) != config.main_updates:
        raise ValueError("base A sample sequence length differs from ablation budget")
    if hash_shared_indices(shared) != compatibility.index_sha256:
        raise ValueError("base A sample sequence hash differs from ablation identity")
    manifest = build_worker_manifest(
        config=config,
        compatibility=compatibility,
        shared_indices=shared,
    )
    if manifest["shared_indices"] != list(shared):
        raise RuntimeError("architecture manifest changed the A sample sequence")
    return manifest


def _load_json_mapping(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def repository_identity() -> dict[str, object]:
    """Return the exact clean Git revision used by train and evaluation."""
    root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError(
            "architecture experiment requires a clean tracked/untracked worktree"
        )
    return {
        "root": str(root),
        "commit": revision,
        "clean": True,
    }


def verify_architecture_preparation(
    protocol_dir: Path,
    *,
    require_repository_match: bool = True,
) -> dict[str, object]:
    """Verify immutable preparation and, by default, the exact Git revision."""
    protocol = Path(protocol_dir)
    _, files, manifest, _ = load_generation(
        protocol / "immutable",
        schema=PREPARATION_SCHEMA,
    )
    required = {
        "resolved_project.yaml",
        "resolved_project.origin.json",
        "worker_manifest.json",
        "preparation.json",
    }
    if set(files) != required:
        raise ValueError("architecture preparation generation files mismatch")
    for name, data in files.items():
        if (protocol / name).read_bytes() != data:
            raise RuntimeError(f"live architecture preparation differs: {name}")
    evidence = json.loads(files["preparation.json"])
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != PREPARATION_SCHEMA
        or evidence.get("version") != 1
        or manifest["identity"]
        != {
            "scene_id": evidence.get("scene_id"),
            "strategy": evidence.get("strategy"),
        }
    ):
        raise ValueError("architecture preparation evidence schema mismatch")
    if require_repository_match and evidence.get("repository") != repository_identity():
        raise RuntimeError("architecture preparation Git revision mismatch")
    return evidence


def prepare_architecture_run(
    *,
    base_config: Path,
    derived_config: Path,
    base_protocol_dir: Path,
    output_dir: Path,
    strategy: str,
    native_lre_anchor_strength: float | None = None,
    mode: BenchmarkMode | str = BenchmarkMode.JOINT_CONDITIONED,
    device: torch.device | str,
    trusted_upstream_artifacts: bool,
    native_contract_dirs: Mapping[str, Path],
    runtime_builder=build_production_runtime,
    native_verifier=verify_native_contract,
) -> dict[str, object]:
    """Prepare one renderer worker with an aligned mode, sequence, and assets."""
    resolved_mode = BenchmarkMode(mode)
    delta = validate_strategy_only_delta(
        base_config,
        derived_config,
        expected_strategy=strategy,
        native_lre_anchor_strength=native_lre_anchor_strength,
    )
    audit_protocol_config(base_config)
    audit_protocol_config(derived_config)
    if set(native_contract_dirs) != {"audiogs", "ftgspp"}:
        raise ValueError("architecture preparation requires AudioGS and FTGS++ contracts")

    base_project = load_project_config(base_config)
    derived_project = load_project_config(derived_config)
    if base_project.scene.scene_id != derived_project.scene.scene_id:
        raise ValueError("base and derived architecture scenes differ")
    scene_id = derived_project.scene.scene_id

    base_manifest_path = (
        Path(base_protocol_dir)
        / "worker_manifests"
        / f"{resolved_mode.value}.json"
    )
    base_manifest = _load_json_mapping(
        base_manifest_path, f"base {resolved_mode.value} worker manifest"
    )
    base_compatibility = BenchmarkCompatibility.from_mapping(
        base_manifest["compatibility"]
    )
    if (
        base_compatibility.scene_id != scene_id
        or base_compatibility.mode != resolved_mode.value
    ):
        raise ValueError("base A worker identity differs from architecture scene")
    training = BenchmarkConfig.from_mapping(base_manifest["training"])
    shared_indices = tuple(base_manifest["shared_indices"])
    if hash_shared_indices(shared_indices) != base_compatibility.index_sha256:
        raise ValueError("base A worker sample sequence hash mismatch")

    base_preparation = _load_json_mapping(
        Path(base_protocol_dir) / "preparation.json", "base A preparation"
    )
    base_runtime = base_preparation.get("runtime")
    if not isinstance(base_runtime, Mapping):
        raise ValueError("base A preparation has no runtime identity")
    if base_runtime.get("dataset_sample_ids") is None:
        raise ValueError("base A preparation has no ordered dataset sample IDs")

    native_records = {}
    for kind in ("audiogs", "ftgspp"):
        contract_dir = Path(native_contract_dirs[kind]).absolute()
        contract = native_verifier(
            contract_dir, expected_scene=scene_id, expected_model_kind=kind
        )
        configured = (
            derived_project.paths.audio_checkpoint
            if kind == "audiogs"
            else derived_project.paths.visual_checkpoint
        )
        configured = (
            configured
            if configured.is_absolute()
            else Path(base_config).parent / configured
        )
        if (
            contract["inputs"]["protocol_config"]["sha256"]
            != delta["base_config_sha256"]
            or Path(contract["checkpoint"]["path"]).resolve()
            != configured.resolve()
            or contract["checkpoint"]["sha256"] != sha256_file(configured)
        ):
            raise ValueError(
                f"{kind} native contract is not the exact reusable A initialization"
            )
        native_records[kind] = {
            "path": str(contract_dir),
            "manifest_sha256": contract["_manifest_sha256"],
            "checkpoint_sha256": contract["checkpoint"]["sha256"],
        }

    protocol = Path(output_dir) / "protocol"
    if protocol.exists() and any(protocol.iterdir()):
        raise FileExistsError(f"architecture protocol already exists: {protocol}")
    protocol.mkdir(parents=True, exist_ok=True)
    resolved = protocol / "resolved_project.yaml"
    write_resolved_project_config(
        derived_config,
        resolved,
        relative_path_root=Path(base_config).parent,
    )

    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = runtime_builder(
        config_path=resolved,
        device=torch.device(device),
        trusted_upstream_artifacts=trusted_upstream_artifacts,
    )
    if not isinstance(runtime, BenchmarkRuntime):
        raise TypeError("architecture runtime builder must return BenchmarkRuntime")
    with runtime:
        common_initialization = common_model_initialization_sha256(runtime.model)
        if common_initialization != base_compatibility.model_initialization_sha256:
            raise RuntimeError(
                "B/C common initialization differs from the completed A model"
            )
        gate_names = [
            name
            for name in runtime.model.state_dict()
            if name.endswith(_GATE_STATE_SUFFIX)
        ]
        expected_gate_count = 1 if strategy == "gated_native_residual" else 0
        if len(gate_names) != expected_gate_count:
            raise RuntimeError("architecture gate state does not match strategy")
        if gate_names:
            gate = runtime.model.state_dict()[gate_names[0]]
            if gate.ndim != 0 or float(gate) != 0.0:
                raise RuntimeError("architecture gate must initialize at logit zero")
        if (
            tuple(runtime.dataset_sample_ids)
            != tuple(base_runtime["dataset_sample_ids"])
            or runtime.dataset_identity_sha256
            != base_runtime.get("dataset_identity_sha256")
        ):
            raise RuntimeError("B/C ordered training dataset differs from A")
        runtime_identity = {
            "config_sha256": runtime.config_sha256,
            "source_sha256": runtime.source_sha256,
            "visual_initialization_sha256": runtime.visual_initialization_sha256,
            "audio_initialization_sha256": runtime.audio_initialization_sha256,
            "model_initialization_sha256": runtime.model_initialization_sha256,
            "common_model_initialization_sha256": common_initialization,
            "dataset_identity_sha256": runtime.dataset_identity_sha256,
            "dataset_sample_ids": list(runtime.dataset_sample_ids),
        }

    compatibility = BenchmarkCompatibility(
        scene_id=scene_id,
        mode=resolved_mode.value,
        train_cameras=base_compatibility.train_cameras,
        test_camera=base_compatibility.test_camera,
        seed=training.seed,
        index_sha256=base_compatibility.index_sha256,
        visual_initialization_sha256=runtime_identity[
            "visual_initialization_sha256"
        ],
        audio_initialization_sha256=runtime_identity["audio_initialization_sha256"],
        model_initialization_sha256=runtime_identity["model_initialization_sha256"],
        source_sha256=runtime_identity["source_sha256"],
        config_sha256=runtime_identity["config_sha256"],
    )
    manifest = build_aligned_worker_manifest(
        base_manifest=base_manifest,
        compatibility=compatibility,
        config=training,
        mode=resolved_mode,
    )
    manifest_path = protocol / "worker_manifest.json"
    manifest_bytes = canonical_json(manifest)
    _atomic_write(manifest_path, manifest_bytes)

    evidence = {
        "schema": PREPARATION_SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "strategy": strategy,
        "mode": resolved_mode.value,
        "repository": repository_identity(),
        "base_a": {
            "config_path": str(Path(base_config).absolute()),
            "config_sha256": delta["base_config_sha256"],
            "protocol_dir": str(Path(base_protocol_dir).absolute()),
            "worker_manifest_path": str(base_manifest_path.absolute()),
            "worker_manifest_sha256": sha256_file(base_manifest_path),
            "index_sha256": base_compatibility.index_sha256,
            "model_initialization_sha256": (
                base_compatibility.model_initialization_sha256
            ),
        },
        "derived": {
            "config_path": str(Path(derived_config).absolute()),
            "config_sha256": delta["derived_config_sha256"],
            "resolved_config": str(resolved.absolute()),
            "resolved_config_sha256": sha256_file(resolved),
            "worker_manifest": str(manifest_path.absolute()),
            "worker_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "runtime": runtime_identity,
        "native_contracts": native_records,
        "alignment": {
            "same_common_initialization_as_a": True,
            "same_ordered_dataset_as_a": True,
            "same_shared_indices_as_a": True,
            "shared_indices_sha256": base_compatibility.index_sha256,
            "only_config_delta": (
                ["model.audio_render_strategy"]
                if native_lre_anchor_strength is None
                else [
                    "model.audio_render_strategy",
                    "model.native_lre_anchor_strength",
                ]
            ),
        },
    }
    evidence_bytes = canonical_json(evidence)
    _atomic_write(protocol / "preparation.json", evidence_bytes)
    publish_generation(
        protocol / "immutable",
        schema=PREPARATION_SCHEMA,
        identity={"scene_id": scene_id, "strategy": strategy},
        files={
            "resolved_project.yaml": resolved.read_bytes(),
            "resolved_project.origin.json": resolved.with_name(
                "resolved_project.origin.json"
            ).read_bytes(),
            "worker_manifest.json": manifest_bytes,
            "preparation.json": evidence_bytes,
        },
    )
    return evidence


__all__ = [
    "ABLATION_STRATEGIES",
    "build_aligned_worker_manifest",
    "common_model_initialization_sha256",
    "prepare_architecture_run",
    "repository_identity",
    "validate_strategy_only_delta",
    "verify_architecture_preparation",
]
