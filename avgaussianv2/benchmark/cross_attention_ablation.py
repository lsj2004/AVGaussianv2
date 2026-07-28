"""Strict preparation for AudioGS-native cross-attention conditioning."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from avgaussianv2.benchmark.architecture_ablation import (
    build_aligned_worker_manifest,
    repository_identity,
)
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
    hash_shared_indices,
)
from avgaussianv2.config import load_project_config


PREPARATION_SCHEMA = "avgaussianv2.cross-attention-preparation"
CROSS_ATTENTION_SYSTEM = "cross_attention"
CAUSAL_EVALUATION_SYSTEMS = (
    "cross_attention",
    "cross_attention_no_rgbd",
    "cross_attention_shuffled_rgbd",
    "cross_attention_no_gaussians",
    "cross_attention_no_pose",
)
MASK_CROSS_ATTENTION_SYSTEM = "cross_attention_masks"
MASK_CAUSAL_EVALUATION_SYSTEMS = (
    "cross_attention_masks",
    "cross_attention_masks_no_rgbd",
    "cross_attention_masks_shuffled_rgbd",
)
QUERY_P1_SYSTEM = "query_dependent_p1"
QUERY_P1_CAUSAL_EVALUATION_SYSTEMS = (
    "query_dependent_p1",
    "query_dependent_p1_no_rgbd",
    "query_dependent_p1_wrong_camera",
)
ALL_CROSS_ATTENTION_EVALUATION_SYSTEMS = (
    *CAUSAL_EVALUATION_SYSTEMS,
    *MASK_CAUSAL_EVALUATION_SYSTEMS,
    *QUERY_P1_CAUSAL_EVALUATION_SYSTEMS,
)


@dataclass(frozen=True)
class CrossAttentionVariant:
    backend: str
    system: str
    evaluation_systems: tuple[str, ...]
    audio_query_input: str
    cross_attention_memory: tuple[str, ...]


_VARIANTS = {
    "cross_attention_tokens": CrossAttentionVariant(
        backend="cross_attention_tokens",
        system=CROSS_ATTENTION_SYSTEM,
        evaluation_systems=CAUSAL_EVALUATION_SYSTEMS,
        audio_query_input="source_audio_stft_tokens",
        cross_attention_memory=(
            "rgbd_tokens",
            "pose_tokens",
            "explicit_audiogs_gaussian_attribute_tokens",
        ),
    ),
    "cross_attention_masks": CrossAttentionVariant(
        backend="cross_attention_masks",
        system=MASK_CROSS_ATTENTION_SYSTEM,
        evaluation_systems=MASK_CAUSAL_EVALUATION_SYSTEMS,
        audio_query_input="audiogs_mono_diff_feature_patch_tokens",
        cross_attention_memory=("rgbd_tokens",),
    ),
    "query_dependent_p1": CrossAttentionVariant(
        backend="query_dependent_p1",
        system=QUERY_P1_SYSTEM,
        evaluation_systems=QUERY_P1_CAUSAL_EVALUATION_SYSTEMS,
        audio_query_input="native_audiogs_target_view_time_frequency_features",
        cross_attention_memory=(
            "rgb_tokens",
            "metric_depth_geometry",
            "world_positions",
            "world_normals",
            "condition_camera_pose",
        ),
    ),
}


def cross_attention_variant(audio_backend: str) -> CrossAttentionVariant:
    try:
        return _VARIANTS[audio_backend]
    except KeyError as error:
        raise ValueError(
            f"unsupported strict cross-attention backend {audio_backend!r}"
        ) from error


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load cross-attention config {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"cross-attention config {path} must be a mapping")
    return copy.deepcopy(dict(value))


def _load_json(path: Path, label: str) -> dict:
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


def validate_backend_only_delta(
    base_config: Path,
    derived_config: Path,
    *,
    expected_backend: str | None = None,
) -> dict[str, str]:
    """Require the raw protocol to change only ``model.audio_backend``."""
    base = _load_yaml(Path(base_config))
    derived = _load_yaml(Path(derived_config))
    expected = copy.deepcopy(base)
    model = expected.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("base model config must be a mapping")
    derived_model = derived.get("model")
    if not isinstance(derived_model, Mapping):
        raise ValueError("derived model config must be a mapping")
    backend = str(derived_model.get("audio_backend", ""))
    cross_attention_variant(backend)
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(
            f"derived audio backend must be {expected_backend!r}, got {backend!r}"
        )
    model["audio_backend"] = backend
    if derived != expected:
        raise ValueError(
            "cross-attention config may change only model.audio_backend"
        )
    return {
        "audio_backend": backend,
        "base_config_sha256": hashlib.sha256(
            Path(base_config).read_bytes()
        ).hexdigest(),
        "derived_config_sha256": hashlib.sha256(
            Path(derived_config).read_bytes()
        ).hexdigest(),
    }


def verify_cross_attention_preparation(
    protocol_dir: Path,
    *,
    require_repository_match: bool = True,
) -> dict[str, object]:
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
        raise ValueError("cross-attention preparation generation files mismatch")
    for name, data in files.items():
        if (protocol / name).read_bytes() != data:
            raise RuntimeError(f"live cross-attention preparation differs: {name}")
    evidence = json.loads(files["preparation.json"])
    if not isinstance(evidence, dict):
        raise ValueError("cross-attention preparation evidence must be an object")
    backend = evidence.get("audio_backend")
    if backend is None and evidence.get("system") == CROSS_ATTENTION_SYSTEM:
        # Version-1 preparations created before the mask renderer existed did
        # not persist the otherwise implied backend name.
        backend = "cross_attention_tokens"
    variant = cross_attention_variant(str(backend))
    if (
        evidence.get("schema") != PREPARATION_SCHEMA
        or evidence.get("version") != 1
        or evidence.get("system") != variant.system
        or tuple(evidence.get("causal_evaluation_systems", ()))
        != variant.evaluation_systems
        or manifest["identity"]
        != {
            "scene_id": evidence.get("scene_id"),
            "system": variant.system,
        }
    ):
        raise ValueError("cross-attention preparation evidence schema mismatch")
    if require_repository_match and evidence.get("repository") != repository_identity():
        raise RuntimeError("cross-attention preparation Git revision mismatch")
    return evidence


def prepare_cross_attention_run(
    *,
    base_config: Path,
    derived_config: Path,
    base_protocol_dir: Path,
    output_dir: Path,
    device: torch.device | str,
    trusted_upstream_artifacts: bool,
    ftgspp_contract_dir: Path,
    audiogs_contract_dir: Path,
    runtime_builder=build_production_runtime,
    native_verifier=verify_native_contract,
) -> dict[str, object]:
    """Bind cross-attention to A's data, budget, and both Gaussian checkpoints."""
    delta = validate_backend_only_delta(base_config, derived_config)
    variant = cross_attention_variant(delta["audio_backend"])
    audit_protocol_config(base_config)
    audit_protocol_config(derived_config)
    base_project = load_project_config(base_config)
    derived_project = load_project_config(derived_config)
    if (
        derived_project.model.audio_backend != variant.backend
        or base_project.scene.scene_id != derived_project.scene.scene_id
    ):
        raise ValueError("invalid cross-attention project identity")
    scene_id = derived_project.scene.scene_id

    base_manifest_path = (
        Path(base_protocol_dir) / "worker_manifests" / "joint_conditioned.json"
    )
    base_manifest = _load_json(base_manifest_path, "base A worker manifest")
    base_compatibility = BenchmarkCompatibility.from_mapping(
        base_manifest["compatibility"]
    )
    if (
        base_compatibility.scene_id != scene_id
        or base_compatibility.mode != BenchmarkMode.JOINT_CONDITIONED.value
    ):
        raise ValueError("base A worker identity differs from cross-attention scene")
    training = BenchmarkConfig.from_mapping(base_manifest["training"])
    shared_indices = tuple(base_manifest["shared_indices"])
    if hash_shared_indices(shared_indices) != base_compatibility.index_sha256:
        raise ValueError("base A sample sequence hash mismatch")

    base_preparation = _load_json(
        Path(base_protocol_dir) / "preparation.json",
        "base A preparation",
    )
    base_runtime = base_preparation.get("runtime")
    if (
        not isinstance(base_runtime, Mapping)
        or not isinstance(base_runtime.get("dataset_sample_ids"), list)
    ):
        raise ValueError("base A preparation has no ordered dataset identity")

    contract_dir = Path(ftgspp_contract_dir).absolute()
    contract = native_verifier(
        contract_dir,
        expected_scene=scene_id,
        expected_model_kind="ftgspp",
    )
    if (
        contract["inputs"]["protocol_config"]["sha256"]
        != delta["base_config_sha256"]
        or Path(contract["checkpoint"]["path"]).resolve()
        != derived_project.paths.visual_checkpoint.resolve()
        or contract["checkpoint"]["sha256"]
        != sha256_file(derived_project.paths.visual_checkpoint)
    ):
        raise ValueError("FTGS++ contract is not A's exact visual initialization")
    audio_contract_dir = Path(audiogs_contract_dir).absolute()
    audio_contract = native_verifier(
        audio_contract_dir,
        expected_scene=scene_id,
        expected_model_kind="audiogs",
    )
    if (
        audio_contract["inputs"]["protocol_config"]["sha256"]
        != delta["base_config_sha256"]
        or Path(audio_contract["checkpoint"]["path"]).resolve()
        != derived_project.paths.audio_checkpoint.resolve()
        or audio_contract["checkpoint"]["sha256"]
        != sha256_file(derived_project.paths.audio_checkpoint)
    ):
        raise ValueError(
            "AudioGS contract is not A's exact acoustic-Gaussian initialization"
        )

    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with tempfile.TemporaryDirectory(prefix="avgaussianv2-cross-prepare-") as temporary:
        runtime_resolved = Path(temporary) / "resolved_project.yaml"
        write_resolved_project_config(derived_config, runtime_resolved)
        runtime = runtime_builder(
            config_path=runtime_resolved,
            device=torch.device(device),
            trusted_upstream_artifacts=trusted_upstream_artifacts,
        )
        if not isinstance(runtime, BenchmarkRuntime):
            raise TypeError("cross-attention runtime builder must return BenchmarkRuntime")
        with runtime:
            if (
                runtime.visual_initialization_sha256
                != base_compatibility.visual_initialization_sha256
                or tuple(runtime.dataset_sample_ids)
                != tuple(base_runtime["dataset_sample_ids"])
                or runtime.dataset_identity_sha256
                != base_runtime.get("dataset_identity_sha256")
            ):
                raise RuntimeError(
                    "cross-attention visual initialization or ordered dataset differs from A"
                )
            trainable_parameters = sum(
                parameter.numel() for parameter in runtime.model.parameters()
            )
            audio_parameters = sum(
                parameter.numel() for parameter in runtime.model.audio.parameters()
            )
            runtime_identity = {
                "config_sha256": runtime.config_sha256,
                "source_sha256": runtime.source_sha256,
                "visual_initialization_sha256": runtime.visual_initialization_sha256,
                "audio_initialization_sha256": runtime.audio_initialization_sha256,
                "model_initialization_sha256": runtime.model_initialization_sha256,
                "dataset_identity_sha256": runtime.dataset_identity_sha256,
                "dataset_sample_ids": list(runtime.dataset_sample_ids),
                "model_parameters": trainable_parameters,
                "audio_parameters": audio_parameters,
            }

    protocol = Path(output_dir) / "protocol"
    if protocol.exists() and any(protocol.iterdir()):
        raise FileExistsError(f"cross-attention protocol already exists: {protocol}")
    protocol.mkdir(parents=True, exist_ok=True)
    resolved = protocol / "resolved_project.yaml"
    write_resolved_project_config(derived_config, resolved)

    compatibility = BenchmarkCompatibility(
        scene_id=scene_id,
        mode=BenchmarkMode.JOINT_CONDITIONED.value,
        train_cameras=base_compatibility.train_cameras,
        test_camera=base_compatibility.test_camera,
        seed=training.seed,
        index_sha256=base_compatibility.index_sha256,
        visual_initialization_sha256=runtime_identity[
            "visual_initialization_sha256"
        ],
        audio_initialization_sha256=runtime_identity[
            "audio_initialization_sha256"
        ],
        model_initialization_sha256=runtime_identity[
            "model_initialization_sha256"
        ],
        source_sha256=runtime_identity["source_sha256"],
        config_sha256=runtime_identity["config_sha256"],
    )
    worker_manifest = build_aligned_worker_manifest(
        base_manifest=base_manifest,
        compatibility=compatibility,
        config=training,
    )
    worker_bytes = canonical_json(worker_manifest)
    _atomic_write(protocol / "worker_manifest.json", worker_bytes)

    evidence = {
        "schema": PREPARATION_SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "system": variant.system,
        "audio_backend": variant.backend,
        "repository": repository_identity(),
        "base_a": {
            "config_path": str(Path(base_config).absolute()),
            "config_sha256": delta["base_config_sha256"],
            "protocol_dir": str(Path(base_protocol_dir).absolute()),
            "worker_manifest_sha256": sha256_file(base_manifest_path),
            "index_sha256": base_compatibility.index_sha256,
        },
        "derived": {
            "config_path": str(Path(derived_config).absolute()),
            "config_sha256": delta["derived_config_sha256"],
            "resolved_config_sha256": sha256_file(resolved),
            "worker_manifest_sha256": hashlib.sha256(worker_bytes).hexdigest(),
        },
        "runtime": runtime_identity,
        "ftgspp_contract": {
            "path": str(contract_dir),
            "manifest_sha256": contract["_manifest_sha256"],
            "checkpoint_sha256": contract["checkpoint"]["sha256"],
        },
        "audiogs_contract": {
            "path": str(audio_contract_dir),
            "manifest_sha256": audio_contract["_manifest_sha256"],
            "checkpoint_sha256": audio_contract["checkpoint"]["sha256"],
        },
        "alignment": {
            "comparison_scope": "shared_audiogs_gaussians_postprocessor_ablation",
            "same_visual_initialization_as_a": True,
            "same_audiogs_checkpoint_as_a": True,
            "same_ordered_dataset_as_a": True,
            "same_shared_indices_as_a": True,
            "same_update_budget_as_a": True,
            "shared_indices_sha256": base_compatibility.index_sha256,
            "only_raw_config_delta": "model.audio_backend",
            "audiogs_unet_used_by_cross_attention": False,
            "audio_query_input": variant.audio_query_input,
            "residual_anchor": "native_audiogs_gaussian_render",
            "cross_attention_memory": list(variant.cross_attention_memory),
            "renderer_contract": (
                "audiogs_mono_diff_features_to_mono_diff_masks"
                if variant.backend == "cross_attention_masks"
                else (
                    "query_dependent_geometry_biased_complex_residual"
                    if variant.backend == "query_dependent_p1"
                    else "complex_spectrogram_residual"
                )
            ),
            "audio_criterion": "native_audiogs_checkpoint_criterion",
            "same_frame_camera_contrast": (
                {
                    "weight": derived_project.model.p1_camera_contrast_weight,
                    "margin": derived_project.model.p1_camera_contrast_margin,
                    "warmup_seed_offset": 10_000,
                    "main_seed_offset": 20_000,
                }
                if variant.backend == "query_dependent_p1"
                else None
            ),
        },
        "token_protocol": (
            {
                "audio_position": "deterministic_2d_frequency_time_query_grid",
                "visual_position": "metric_camera_ray_xyz",
                "query_dependent_correspondence": (
                    "listener_head_geometry_bias_per_audio_query_and_visual_token"
                ),
                "acoustic_gaussian_schema": None,
                "pose_tokens": 0,
                "pose_encoding": "listener_geometry_in_audio_queries",
                "memory_modality_embeddings": False,
            }
            if variant.backend == "query_dependent_p1"
            else {
                "audio_position": "deterministic_2d_sinusoidal_frequency_time",
                "visual_position": "deterministic_2d_sinusoidal_row_column",
                "acoustic_gaussian_schema": "audiogs_mono_diff_v1",
                "acoustic_gaussian_position": (
                    "native_frequency_time_grid_then_16x16_structural_pooling"
                ),
                "pose_tokens": 2,
                "memory_modality_embeddings": True,
            }
        ),
        "causal_evaluation_systems": list(variant.evaluation_systems),
    }
    evidence_bytes = canonical_json(evidence)
    _atomic_write(protocol / "preparation.json", evidence_bytes)
    publish_generation(
        protocol / "immutable",
        schema=PREPARATION_SCHEMA,
        identity={"scene_id": scene_id, "system": variant.system},
        files={
            "resolved_project.yaml": resolved.read_bytes(),
            "resolved_project.origin.json": resolved.with_name(
                "resolved_project.origin.json"
            ).read_bytes(),
            "worker_manifest.json": worker_bytes,
            "preparation.json": evidence_bytes,
        },
    )
    return evidence


__all__ = [
    "CAUSAL_EVALUATION_SYSTEMS",
    "ALL_CROSS_ATTENTION_EVALUATION_SYSTEMS",
    "CROSS_ATTENTION_SYSTEM",
    "MASK_CAUSAL_EVALUATION_SYSTEMS",
    "MASK_CROSS_ATTENTION_SYSTEM",
    "QUERY_P1_CAUSAL_EVALUATION_SYSTEMS",
    "QUERY_P1_SYSTEM",
    "CrossAttentionVariant",
    "cross_attention_variant",
    "prepare_cross_attention_run",
    "validate_backend_only_delta",
    "verify_cross_attention_preparation",
]
