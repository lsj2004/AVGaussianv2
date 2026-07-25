"""Production construction and identity checks for the cam38 benchmark worker."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from avgaussianv2.benchmark.assets import AssetAuditError, audit_protocol_config
from avgaussianv2.config import (
    ProjectConfig,
    TrainConfig,
    load_project_config,
    load_project_config_bytes,
)
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.losses import AudioLoss
from avgaussianv2.runtime import build_runtime

TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))
TEST_CAMERA = "cam38"


@dataclass(frozen=True)
class BenchmarkRuntime:
    model: nn.Module
    train_samples: Sequence[AlignedAVSample]
    train_config: TrainConfig
    audio_loss_fn: AudioLoss
    config_sha256: str
    source_sha256: str
    visual_initialization_sha256: str
    audio_initialization_sha256: str
    model_initialization_sha256: str
    dataset_identity_sha256: str
    dataset_sample_ids: tuple[str, ...]
    _input_snapshot: ProductionRuntimeSnapshot | None = None
    _import_guard_handle: Any | None = None

    def __enter__(self) -> BenchmarkRuntime:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._import_guard_handle is not None:
            self._import_guard_handle.remove()
            object.__setattr__(self, "_import_guard_handle", None)
        snapshot = self._input_snapshot
        object.__setattr__(self, "_input_snapshot", None)
        if snapshot is not None:
            return snapshot.__exit__(exc_type, exc, traceback)
        return False


class ProductionRuntimeSnapshot:
    """Immutable inputs shared by one or more production runtime builds."""

    def __init__(
        self, config_path: Path, *, config_origin_path: Path | None = None
    ) -> None:
        self.config_path = Path(config_path).absolute()
        self.config_origin_path = (
            None if config_origin_path is None else Path(config_origin_path).absolute()
        )
        self._stack = ExitStack()
        self._pins: list[Any] = []
        self._source_pins: list[Any] = []
        self.config: ProjectConfig | None = None
        self.audited_config: ProjectConfig | None = None
        self.config_proc_path: Path | None = None
        self.origin_proc_path: Path | None = None
        self.source_config_proc_path: Path | None = None
        self.source_config_semantic_path: Path | None = None
        self.config_sha256 = ""
        self.source_config_sha256 = ""
        self.visual_checkpoint_sha256 = ""
        self.audio_checkpoint_sha256 = ""
        self.manifest_sha256 = ""
        self.source_inventory: dict[str, str] = {}
        self.source_finder: Any | None = None
        self._active = False

    def _pin(self, path: Path, expected_sha256: str | None = None):
        # Imported lazily to avoid the production -> runtime module cycle.
        from avgaussianv2.benchmark.production import _PinnedInput

        pin = self._stack.enter_context(_PinnedInput(path, expected_sha256))
        self._pins.append(pin)
        return pin

    def __enter__(self) -> ProductionRuntimeSnapshot:
        if self._active:
            raise RuntimeError("production runtime snapshot cannot be re-entered")
        self._active = True
        try:
            # This first parse is discovery only.  Every value used by execution
            # is loaded again from the pinned config below.
            discovered = load_project_config(self.config_path)
            config_pin = self._pin(self.config_path)
            self.config_proc_path = config_pin.proc_path
            self.config_sha256 = str(config_pin.expected_sha256)

            origin = self.config_origin_path
            if origin is None and self.config_path.name == "resolved_project.yaml":
                origin = self.config_path.with_name("resolved_project.origin.json")
            if origin is not None:
                origin_pin = self._pin(origin)
                self.origin_proc_path = origin_pin.proc_path
                try:
                    origin_value = json.loads(origin_pin.data)
                    source_path = Path(origin_value["source_path"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "resolved benchmark config origin is invalid"
                    ) from error
                if not source_path.is_absolute():
                    raise ValueError(
                        "resolved benchmark source config path must be absolute"
                    )
                source_pin = self._pin(
                    source_path, origin_value.get("source_sha256")
                )
                self.source_config_proc_path = source_pin.proc_path
                self.source_config_semantic_path = source_path
                self.source_config_sha256 = str(source_pin.expected_sha256)

            config, audited_source_sha256 = load_audited_benchmark_config(
                config_pin.proc_path,
                origin_path=self.origin_proc_path,
                config_semantic_path=self.config_path,
                origin_source_path=self.source_config_proc_path,
                origin_source_semantic_path=self.source_config_semantic_path,
                config_snapshot_data=config_pin.data,
                origin_snapshot_data=(
                    None if origin is None else origin_pin.data
                ),
                origin_source_snapshot_data=(
                    None if origin is None else source_pin.data
                ),
            )
            if (
                discovered.scene != config.scene
                or discovered.model != config.model
                or discovered.train != config.train
                or discovered.paths != config.paths
            ):
                raise RuntimeError("production config changed before it was pinned")
            if self.source_config_sha256:
                if audited_source_sha256 != self.source_config_sha256:
                    raise RuntimeError("pinned source config identity mismatch")
            else:
                self.source_config_sha256 = audited_source_sha256

            self.audited_config = config
            self.source_inventory = upstream_source_inventory(config)
            roots = (
                config.paths.audio_upstream_root,
                config.paths.visual_upstream_root,
            )
            roots_by_kind = {"audiogs": roots[0], "ftgspp": roots[1]}
            for name, digest in sorted(self.source_inventory.items()):
                kind, relative = name.split(":", 1)
                self._source_pins.append(
                    self._pin(roots_by_kind[kind] / relative, digest)
                )

            visual_pin = self._pin(config.paths.visual_checkpoint)
            audio_pin = self._pin(config.paths.audio_checkpoint)
            manifest_pin = self._pin(config.paths.manifest)
            self.visual_checkpoint_sha256 = str(visual_pin.expected_sha256)
            self.audio_checkpoint_sha256 = str(audio_pin.expected_sha256)
            self.manifest_sha256 = str(manifest_pin.expected_sha256)
            self.config = replace(
                config,
                paths=replace(
                    config.paths,
                    visual_checkpoint=visual_pin.proc_path,
                    audio_checkpoint=audio_pin.proc_path,
                    manifest=manifest_pin.proc_path,
                ),
            )
            from avgaussianv2.benchmark.production import _snapshot_source_imports

            self.source_finder = self._stack.enter_context(
                _snapshot_source_imports(self._source_pins, roots)
            )
            return self
        except BaseException:
            self._stack.close()
            self._active = False
            raise

    def verify(self) -> None:
        if not self._active or self.config is None:
            raise RuntimeError("production runtime snapshot is not active")
        for pin in self._pins:
            pin.verify()
        self.assert_no_import_failures()
        if (
            self.audited_config is None
            or upstream_source_inventory(self.audited_config) != self.source_inventory
        ):
            raise RuntimeError("upstream source inventory changed during runtime build")

    def assert_no_import_failures(self) -> None:
        if self.source_finder is None:
            raise RuntimeError("production source snapshot finder is unavailable")
        self.source_finder.assert_no_failures()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        verification_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as error:
            verification_error = error
        cleanup_error: BaseException | None = None
        context_error = verification_error if verification_error is not None else exc
        try:
            self._stack.__exit__(
                None if context_error is None else type(context_error),
                context_error,
                None if context_error is None else context_error.__traceback__,
            )
        except BaseException as error:
            cleanup_error = error
        self._active = False
        self.source_finder = None
        if verification_error is not None:
            if exc is not None:
                exc.add_note(
                    f"production snapshot verification also failed: "
                    f"{verification_error}"
                )
            else:
                raise verification_error
        if cleanup_error is not None:
            if exc is not None:
                exc.add_note(
                    f"production snapshot cleanup also failed: {cleanup_error}"
                )
            else:
                raise cleanup_error
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(model: nn.Module, prefix: str | None = None) -> str:
    """Hash named tensor values, not pickle/container representation."""
    digest = hashlib.sha256()
    selected = [
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if prefix is None or name.startswith(prefix)
    ]
    if not selected:
        label = "model" if prefix is None else prefix.removesuffix(".")
        raise ValueError(f"runtime model has no {label} state")
    for name, tensor in sorted(selected):
        value = tensor.detach().cpu().contiguous()
        metadata = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def state_sha256(model: nn.Module, prefix: str | None = None) -> str:
    return _state_sha256(model, prefix)


def upstream_source_inventory(config: ProjectConfig) -> dict[str, str]:
    """Hash every upstream source file imported by the production adapters."""
    roots = {
        "audiogs": config.paths.audio_upstream_root,
        "ftgspp": config.paths.visual_upstream_root,
    }
    relative = {
        "audiogs": (
            Path("configs/audio_3dgs_replaynvas_viewpoint.yaml"),
            Path("tools/train_audio_3dgs_viewpoint.py"),
            *(
                path.relative_to(roots["audiogs"])
                for path in sorted(roots["audiogs"].joinpath("libs").rglob("*.py"))
            ),
        ),
        "ftgspp": tuple(
            path.relative_to(roots["ftgspp"])
            for path in sorted(roots["ftgspp"].joinpath("ftgspp").rglob("*.py"))
        ),
    }
    inventory: dict[str, str] = {}
    for kind, names in relative.items():
        for relative_path in names:
            name = relative_path.as_posix()
            path = (roots[kind] / name).resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"unsafe or missing {kind} upstream source: {path}")
            inventory[f"{kind}:{name}"] = _file_sha256(path)
    return inventory


def _dataset_identity(
    samples: Sequence[AlignedAVSample], config: ProjectConfig
) -> tuple[tuple[str, ...], str]:
    records = getattr(samples, "records", None)
    if records is None:
        raise TypeError("production training dataset must expose ordered records")
    identities: list[str] = []
    order_keys: list[tuple[int, int]] = []
    for record in records:
        camera = str(record.camera)
        if camera not in config.scene.train_cameras:
            raise ValueError(
                f"training dataset contains non-training camera {camera!r}"
            )
        expected_index = config.scene.camera_mapping[camera]
        if int(record.camera_index) != expected_index:
            raise ValueError(f"training dataset camera mapping mismatch for {camera}")
        order_keys.append(
            (int(record.frame_index), config.scene.train_cameras.index(camera))
        )
        identity = {
            "camera": camera,
            "camera_index": int(record.camera_index),
            "frame_index": int(record.frame_index),
            "scene_id": config.scene.scene_id,
            "time_seconds": float(record.time_seconds),
        }
        identities.append(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if not identities:
        raise ValueError("production training dataset is empty")
    if len(set(identities)) != len(identities):
        raise ValueError("production training dataset has duplicate sample IDs")
    if order_keys != sorted(order_keys):
        raise ValueError("production training dataset sample-ID order is not canonical")
    value = tuple(identities)
    return value, _json_sha256(list(value))


def load_audited_benchmark_config(
    config_path: Path,
    *,
    origin_path: Path | None = None,
    config_semantic_path: Path | None = None,
    origin_source_path: Path | None = None,
    origin_source_semantic_path: Path | None = None,
    config_snapshot_data: bytes | None = None,
    origin_snapshot_data: bytes | None = None,
    origin_source_snapshot_data: bytes | None = None,
) -> tuple[ProjectConfig, str]:
    """Load a canonical or orchestrator-resolved config and return source SHA."""
    config_path = Path(config_path)
    semantic_path = (
        config_path if config_semantic_path is None else Path(config_semantic_path)
    )
    config_data = (
        config_path.read_bytes()
        if config_snapshot_data is None
        else config_snapshot_data
    )
    source_sha256 = hashlib.sha256(config_data).hexdigest()
    try:
        audit_protocol_config(
            config_path,
            semantic_path=semantic_path,
            snapshot_data=config_data,
        )
    except AssetAuditError:
        # Orchestration materializes an absolute-path copy outside configs/.
        # Bind it back to the audited immutable source instead of weakening the
        # canonical Task11 path checks.
        if semantic_path.name != "resolved_project.yaml" and origin_path is None:
            raise
        origin_path = (
            config_path.with_name("resolved_project.origin.json")
            if origin_path is None
            else Path(origin_path)
        )
        try:
            origin_data = (
                origin_path.read_bytes()
                if origin_snapshot_data is None
                else origin_snapshot_data
            )
            origin = json.loads(origin_data)
        except (OSError, ValueError) as error:
            raise ValueError("resolved benchmark config origin is missing") from error
        if (
            not isinstance(origin, dict)
            or set(origin)
            != {
                "schema",
                "version",
                "source_path",
                "source_sha256",
                "resolved_sha256",
            }
            or origin["schema"] != "avgaussianv2.cam38-resolved-config-origin"
            or origin["version"] != 1
            or origin["resolved_sha256"] != hashlib.sha256(config_data).hexdigest()
        ):
            raise ValueError("resolved benchmark config origin contract mismatch")
        source = Path(origin["source_path"])
        source_input = (
            source if origin_source_path is None else Path(origin_source_path)
        )
        source_semantic = (
            source
            if origin_source_semantic_path is None
            else Path(origin_source_semantic_path)
        )
        if (
            not source.is_absolute()
            or source_semantic != source
            or hashlib.sha256(
                source_input.read_bytes()
                if origin_source_snapshot_data is None
                else origin_source_snapshot_data
            ).hexdigest()
            != origin["source_sha256"]
        ):
            raise ValueError("resolved benchmark source config hash mismatch")
        source_data = (
            source_input.read_bytes()
            if origin_source_snapshot_data is None
            else origin_source_snapshot_data
        )
        audit_protocol_config(
            source_input,
            semantic_path=source,
            snapshot_data=source_data,
        )
        source_sha256 = origin["source_sha256"]
        source_config = load_project_config_bytes(
            source_data, base_dir=source.parent
        )
        resolved_config = load_project_config_bytes(
            config_data, base_dir=semantic_path.parent
        )
        path_names = (
            "visual_upstream_root",
            "audio_upstream_root",
            "visual_checkpoint",
            "audio_checkpoint",
            "manifest",
            "visual_memmap",
        )
        if (
            source_config.scene != resolved_config.scene
            or source_config.model != resolved_config.model
            or source_config.train != resolved_config.train
            or any(
                (
                    getattr(source_config.paths, name) is None
                    or getattr(resolved_config.paths, name) is None
                )
                and getattr(source_config.paths, name)
                != getattr(resolved_config.paths, name)
                or (
                    getattr(source_config.paths, name) is not None
                    and getattr(resolved_config.paths, name) is not None
                    and getattr(source_config.paths, name).resolve()
                    != getattr(resolved_config.paths, name).resolve()
                )
                for name in path_names
            )
        ):
            raise ValueError("resolved benchmark config changes protocol semantics")
    config = load_project_config_bytes(
        config_data, base_dir=semantic_path.parent
    )
    if config.scene.train_cameras != TRAIN_CAMERAS:
        raise ValueError("benchmark config must train on exactly cam00 through cam37")
    if config.scene.eval_cameras != (TEST_CAMERA,):
        raise ValueError("benchmark config must reserve exactly cam38 for evaluation")
    if config.scene.camera_mapping != {f"cam{index:02d}": index for index in range(39)}:
        raise ValueError("benchmark config camera mapping must be exactly cam00..cam38")
    return config, source_sha256


def build_production_runtime(
    *,
    config_path: Path,
    device: torch.device,
    trusted_upstream_artifacts: bool,
    config_origin_path: Path | None = None,
    input_snapshot: ProductionRuntimeSnapshot | None = None,
) -> BenchmarkRuntime:
    """Build train-only state and bind it to canonical source evidence.

    ``source_sha256`` is the SHA-256 of canonical JSON containing the real
    project-config, dataset-manifest, visual-checkpoint, audio-checkpoint,
    camera-mapping, and ordered training-sample identity digests.
    """
    if input_snapshot is None:
        snapshot = ProductionRuntimeSnapshot(
            config_path, config_origin_path=config_origin_path
        )
        snapshot.__enter__()
        try:
            runtime = build_production_runtime(
                config_path=config_path,
                device=device,
                trusted_upstream_artifacts=trusted_upstream_artifacts,
                config_origin_path=config_origin_path,
                input_snapshot=snapshot,
            )
            object.__setattr__(runtime, "_input_snapshot", snapshot)

            def audit_imports(_module, _inputs, _output):
                snapshot.assert_no_import_failures()

            object.__setattr__(
                runtime,
                "_import_guard_handle",
                runtime.model.register_forward_hook(
                    audit_imports, always_call=True
                ),
            )
            return runtime
        except BaseException:
            snapshot.__exit__(*sys.exc_info())
            raise
    if not input_snapshot._active or input_snapshot.config is None:
        raise RuntimeError("production runtime input snapshot is not active")
    config = input_snapshot.config
    bundle = build_runtime(
        config,
        device,
        trusted_upstream_artifacts=trusted_upstream_artifacts,
        include_eval=False,
    )
    if bundle.eval_samples is not None:
        raise RuntimeError("benchmark production runtime constructed eval samples")
    sample_ids, dataset_identity_sha256 = _dataset_identity(
        bundle.train_samples, config
    )
    model_initialization_sha256 = _state_sha256(bundle.model)
    evidence = {
        "audio_checkpoint_sha256": input_snapshot.audio_checkpoint_sha256,
        "camera_mapping_sha256": _json_sha256(config.scene.camera_mapping),
        "config_sha256": input_snapshot.config_sha256,
        "dataset_identity_sha256": dataset_identity_sha256,
        "dataset_manifest_sha256": input_snapshot.manifest_sha256,
        "visual_checkpoint_sha256": input_snapshot.visual_checkpoint_sha256,
        "model_initialization_sha256": model_initialization_sha256,
        "upstream_source_inventory": input_snapshot.source_inventory,
    }
    input_snapshot.verify()
    return BenchmarkRuntime(
        model=bundle.model,
        train_samples=bundle.train_samples,
        train_config=config.train,
        audio_loss_fn=bundle.audio_loss_fn,
        config_sha256=evidence["config_sha256"],
        source_sha256=_json_sha256(evidence),
        visual_initialization_sha256=_state_sha256(bundle.model, "visual."),
        audio_initialization_sha256=_state_sha256(bundle.model, "audio."),
        model_initialization_sha256=model_initialization_sha256,
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_sample_ids=sample_ids,
    )


__all__ = [
    "BenchmarkRuntime",
    "ProductionRuntimeSnapshot",
    "build_production_runtime",
    "load_audited_benchmark_config",
    "state_sha256",
    "upstream_source_inventory",
]
