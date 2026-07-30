"""Fixed-update training core for the strict cam38 benchmark.

This module is deliberately independent from the bounded cam10 pilot.  It has
no evaluation input: cam38 targets cannot be supplied to the training API.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.losses import AudioLoss, capture_visual_anchor, dssim
from avgaussianv2.train import (
    DisconnectedAudioVisualGradient,
    TrainStepStats,
    build_joint_optimizer,
    build_warmup_optimizer,
    condition_warmup_step,
    joint_train_step,
)

SCHEMA = "avgaussianv2.cam38-fixed-budget"
SCHEMA_VERSION = 1
ALGORITHM = "avgaussianv2.cam38-fixed-budget-v1"
TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))
TEST_CAMERA = "cam38"
_CHECKPOINT_KEYS = {
    "schema",
    "version",
    "fingerprint",
    "compatibility",
    "stage",
    "warmup_step",
    "main_step",
    "model",
    "optimizer",
    "visual_anchor",
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "io",
    "gradient_guard",
}


class BenchmarkMode(str, Enum):
    JOINT_CONDITIONED = "joint_conditioned"
    AUDIO_ONLY = "audio_only"
    VISUAL_ONLY = "visual_only"


class BenchmarkResumeError(RuntimeError):
    """Raised before mutation when an output cannot be resumed exactly."""


@dataclass(frozen=True)
class BenchmarkConfig:
    main_updates: int = 30_000
    conditioner_warmup_steps: int = 2_000
    checkpoint_every: int = 500
    journal_every: int = 10
    milestones: tuple[int, ...] = (5_000, 10_000, 30_000)
    seed: int = 42
    batch_size: int = 1
    selection: str = "final"

    @classmethod
    def from_mapping(cls, value: object) -> BenchmarkConfig:
        expected = {item.name for item in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("benchmark training config fields mismatch")
        try:
            config = cls(
                **{
                    **dict(value),
                    "milestones": tuple(value["milestones"]),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid benchmark training config: {error}") from error
        config.validate()
        return config

    def validate(self, *, strict_protocol: bool = True) -> None:
        for name in (
            "main_updates",
            "conditioner_warmup_steps",
            "checkpoint_every",
            "journal_every",
            "seed",
            "batch_size",
        ):
            if not isinstance(getattr(self, name), int) or isinstance(
                getattr(self, name), bool
            ):
                raise TypeError(f"{name} must be an integer")
        if not isinstance(self.milestones, tuple) or any(
            not isinstance(step, int) or isinstance(step, bool)
            for step in self.milestones
        ):
            raise TypeError("milestones must be a tuple of integers")
        if not isinstance(self.selection, str):
            raise TypeError("selection must be a string")
        if self.main_updates <= 0:
            raise ValueError("main_updates must be positive")
        if self.conditioner_warmup_steps < 0:
            raise ValueError("conditioner_warmup_steps must be nonnegative")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.journal_every <= 0:
            raise ValueError("journal_every must be positive")
        if (
            not self.milestones
            or tuple(sorted(set(self.milestones))) != self.milestones
            or self.milestones[-1] != self.main_updates
            or any(step <= 0 or step > self.main_updates for step in self.milestones)
        ):
            raise ValueError(
                "milestones must be sorted, unique, and end at main_updates"
            )
        if self.batch_size != 1:
            raise ValueError("benchmark batch_size must be 1")
        if self.selection != "final":
            raise ValueError("benchmark selection must be final")
        if strict_protocol:
            if self.main_updates != 30_000:
                raise ValueError(
                    "strict benchmark requires exactly 30,000 main updates"
                )
            if self.conditioner_warmup_steps != 2_000:
                raise ValueError("strict benchmark requires exactly 2,000 warmup steps")
            if self.checkpoint_every != 500:
                raise ValueError("strict benchmark checkpoint interval must be 500")
            if self.milestones != (5_000, 10_000, 30_000):
                raise ValueError("strict benchmark milestones must be 5000/10000/30000")
            if self.seed != 42:
                raise ValueError("strict benchmark seed must be 42")


def _digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class BenchmarkCompatibility:
    scene_id: str
    mode: str
    train_cameras: tuple[str, ...]
    test_camera: str
    seed: int
    index_sha256: str
    visual_initialization_sha256: str
    audio_initialization_sha256: str
    model_initialization_sha256: str
    source_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must not be empty")
        try:
            object.__setattr__(self, "mode", BenchmarkMode(self.mode).value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported benchmark mode: {self.mode!r}") from error
        if self.train_cameras != TRAIN_CAMERAS:
            raise ValueError("train_cameras must be exactly cam00 through cam37")
        if self.test_camera != TEST_CAMERA:
            raise ValueError("test_camera must be exactly cam38")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("benchmark seed must be an integer")
        if self.seed != 42:
            raise ValueError("benchmark seed must be 42")
        for name in (
            "index_sha256",
            "visual_initialization_sha256",
            "audio_initialization_sha256",
            "model_initialization_sha256",
            "source_sha256",
            "config_sha256",
        ):
            _digest(name, getattr(self, name))

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["train_cameras"] = list(self.train_cameras)
        return value

    @classmethod
    def from_mapping(cls, value: object) -> BenchmarkCompatibility:
        if not isinstance(value, Mapping) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise BenchmarkResumeError("compatibility fields mismatch")
        try:
            return cls(
                **{
                    **dict(value),
                    "train_cameras": tuple(value["train_cameras"]),
                }
            )
        except (TypeError, ValueError) as error:
            raise BenchmarkResumeError(f"invalid compatibility: {error}") from error


@dataclass
class CheckpointIO:
    checkpoint_writes: int = 0
    checkpoint_bytes: int = 0
    checkpoint_seconds: float = 0.0
    journal_writes: int = 0
    journal_bytes: int = 0

    def to_mapping(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> CheckpointIO:
        expected = {item.name for item in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise BenchmarkResumeError("checkpoint I/O counters must be a mapping")
        try:
            result = cls(**{item.name: value[item.name] for item in fields(cls)})
        except (KeyError, TypeError, ValueError) as error:
            raise BenchmarkResumeError("invalid checkpoint I/O counters") from error
        for name in (
            "checkpoint_writes",
            "checkpoint_bytes",
            "journal_writes",
            "journal_bytes",
        ):
            item = getattr(result, name)
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise BenchmarkResumeError("invalid checkpoint I/O counters")
        if (
            not isinstance(result.checkpoint_seconds, (int, float))
            or isinstance(result.checkpoint_seconds, bool)
            or not math.isfinite(result.checkpoint_seconds)
            or result.checkpoint_seconds < 0
        ):
            raise BenchmarkResumeError("invalid checkpoint I/O counters")
        result.checkpoint_seconds = float(result.checkpoint_seconds)
        return result


@dataclass(frozen=True)
class BenchmarkTrainingResult:
    mode: BenchmarkMode
    completed_warmup_steps: int
    completed_main_updates: int
    resumed_from_main_step: int
    redone_main_updates: int
    selection: str
    final_checkpoint: Path
    milestones: tuple[Path, ...]
    io: CheckpointIO


def make_shared_indices(
    dataset_length: int, updates: int = 30_000, seed: int = 42
) -> tuple[int, ...]:
    for name, value in (
        ("dataset_length", dataset_length),
        ("updates", updates),
        ("seed", seed),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")
    if updates <= 0:
        raise ValueError("updates must be positive")
    generator = random.Random(seed)
    return tuple(generator.randrange(dataset_length) for _ in range(updates))


def hash_shared_indices(indices: Sequence[int]) -> str:
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("shared sample indices must be integers")
    encoded = json.dumps(list(indices), separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_benchmark_fingerprint(
    *,
    config: BenchmarkConfig,
    compatibility: BenchmarkCompatibility,
    train_config: TrainConfig,
    model: nn.Module,
) -> dict[str, object]:
    inputs = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "config": {
            **asdict(config),
            "milestones": list(config.milestones),
        },
        "compatibility": compatibility.to_mapping(),
        "train_config": asdict(train_config),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_format_version": str(
            getattr(model, "checkpoint_format_version", "state-dict-v1")
        ),
    }
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "inputs": inputs}


def build_worker_manifest(
    *,
    config: BenchmarkConfig,
    compatibility: BenchmarkCompatibility,
    shared_indices: Sequence[int],
) -> dict[str, object]:
    """Build the independent, JSON-safe worker schema persisted by orchestration."""
    config.validate()
    indices = tuple(shared_indices)
    if len(indices) != config.main_updates:
        raise ValueError("shared sample-index sequence must equal main_updates")
    if hash_shared_indices(indices) != compatibility.index_sha256:
        raise ValueError("shared sample-index sequence hash mismatch")
    return {
        "schema": "avgaussianv2.cam38-benchmark-worker",
        "version": 1,
        "mode": compatibility.mode,
        "training": {
            **asdict(config),
            "milestones": list(config.milestones),
        },
        "compatibility": compatibility.to_mapping(),
        "shared_indices": list(indices),
    }


def _set_enabled(parameters: Sequence[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def configure_benchmark_mode(
    model: nn.Module, mode: BenchmarkMode | str, stage: str
) -> None:
    resolved = BenchmarkMode(mode)
    if stage not in {"warmup", "main"}:
        raise ValueError("stage must be warmup or main")
    if not hasattr(model, "condition_enabled"):
        raise TypeError("benchmark model must expose condition_enabled")
    model.unfreeze_all()
    groups = model.named_parameter_groups()
    if stage == "warmup":
        if resolved is not BenchmarkMode.JOINT_CONDITIONED:
            raise ValueError("conditioner warmup is only valid for joint_conditioned")
        model.condition_enabled = True
        for parameters in groups.values():
            _set_enabled(parameters, False)
        for name in ("condition_encoder", "film"):
            _set_enabled(groups[name], True)
        return

    for parameters in groups.values():
        _set_enabled(parameters, False)
    if resolved is BenchmarkMode.JOINT_CONDITIONED:
        for parameters in groups.values():
            _set_enabled(parameters, True)
        model.condition_enabled = True
    elif resolved is BenchmarkMode.AUDIO_ONLY:
        for name in ("acoustic", "audio_unet"):
            _set_enabled(groups[name], True)
        model.condition_enabled = False
    else:
        _set_enabled(groups["visual"], True)
        model.condition_enabled = False


def _require_training_sample(sample: AlignedAVSample) -> AlignedAVSample:
    if sample.camera not in TRAIN_CAMERAS:
        raise ValueError(
            f"training sample camera {sample.camera!r} is outside cam00 through cam37"
        )
    return sample


def _audio_only_step(
    model: nn.Module,
    sample: AlignedAVSample,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    audio_loss_fn: AudioLoss,
) -> TrainStepStats:
    optimizer.zero_grad(set_to_none=True)
    forward_audio_only = getattr(model, "forward_audio_only", None)
    if not callable(forward_audio_only):
        raise TypeError("audio-only benchmark model must expose forward_audio_only()")
    predicted_audio = forward_audio_only(sample)
    raw = audio_loss_fn(predicted_audio, sample.target_audio)
    loss = raw["total_loss"] if isinstance(raw, Mapping) else raw
    if not isinstance(loss, Tensor) or loss.ndim:
        raise ValueError("audio loss must resolve to a scalar tensor")
    total = float(config.lambda_audio) * loss
    if not torch.isfinite(total):
        raise ValueError("audio-only loss is not finite")
    total.backward()
    optimizer.step()
    return TrainStepStats(
        float(total.detach().cpu()),
        {"audio": float(loss.detach().cpu())},
        {},
        0.0,
    )


def _visual_only_step(
    model: nn.Module,
    sample: AlignedAVSample,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    visual_anchor: Mapping[str, Tensor],
) -> TrainStepStats:
    optimizer.zero_grad(set_to_none=True)
    render_rgbd = getattr(model, "render_rgbd", None)
    if not callable(render_rgbd):
        raise TypeError("visual-only benchmark model must expose render_rgbd()")
    rgbd = render_rgbd(sample)
    target = sample.target_rgb.to(rgbd.rgb)
    rgb_l1 = F.l1_loss(rgbd.rgb, target)
    rgb = rgb_l1 + float(config.lambda_dssim) * dssim(rgbd.rgb, target)
    current = dict(model.visual.named_parameters())
    if current.keys() != visual_anchor.keys():
        raise ValueError("visual anchor parameters changed during benchmark")
    anchor_parts = [
        (parameter - visual_anchor[name].to(parameter)).square().mean()
        for name, parameter in current.items()
    ]
    anchor = (
        torch.stack(anchor_parts).mean()
        if anchor_parts
        else torch.zeros((), device=rgb.device, dtype=rgb.dtype)
    )
    total = float(config.lambda_rgb) * rgb + float(config.lambda_visual_anchor) * anchor
    if not torch.isfinite(total):
        raise ValueError("visual-only loss is not finite")
    total.backward()
    optimizer.step()
    return TrainStepStats(
        float(total.detach().cpu()),
        {
            "rgb": float(rgb.detach().cpu()),
            "rgb_l1": float(rgb_l1.detach().cpu()),
            "visual_anchor": float(anchor.detach().cpu()),
        },
        {},
        0.0,
    )


def _atomic_bytes(path: Path, data: bytes) -> int:
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(data)


def _torch_bytes(value: object) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _persist_io_sidecar(
    output: Path,
    fingerprint_sha256: str,
    io_counters: CheckpointIO,
    *,
    committed_checkpoints: Mapping[str, str] | None = None,
) -> None:
    if committed_checkpoints is None:
        sidecar_path = output / "checkpoint_io.json"
        if sidecar_path.is_file():
            previous_io, previous_commits = _load_io_sidecar(
                output, fingerprint_sha256=fingerprint_sha256
            )
            if any(
                getattr(previous_io, field.name) > getattr(io_counters, field.name)
                for field in fields(CheckpointIO)
            ):
                raise BenchmarkResumeError("checkpoint I/O sidecar counter rollback")
            committed_checkpoints = previous_commits
        else:
            committed_checkpoints = {}
    _atomic_bytes(
        output / "checkpoint_io.json",
        _json_bytes(
            {
                "schema": f"{SCHEMA}.checkpoint-io",
                "version": SCHEMA_VERSION,
                "fingerprint_sha256": fingerprint_sha256,
                "io": io_counters.to_mapping(),
                "committed_checkpoints": dict(committed_checkpoints),
            }
        ),
    )


def _load_io_sidecar(
    output: Path,
    *,
    fingerprint_sha256: str,
    checkpoint_io: CheckpointIO | None = None,
) -> tuple[CheckpointIO, dict[str, str]]:
    try:
        payload = json.loads(
            (output / "checkpoint_io.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise BenchmarkResumeError(
            f"cannot read checkpoint I/O sidecar: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema",
            "version",
            "fingerprint_sha256",
            "io",
            "committed_checkpoints",
        }
        or payload["schema"] != f"{SCHEMA}.checkpoint-io"
        or payload["version"] != SCHEMA_VERSION
        or payload["fingerprint_sha256"] != fingerprint_sha256
    ):
        raise BenchmarkResumeError("checkpoint I/O sidecar metadata mismatch")
    sidecar = CheckpointIO.from_mapping(payload["io"])
    if checkpoint_io is not None:
        for field in fields(CheckpointIO):
            if getattr(sidecar, field.name) < getattr(checkpoint_io, field.name):
                raise BenchmarkResumeError("checkpoint I/O sidecar counter rollback")
    committed = payload["committed_checkpoints"]
    if not isinstance(committed, Mapping) or any(
        not isinstance(name, str)
        or "/" in name
        or not name.startswith(("main_step_", "warmup_step_"))
        or not name.endswith(".pt")
        or not name.removesuffix(".pt").rsplit("_", 1)[-1].isdigit()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in committed.items()
    ):
        raise BenchmarkResumeError("checkpoint I/O sidecar commit map mismatch")
    return sidecar, dict(committed)


def _checkpoint_is_committed(
    checkpoint_io: CheckpointIO, sidecar_io: CheckpointIO
) -> bool:
    return all(
        getattr(checkpoint_io, field.name) <= getattr(sidecar_io, field.name)
        for field in fields(CheckpointIO)
    )


def verify_resume_artifacts(
    output: Path,
    *,
    worker_manifest: Mapping[str, object],
) -> None:
    """Strictly verify durable resume state without mutating model or files."""
    benchmark = BenchmarkConfig.from_mapping(worker_manifest.get("training"))
    compatibility_value = worker_manifest.get("compatibility")
    compatibility = BenchmarkCompatibility.from_mapping(compatibility_value)
    expected_warmup = (
        benchmark.conditioner_warmup_steps
        if compatibility.mode == BenchmarkMode.JOINT_CONDITIONED.value
        else 0
    )
    contract_path = output / "contract.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkResumeError(f"cannot read benchmark contract: {error}") from error
    if (
        not isinstance(contract, Mapping)
        or set(contract)
        != {
            "schema",
            "version",
            "fingerprint",
            "compatibility",
            "shared_indices",
            "selection",
            "milestones",
        }
        or contract["schema"] != f"{SCHEMA}.contract"
        or contract["version"] != SCHEMA_VERSION
        or contract["compatibility"] != worker_manifest.get("compatibility")
        or contract["shared_indices"] != worker_manifest.get("shared_indices")
        or contract["selection"] != "final"
        or contract["milestones"] != list(benchmark.milestones)
    ):
        raise BenchmarkResumeError("benchmark contract/fingerprint mismatch")
    fingerprint = contract["fingerprint"]
    if (
        not isinstance(fingerprint, Mapping)
        or set(fingerprint) != {"sha256", "inputs"}
        or not isinstance(fingerprint["inputs"], Mapping)
        or hashlib.sha256(
            json.dumps(
                fingerprint["inputs"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        != fingerprint["sha256"]
    ):
        raise BenchmarkResumeError("benchmark fingerprint mismatch")
    fingerprint_sha256 = str(fingerprint["sha256"])
    progress_path = output / "progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkResumeError(f"cannot read progress journal: {error}") from error
    progress_fields = {
        "schema",
        "version",
        "stage",
        "observed_warmup_step",
        "observed_main_step",
        "exact_warmup_step",
        "exact_main_step",
        "maximum_replay_updates",
        "fingerprint_sha256",
    }
    if (
        not isinstance(progress, Mapping)
        or set(progress) != progress_fields
        or progress["schema"] != f"{SCHEMA}.progress"
        or progress["version"] != SCHEMA_VERSION
        or progress["stage"] not in {"warmup", "main"}
        or progress["fingerprint_sha256"] != fingerprint_sha256
        or progress["maximum_replay_updates"] != benchmark.checkpoint_every
        or any(
            not isinstance(progress[name], int)
            or isinstance(progress[name], bool)
            or progress[name] < 0
            for name in (
                "observed_warmup_step",
                "observed_main_step",
                "exact_warmup_step",
                "exact_main_step",
            )
        )
        or progress["exact_warmup_step"] > progress["observed_warmup_step"]
        or progress["exact_main_step"] > progress["observed_main_step"]
        or progress["observed_warmup_step"] > expected_warmup
        or progress["observed_main_step"] > benchmark.main_updates
        or (
            progress["stage"] == "warmup"
            and (
                compatibility.mode != BenchmarkMode.JOINT_CONDITIONED.value
                or progress["observed_main_step"] != 0
                or progress["exact_main_step"] != 0
            )
        )
        or (
            progress["stage"] == "main"
            and progress["observed_warmup_step"] != expected_warmup
        )
    ):
        raise BenchmarkResumeError("progress journal metadata mismatch")
    sidecar, committed = _load_io_sidecar(
        output, fingerprint_sha256=fingerprint_sha256
    )
    del sidecar
    checkpoints = output / "checkpoints"
    actual_checkpoints = (
        {path.name: path for path in checkpoints.iterdir()}
        if checkpoints.exists()
        else {}
    )
    if set(actual_checkpoints) != set(committed):
        raise BenchmarkResumeError("committed rolling checkpoint inventory mismatch")
    required_exact = (
        None
        if progress["exact_warmup_step"] == progress["exact_main_step"] == 0
        else (
            f"main_step_{progress['exact_main_step']:06d}.pt"
            if progress["exact_main_step"] > 0
            else f"warmup_step_{progress['exact_warmup_step']:06d}.pt"
        )
    )
    if required_exact is not None and required_exact not in committed:
        raise BenchmarkResumeError("progress has no committed exact checkpoint")
    for name, path in actual_checkpoints.items():
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise BenchmarkResumeError(f"unsafe committed checkpoint: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != committed[name]:
            raise BenchmarkResumeError(f"committed checkpoint hash mismatch: {name}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise BenchmarkResumeError(
                f"cannot load exact checkpoint: {error}"
            ) from error
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _CHECKPOINT_KEYS
            or payload.get("schema") != SCHEMA
            or payload.get("version") != SCHEMA_VERSION
            or payload.get("fingerprint") != fingerprint
            or payload.get("compatibility") != compatibility_value
            or payload.get("stage") not in {"warmup", "main"}
            or not isinstance(payload.get("warmup_step"), int)
            or isinstance(payload.get("warmup_step"), bool)
            or not isinstance(payload.get("main_step"), int)
            or isinstance(payload.get("main_step"), bool)
        ):
            raise BenchmarkResumeError("checkpoint schema/fingerprint mismatch")
        if (
            payload["warmup_step"] < 0
            or payload["main_step"] < 0
            or (
                payload["stage"] == "warmup"
                and (
                    compatibility.mode != BenchmarkMode.JOINT_CONDITIONED.value
                    or payload["warmup_step"] > expected_warmup
                    or payload["main_step"] != 0
                )
            )
            or (
                payload["stage"] == "main"
                and (
                    payload["warmup_step"] != expected_warmup
                    or payload["main_step"] > benchmark.main_updates
                )
            )
        ):
            raise BenchmarkResumeError("checkpoint stage/step mismatch")
        step = (
            payload["warmup_step"]
            if payload["stage"] == "warmup"
            else payload["main_step"]
        )
        if name != f"{payload['stage']}_step_{step:06d}.pt":
            raise BenchmarkResumeError("checkpoint filename/step mismatch")
    journal_path = output / "artifact_journal.json"
    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BenchmarkResumeError(
                f"cannot read artifact transaction journal: {error}"
            ) from error
        if (
            not isinstance(journal, Mapping)
            or set(journal)
            != {"schema", "version", "fingerprint_sha256", "sha256"}
            or journal["schema"] != f"{SCHEMA}.artifact-journal"
            or journal["version"] != SCHEMA_VERSION
            or journal["fingerprint_sha256"] != fingerprint_sha256
            or not isinstance(journal["sha256"], Mapping)
        ):
            raise BenchmarkResumeError("artifact transaction journal mismatch")
        for relative, digest in journal["sha256"].items():
            artifact = output / str(relative)
            try:
                metadata = artifact.lstat()
            except OSError:
                metadata = None
            if (
                not isinstance(relative, str)
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or metadata is None
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
            ):
                raise BenchmarkResumeError("artifact transaction hash mismatch")


def _progress_allows_fresh_resume(
    output: Path,
    *,
    fingerprint_sha256: str,
    initial_stage: str,
    maximum_replay_updates: int,
) -> bool:
    try:
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkResumeError(f"cannot read progress journal: {error}") from error
    expected = {
        "schema",
        "version",
        "stage",
        "observed_warmup_step",
        "observed_main_step",
        "exact_warmup_step",
        "exact_main_step",
        "maximum_replay_updates",
        "fingerprint_sha256",
    }
    if (
        not isinstance(progress, Mapping)
        or set(progress) != expected
        or progress["schema"] != f"{SCHEMA}.progress"
        or progress["version"] != SCHEMA_VERSION
        or progress["stage"] != initial_stage
        or progress["fingerprint_sha256"] != fingerprint_sha256
        or progress["maximum_replay_updates"] != maximum_replay_updates
        or any(
            not isinstance(progress[name], int)
            or isinstance(progress[name], bool)
            or progress[name] != 0
            for name in (
                "observed_warmup_step",
                "observed_main_step",
                "exact_warmup_step",
                "exact_main_step",
            )
        )
    ):
        raise BenchmarkResumeError("progress journal does not permit fresh resume")
    return True


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BenchmarkResumeError(f"invalid checkpoint {name}")
    return value


def _validate_tensor_mapping(
    name: str, value: object, expected: Mapping[str, Tensor]
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise BenchmarkResumeError(f"checkpoint {name} keys mismatch")
    for key, expected_tensor in expected.items():
        actual = value[key]
        if (
            not isinstance(actual, Tensor)
            or actual.shape != expected_tensor.shape
            or actual.dtype != expected_tensor.dtype
        ):
            raise BenchmarkResumeError(f"checkpoint {name} tensor mismatch: {key}")


def _validate_optimizer_state(
    value: object, expected_optimizer: torch.optim.Optimizer
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise BenchmarkResumeError("checkpoint optimizer fields mismatch")
    state = value["state"]
    groups = value["param_groups"]
    expected_groups = expected_optimizer.state_dict()["param_groups"]
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        raise BenchmarkResumeError("invalid checkpoint optimizer")
    if len(groups) != len(expected_groups):
        raise BenchmarkResumeError("checkpoint optimizer group count mismatch")
    parameter_ids: set[int] = set()
    parameter_shapes: dict[int, torch.Size] = {}
    for group_index, (actual, expected) in enumerate(zip(groups, expected_groups)):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise BenchmarkResumeError("checkpoint optimizer group fields mismatch")
        if any(
            type(actual[key]) is not type(expected[key]) or actual[key] != expected[key]
            for key in expected
            if key != "params"
        ):
            raise BenchmarkResumeError("checkpoint optimizer group values mismatch")
        parameters = actual.get("params")
        expected_parameters = expected["params"]
        if (
            not isinstance(parameters, list)
            or len(parameters) != len(expected_parameters)
            or any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in parameters
            )
        ):
            raise BenchmarkResumeError("checkpoint optimizer parameters mismatch")
        parameter_ids.update(parameters)
        live_parameters = expected_optimizer.param_groups[group_index]["params"]
        if len(live_parameters) != len(parameters):
            raise BenchmarkResumeError("checkpoint optimizer parameters mismatch")
        parameter_shapes.update(
            {
                identifier: parameter.shape
                for identifier, parameter in zip(parameters, live_parameters)
            }
        )
    if any(
        not isinstance(key, int) or isinstance(key, bool) or key not in parameter_ids
        for key in state
    ):
        raise BenchmarkResumeError("checkpoint optimizer state mismatch")
    for identifier, item in state.items():
        if not isinstance(item, Mapping):
            raise BenchmarkResumeError("checkpoint optimizer state mismatch")
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required.issubset(item) or not set(item).issubset(allowed):
            raise BenchmarkResumeError("checkpoint optimizer state fields mismatch")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = item[name]
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != parameter_shapes[identifier]
            ):
                raise BenchmarkResumeError("checkpoint optimizer tensor mismatch")
        step = item["step"]
        if not isinstance(step, Tensor) or step.numel() != 1:
            raise BenchmarkResumeError("checkpoint optimizer step mismatch")
        maximum = item.get("max_exp_avg_sq")
        if maximum is not None and (
            not isinstance(maximum, Tensor)
            or maximum.shape != parameter_shapes[identifier]
        ):
            raise BenchmarkResumeError("checkpoint optimizer tensor mismatch")


def _validate_rng_state(payload: Mapping[str, object]) -> None:
    torch_state = payload["torch_rng_state"]
    if (
        not isinstance(torch_state, Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
    ):
        raise BenchmarkResumeError("invalid torch RNG state")
    try:
        torch.Generator(device="cpu").set_state(torch_state)
    except Exception as error:
        raise BenchmarkResumeError("invalid torch RNG state") from error
    try:
        probe = random.Random()
        probe.setstate(payload["python_rng_state"])
    except Exception as error:
        raise BenchmarkResumeError("invalid Python RNG state") from error
    numpy_state = payload["numpy_rng_state"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise BenchmarkResumeError("invalid NumPy RNG state")
    numpy_values = numpy_state["state"]
    if (
        numpy_state["bit_generator"] != "MT19937"
        or not isinstance(numpy_values, Tensor)
        or numpy_values.dtype != torch.int64
        or numpy_values.shape != (624,)
    ):
        raise BenchmarkResumeError("invalid NumPy RNG state")
    position = _require_int("NumPy RNG position", numpy_state["position"])
    has_gauss = _require_int("NumPy RNG has_gauss", numpy_state["has_gauss"])
    cached = numpy_state["cached_gaussian"]
    if position > 624 or has_gauss not in {0, 1} or not isinstance(cached, float):
        raise BenchmarkResumeError("invalid NumPy RNG state")
    try:
        np.random.RandomState().set_state(
            (
                "MT19937",
                numpy_values.cpu().numpy().astype(np.uint32),
                position,
                has_gauss,
                cached,
            )
        )
    except Exception as error:
        raise BenchmarkResumeError("invalid NumPy RNG state") from error
    cuda_states = payload["cuda_rng_states"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(item, Tensor) or item.dtype != torch.uint8 or item.ndim != 1
        for item in cuda_states
    ):
        raise BenchmarkResumeError("invalid CUDA RNG state")
    if len(cuda_states) != torch.cuda.device_count():
        raise BenchmarkResumeError("CUDA RNG device-count mismatch")
    try:
        for index, state in enumerate(cuda_states):
            torch.Generator(device=f"cuda:{index}").set_state(state)
    except Exception as error:
        raise BenchmarkResumeError("invalid CUDA RNG state") from error


class FixedBudgetTrainer:
    """Train one benchmark mode without constructing or accepting eval data."""

    def __init__(self, config: BenchmarkConfig) -> None:
        config.validate(strict_protocol=False)
        self.config = config

    def _checkpoint_payload(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        stage: str,
        warmup_step: int,
        main_step: int,
        fingerprint: Mapping[str, object],
        compatibility: BenchmarkCompatibility,
        io_counters: CheckpointIO,
        visual_anchor: Mapping[str, Tensor],
        consecutive_zero_audio_visual_probes: int,
        audio_visual_probe_count: int,
    ) -> dict[str, object]:
        numpy_state = np.random.get_state()
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "fingerprint": dict(fingerprint),
            "compatibility": compatibility.to_mapping(),
            "stage": stage,
            "warmup_step": warmup_step,
            "main_step": main_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "visual_anchor": dict(visual_anchor),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": numpy_state[0],
                # torch 2.5 cannot serialize uint32 typed storage.  MT19937
                # values fit losslessly in signed int64.
                "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "io": io_counters.to_mapping(),
            "gradient_guard": {
                "consecutive_zero_audio_visual_probes": (
                    consecutive_zero_audio_visual_probes
                ),
                "audio_visual_probe_count": audio_visual_probe_count,
            },
        }

    @staticmethod
    def _load_checkpoint(
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_visual_anchor: Mapping[str, Tensor],
        expected_fingerprint: Mapping[str, object],
        expected_compatibility: BenchmarkCompatibility,
        config: BenchmarkConfig,
        train_config: TrainConfig,
        mode: BenchmarkMode,
        expected_stage: str | None = None,
        expected_main_step: int | None = None,
        mutate_model: bool = True,
    ) -> dict[str, object]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise BenchmarkResumeError(
                f"cannot load exact checkpoint: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise BenchmarkResumeError("checkpoint root must be a mapping")
        if set(payload) != _CHECKPOINT_KEYS:
            raise BenchmarkResumeError("checkpoint fields mismatch")
        if payload.get("schema") != SCHEMA or payload.get("version") != SCHEMA_VERSION:
            raise BenchmarkResumeError("checkpoint schema mismatch")
        if payload.get("fingerprint") != expected_fingerprint:
            raise BenchmarkResumeError("checkpoint fingerprint mismatch")
        actual = BenchmarkCompatibility.from_mapping(payload.get("compatibility"))
        if actual != expected_compatibility:
            raise BenchmarkResumeError("checkpoint compatibility mismatch")
        stage = payload["stage"]
        if stage not in {"warmup", "main"} or (
            expected_stage is not None and stage != expected_stage
        ):
            raise BenchmarkResumeError("checkpoint stage mismatch")
        warmup_step = _require_int("warmup_step", payload["warmup_step"])
        main_step = _require_int("main_step", payload["main_step"])
        expected_warmup = (
            config.conditioner_warmup_steps
            if mode is BenchmarkMode.JOINT_CONDITIONED
            else 0
        )
        if stage == "warmup":
            if (
                mode is not BenchmarkMode.JOINT_CONDITIONED
                or warmup_step > config.conditioner_warmup_steps
                or main_step != 0
            ):
                raise BenchmarkResumeError("checkpoint stage/step mismatch")
        elif (
            warmup_step != expected_warmup
            or main_step > config.main_updates
            or (expected_main_step is not None and main_step != expected_main_step)
        ):
            raise BenchmarkResumeError("checkpoint stage/step mismatch")
        if expected_main_step is not None and main_step != expected_main_step:
            raise BenchmarkResumeError("checkpoint main step mismatch")
        filename_step = warmup_step if stage == "warmup" else main_step
        expected_name = f"{stage}_step_{filename_step:06d}.pt"
        if path.parent.name == "checkpoints" and path.name != expected_name:
            raise BenchmarkResumeError("checkpoint filename/step mismatch")
        _validate_tensor_mapping("model", payload["model"], model.state_dict())
        _validate_tensor_mapping(
            "visual anchor", payload["visual_anchor"], expected_visual_anchor
        )
        _validate_optimizer_state(payload["optimizer"], optimizer)
        _validate_rng_state(payload)
        CheckpointIO.from_mapping(payload["io"])
        guard = payload["gradient_guard"]
        if not isinstance(guard, Mapping) or set(guard) != {
            "consecutive_zero_audio_visual_probes",
            "audio_visual_probe_count",
        }:
            raise BenchmarkResumeError("checkpoint gradient guard fields mismatch")
        consecutive = _require_int(
            "consecutive zero audio-visual probes",
            guard["consecutive_zero_audio_visual_probes"],
        )
        probes = _require_int(
            "audio-visual probe count", guard["audio_visual_probe_count"]
        )
        expected_probes = (
            main_step // train_config.gradient_probe_interval
            if mode is BenchmarkMode.JOINT_CONDITIONED and stage == "main"
            else 0
        )
        if (
            mode is not BenchmarkMode.JOINT_CONDITIONED
            and (consecutive != 0 or probes != 0)
        ) or (
            consecutive > probes
            or probes != expected_probes
            or consecutive >= train_config.max_zero_audio_visual_grad_steps
        ):
            raise BenchmarkResumeError("checkpoint gradient guard state mismatch")
        try:
            if mutate_model:
                model.load_state_dict(payload["model"], strict=True)
        except Exception as error:
            raise BenchmarkResumeError(f"checkpoint model mismatch: {error}") from error
        return dict(payload)

    def _write_journal(
        self,
        output: Path,
        *,
        stage: str,
        observed_warmup_step: int,
        observed_main_step: int,
        exact_warmup_step: int,
        exact_main_step: int,
        fingerprint_sha256: str,
        io_counters: CheckpointIO,
    ) -> None:
        payload = {
            "schema": f"{SCHEMA}.progress",
            "version": SCHEMA_VERSION,
            "stage": stage,
            "observed_warmup_step": observed_warmup_step,
            "observed_main_step": observed_main_step,
            "exact_warmup_step": exact_warmup_step,
            "exact_main_step": exact_main_step,
            "maximum_replay_updates": self.config.checkpoint_every,
            "fingerprint_sha256": fingerprint_sha256,
        }
        data = _json_bytes(payload)
        io_counters.journal_bytes += _atomic_bytes(output / "progress.json", data)
        io_counters.journal_writes += 1
        _persist_io_sidecar(output, fingerprint_sha256, io_counters)

    def _write_artifact_journal(
        self,
        output: Path,
        *,
        fingerprint_sha256: str,
        io_counters: CheckpointIO,
    ) -> None:
        artifacts: dict[str, str] = {}
        for step in self.config.milestones:
            path = output / "milestones" / f"step_{step:06d}.pt"
            if path.is_file():
                artifacts[f"milestones/{path.name}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        final = output / "final.pt"
        if final.is_file():
            artifacts["final.pt"] = hashlib.sha256(final.read_bytes()).hexdigest()
        started = time.monotonic()
        _atomic_bytes(
            output / "artifact_journal.json",
            _json_bytes(
                {
                    "schema": f"{SCHEMA}.artifact-journal",
                    "version": SCHEMA_VERSION,
                    "fingerprint_sha256": fingerprint_sha256,
                    "sha256": artifacts,
                }
            ),
        )
        io_counters.checkpoint_seconds += time.monotonic() - started
        _persist_io_sidecar(output, fingerprint_sha256, io_counters)

    def _write_checkpoint(
        self,
        output: Path,
        payload: Mapping[str, object],
        *,
        stage: str,
        progress_step: int,
        main_step: int,
        milestone: bool = False,
        final: bool = False,
        io_counters: CheckpointIO,
    ) -> None:
        started = time.monotonic()
        destinations = [output / "checkpoints" / f"{stage}_step_{progress_step:06d}.pt"]
        if milestone:
            destinations.append(output / "milestones" / f"step_{main_step:06d}.pt")
        if final:
            destinations.append(output / "final.pt")
        persisted = dict(payload)
        io_counters.checkpoint_writes += len(destinations)
        # Account for this checkpoint in the snapshot it persists.  Since the
        # decimal byte count can itself change serialization size, converge the
        # tiny fixed point before publishing identical bytes to all destinations.
        previous_bytes = io_counters.checkpoint_bytes
        for _ in range(8):
            persisted["io"] = io_counters.to_mapping()
            data = _torch_bytes(persisted)
            total_bytes = previous_bytes + len(data) * len(destinations)
            if total_bytes == io_counters.checkpoint_bytes:
                break
            io_counters.checkpoint_bytes = total_bytes
        persisted["io"] = io_counters.to_mapping()
        data = _torch_bytes(persisted)
        final_bytes = previous_bytes + len(data) * len(destinations)
        if final_bytes != io_counters.checkpoint_bytes:
            io_counters.checkpoint_bytes = final_bytes
            persisted["io"] = io_counters.to_mapping()
            data = _torch_bytes(persisted)
            if previous_bytes + len(data) * len(destinations) != final_bytes:
                raise RuntimeError("checkpoint I/O byte counter did not converge")
        for destination in destinations:
            if _atomic_bytes(destination, data) != len(data):
                raise RuntimeError("short checkpoint write")
        io_counters.checkpoint_seconds += time.monotonic() - started
        fingerprint = payload.get("fingerprint")
        if not isinstance(fingerprint, Mapping) or not isinstance(
            fingerprint.get("sha256"), str
        ):
            raise RuntimeError("checkpoint fingerprint is invalid")
        periodic = sorted(
            (output / "checkpoints").glob("*.pt"),
            key=lambda path: (
                path.name.startswith("main_"),
                int(path.stem.rsplit("_", 1)[1]),
            ),
        )
        _persist_io_sidecar(
            output,
            fingerprint["sha256"],
            io_counters,
            committed_checkpoints={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in periodic
            },
        )
        # Milestones have their own immutable copies.  The rolling directory
        # therefore contains only the newest exact point and its predecessor.
        for stale in periodic[:-2]:
            stale.unlink()

    def run(
        self,
        *,
        model: nn.Module,
        train_samples: Sequence[AlignedAVSample],
        shared_indices: Sequence[int],
        mode: BenchmarkMode | str,
        train_config: TrainConfig,
        audio_loss_fn: AudioLoss,
        output_dir: str | Path,
        compatibility: BenchmarkCompatibility,
        resume: bool = False,
        interrupt_after_main_step: int | None = None,
    ) -> BenchmarkTrainingResult:
        resolved_mode = BenchmarkMode(mode)
        if compatibility.mode != resolved_mode.value:
            raise ValueError("compatibility mode does not match requested mode")
        if len(shared_indices) != self.config.main_updates:
            raise ValueError(
                "shared sample-index sequence must equal the main-update budget"
            )
        if hash_shared_indices(shared_indices) != compatibility.index_sha256:
            raise ValueError("shared sample-index sequence hash mismatch")
        if (
            train_config.seed != self.config.seed
            or train_config.joint_steps != self.config.main_updates
            or train_config.warmup_steps != self.config.conditioner_warmup_steps
        ):
            raise ValueError("train config budget/seed does not match benchmark config")
        if not train_samples:
            raise ValueError("training dataset must not be empty")
        if any(index < 0 or index >= len(train_samples) for index in shared_indices):
            raise ValueError("shared sample-index sequence is out of range")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fingerprint = build_benchmark_fingerprint(
            config=self.config,
            compatibility=compatibility,
            train_config=train_config,
            model=model,
        )
        contract_path = output / "contract.json"
        contract = {
            "schema": f"{SCHEMA}.contract",
            "version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "compatibility": compatibility.to_mapping(),
            "shared_indices": list(shared_indices),
            "selection": "final",
            "milestones": list(self.config.milestones),
        }
        exact_checkpoint = sorted((output / "checkpoints").glob("*.pt"))
        if not resume and any(
            path.name != ".benchmark.lock" for path in output.iterdir()
        ):
            raise BenchmarkResumeError("benchmark output already exists; use --resume")
        if not resume:
            _atomic_bytes(contract_path, _json_bytes(contract))
        else:
            try:
                actual_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise BenchmarkResumeError(
                    f"cannot read benchmark contract: {error}"
                ) from error
            if actual_contract != contract:
                raise BenchmarkResumeError("benchmark contract/fingerprint mismatch")
        io_counters = CheckpointIO()
        visual_anchor = capture_visual_anchor(model.visual)
        warmup_step = 0
        main_step = 0
        resumed_from_main_step = 0
        consecutive_zero_audio_visual_probes = 0
        audio_visual_probe_count = 0
        optimizer: torch.optim.Optimizer | None = None
        stage = (
            "warmup"
            if resolved_mode is BenchmarkMode.JOINT_CONDITIONED
            and self.config.conditioner_warmup_steps
            else "main"
        )
        initial_stage = stage
        checkpoint_path: Path | None = None
        publish_initial_checkpoint = not resume
        if resume:
            sidecar_path = output / "checkpoint_io.json"
            if sidecar_path.is_file():
                sidecar_io, committed_checkpoints = _load_io_sidecar(
                    output, fingerprint_sha256=str(fingerprint["sha256"])
                )
            else:
                sidecar_io = CheckpointIO()
                committed_checkpoints = {}
            candidates = sorted(
                exact_checkpoint,
                key=lambda path: (
                    path.name.startswith("main_"),
                    int(path.stem.rsplit("_", 1)[1]),
                ),
                reverse=True,
            )
            original_requires_grad = {
                id(parameter): parameter.requires_grad
                for parameter in model.parameters()
            }
            original_condition_enabled = model.condition_enabled
            original_model_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            original_torch_rng_state = torch.get_rng_state().clone()
            original_python_rng_state = random.getstate()
            original_numpy_rng_state = np.random.get_state()
            original_numpy_rng_state = (
                original_numpy_rng_state[0],
                original_numpy_rng_state[1].copy(),
                original_numpy_rng_state[2],
                original_numpy_rng_state[3],
                original_numpy_rng_state[4],
            )
            original_cuda_rng_states = (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            )
            original_optimizer_state: dict[str, object] | None = None
            try:
                payload: dict[str, object] | None = None
                for candidate in candidates:
                    expected_digest = committed_checkpoints.get(candidate.name)
                    if expected_digest is None:
                        continue
                    candidate_stage = (
                        "main" if candidate.name.startswith("main_") else "warmup"
                    )
                    configure_benchmark_mode(model, resolved_mode, candidate_stage)
                    candidate_optimizer = (
                        build_warmup_optimizer(model, train_config.condition_lr)
                        if candidate_stage == "warmup"
                        else build_joint_optimizer(model, train_config)
                    )
                    candidate_payload = self._load_checkpoint(
                        candidate,
                        model=model,
                        optimizer=candidate_optimizer,
                        expected_visual_anchor=visual_anchor,
                        expected_fingerprint=fingerprint,
                        expected_compatibility=compatibility,
                        config=self.config,
                        train_config=train_config,
                        mode=resolved_mode,
                        expected_stage=candidate_stage,
                        mutate_model=False,
                    )
                    candidate_io = CheckpointIO.from_mapping(candidate_payload["io"])
                    if (
                        hashlib.sha256(candidate.read_bytes()).hexdigest()
                        != expected_digest
                    ):
                        raise BenchmarkResumeError(
                            f"committed checkpoint hash mismatch: {candidate.name}"
                        )
                    if not _checkpoint_is_committed(candidate_io, sidecar_io):
                        raise BenchmarkResumeError(
                            "checkpoint I/O sidecar counter rollback"
                        )
                    checkpoint_path = candidate
                    stage = candidate_stage
                    optimizer = candidate_optimizer
                    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    payload = candidate_payload
                    break
                if payload is None:
                    _progress_allows_fresh_resume(
                        output,
                        fingerprint_sha256=str(fingerprint["sha256"]),
                        initial_stage=initial_stage,
                        maximum_replay_updates=self.config.checkpoint_every,
                    )
                    stage = initial_stage
                    configure_benchmark_mode(model, resolved_mode, stage)
                    optimizer = (
                        build_warmup_optimizer(model, train_config.condition_lr)
                        if stage == "warmup"
                        else build_joint_optimizer(model, train_config)
                    )
                    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    io_counters = sidecar_io
                    publish_initial_checkpoint = True
                else:
                    io_counters = sidecar_io
                    model.load_state_dict(payload["model"], strict=True)
                    optimizer.load_state_dict(payload["optimizer"])
                    torch.set_rng_state(payload["torch_rng_state"])
                    random.setstate(payload["python_rng_state"])
                    numpy_state = payload["numpy_rng_state"]
                    if not isinstance(numpy_state, Mapping):
                        raise BenchmarkResumeError("invalid NumPy RNG state")
                    np.random.set_state(
                        (
                            str(numpy_state["bit_generator"]),
                            numpy_state["state"].cpu().numpy().astype(np.uint32),
                            int(numpy_state["position"]),
                            int(numpy_state["has_gauss"]),
                            float(numpy_state["cached_gaussian"]),
                        )
                    )
                    cuda_states = payload["cuda_rng_states"]
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_states)
                    anchor_value = payload["visual_anchor"]
                    if not isinstance(anchor_value, Mapping):
                        raise BenchmarkResumeError("invalid visual anchor")
                    visual_anchor = {
                        str(name): tensor for name, tensor in anchor_value.items()
                    }
                    warmup_step = int(payload["warmup_step"])
                    main_step = int(payload["main_step"])
                    resumed_from_main_step = main_step
                    guard = payload["gradient_guard"]
                    consecutive_zero_audio_visual_probes = int(
                        guard["consecutive_zero_audio_visual_probes"]
                    )
                    audio_visual_probe_count = int(guard["audio_visual_probe_count"])
            except BaseException:
                model.load_state_dict(original_model_state, strict=True)
                if optimizer is not None and original_optimizer_state is not None:
                    optimizer.load_state_dict(original_optimizer_state)
                torch.set_rng_state(original_torch_rng_state)
                random.setstate(original_python_rng_state)
                np.random.set_state(original_numpy_rng_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(original_cuda_rng_states)
                for parameter in model.parameters():
                    parameter.requires_grad_(original_requires_grad[id(parameter)])
                model.condition_enabled = original_condition_enabled
                raise
        else:
            configure_benchmark_mode(
                model, resolved_mode, "warmup" if stage == "warmup" else "main"
            )
            optimizer = (
                build_warmup_optimizer(model, train_config.condition_lr)
                if stage == "warmup"
                else build_joint_optimizer(model, train_config)
            )
            self._write_journal(
                output,
                stage=stage,
                observed_warmup_step=0,
                observed_main_step=0,
                exact_warmup_step=0,
                exact_main_step=0,
                fingerprint_sha256=str(fingerprint["sha256"]),
                io_counters=io_counters,
            )
        if publish_initial_checkpoint:
            initial_payload = self._checkpoint_payload(
                model=model,
                optimizer=optimizer,
                stage=stage,
                warmup_step=0,
                main_step=0,
                fingerprint=fingerprint,
                compatibility=compatibility,
                io_counters=io_counters,
                visual_anchor=visual_anchor,
                consecutive_zero_audio_visual_probes=0,
                audio_visual_probe_count=0,
            )
            self._write_checkpoint(
                output,
                initial_payload,
                stage=stage,
                progress_step=0,
                main_step=0,
                io_counters=io_counters,
            )

        if stage == "warmup":
            configure_benchmark_mode(model, resolved_mode, "warmup")
            if optimizer is None:
                optimizer = build_warmup_optimizer(model, train_config.condition_lr)
            warmup_indices = make_shared_indices(
                len(train_samples),
                self.config.conditioner_warmup_steps,
                self.config.seed,
            )
            while warmup_step < self.config.conditioner_warmup_steps:
                sample_index = warmup_indices[warmup_step]
                condition_warmup_step(
                    model,
                    _require_training_sample(train_samples[sample_index]),
                    optimizer,
                    audio_loss_fn,
                )
                warmup_step += 1
                if (
                    warmup_step % self.config.journal_every == 0
                    or warmup_step == self.config.conditioner_warmup_steps
                ):
                    self._write_journal(
                        output,
                        stage="warmup",
                        observed_warmup_step=warmup_step,
                        observed_main_step=0,
                        exact_warmup_step=warmup_step
                        - warmup_step % self.config.checkpoint_every,
                        exact_main_step=0,
                        fingerprint_sha256=str(fingerprint["sha256"]),
                        io_counters=io_counters,
                    )
                if (
                    warmup_step % self.config.checkpoint_every == 0
                    or warmup_step == self.config.conditioner_warmup_steps
                ):
                    payload = self._checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        stage="warmup",
                        warmup_step=warmup_step,
                        main_step=0,
                        fingerprint=fingerprint,
                        compatibility=compatibility,
                        io_counters=io_counters,
                        visual_anchor=visual_anchor,
                        consecutive_zero_audio_visual_probes=0,
                        audio_visual_probe_count=0,
                    )
                    self._write_checkpoint(
                        output,
                        payload,
                        stage="warmup",
                        progress_step=warmup_step,
                        main_step=0,
                        io_counters=io_counters,
                    )
            stage = "main"
            optimizer = None

        configure_benchmark_mode(model, resolved_mode, "main")
        if optimizer is None:
            optimizer = build_joint_optimizer(model, train_config)
        observed_at_resume = 0
        journal_path = output / "progress.json"
        if resume and journal_path.exists():
            try:
                observed_at_resume = int(
                    json.loads(journal_path.read_text(encoding="utf-8"))[
                        "observed_main_step"
                    ]
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise BenchmarkResumeError(
                    f"invalid progress journal: {error}"
                ) from error

        while main_step < self.config.main_updates:
            sample = _require_training_sample(train_samples[shared_indices[main_step]])
            next_step = main_step + 1
            if resolved_mode is BenchmarkMode.JOINT_CONDITIONED:
                stats = joint_train_step(
                    model,
                    sample,
                    optimizer,
                    train_config,
                    audio_loss_fn,
                    visual_anchor,
                    probe_audio_visual_gradient=(
                        next_step % train_config.gradient_probe_interval == 0
                    ),
                )
                if next_step % train_config.gradient_probe_interval == 0:
                    audio_visual_probe_count += 1
                    consecutive_zero_audio_visual_probes = (
                        consecutive_zero_audio_visual_probes + 1
                        if stats.audio_to_visual_grad_norm == 0
                        else 0
                    )
                    if (
                        consecutive_zero_audio_visual_probes
                        >= train_config.max_zero_audio_visual_grad_steps
                    ):
                        raise DisconnectedAudioVisualGradient(
                            "audio loss did not reach visual parameters for "
                            f"{consecutive_zero_audio_visual_probes} consecutive probes"
                        )
            elif resolved_mode is BenchmarkMode.AUDIO_ONLY:
                _audio_only_step(model, sample, optimizer, train_config, audio_loss_fn)
            else:
                _visual_only_step(model, sample, optimizer, train_config, visual_anchor)
            main_step = next_step
            if (
                main_step % self.config.journal_every == 0
                or main_step == self.config.main_updates
            ):
                self._write_journal(
                    output,
                    stage="main",
                    observed_warmup_step=warmup_step,
                    observed_main_step=main_step,
                    exact_warmup_step=warmup_step,
                    exact_main_step=main_step
                    - main_step % self.config.checkpoint_every,
                    fingerprint_sha256=str(fingerprint["sha256"]),
                    io_counters=io_counters,
                )
            milestone = main_step in self.config.milestones
            if main_step % self.config.checkpoint_every == 0 or milestone:
                payload = self._checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    stage="main",
                    warmup_step=warmup_step,
                    main_step=main_step,
                    fingerprint=fingerprint,
                    compatibility=compatibility,
                    io_counters=io_counters,
                    visual_anchor=visual_anchor,
                    consecutive_zero_audio_visual_probes=(
                        consecutive_zero_audio_visual_probes
                    ),
                    audio_visual_probe_count=audio_visual_probe_count,
                )
                self._write_checkpoint(
                    output,
                    payload,
                    stage="main",
                    progress_step=main_step,
                    main_step=main_step,
                    milestone=milestone,
                    final=main_step == self.config.main_updates,
                    io_counters=io_counters,
                )
                if milestone:
                    self._write_artifact_journal(
                        output,
                        fingerprint_sha256=str(fingerprint["sha256"]),
                        io_counters=io_counters,
                    )
            if interrupt_after_main_step == main_step:
                raise RuntimeError("injected benchmark interruption")

        final = output / "final.pt"
        artifact_manifest_path = output / "artifact_hashes.json"
        artifact_paths = {
            f"milestones/step_{step:06d}.pt": (
                output / "milestones" / f"step_{step:06d}.pt"
            )
            for step in self.config.milestones
        }
        artifact_paths["final.pt"] = final
        if resume and resumed_from_main_step == self.config.main_updates:
            if optimizer is None:
                raise BenchmarkResumeError("completed checkpoint has no optimizer")
            manifest_needs_publication = False
            try:
                artifact_manifest = json.loads(
                    artifact_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                artifact_manifest = None
                manifest_needs_publication = True
            manifest_fields = (
                set(artifact_manifest)
                if isinstance(artifact_manifest, Mapping)
                else set()
            )
            manifest_metadata_valid = (
                isinstance(artifact_manifest, Mapping)
                and manifest_fields
                == {"schema", "version", "fingerprint_sha256", "sha256"}
                and artifact_manifest["schema"] == f"{SCHEMA}.artifacts"
                and artifact_manifest["version"] == SCHEMA_VERSION
                and artifact_manifest["fingerprint_sha256"] == fingerprint["sha256"]
                and isinstance(artifact_manifest["sha256"], Mapping)
                and set(artifact_manifest["sha256"]).issubset(artifact_paths)
            )
            complete_manifest = manifest_metadata_valid and set(
                artifact_manifest["sha256"]
            ) == set(artifact_paths)
            if artifact_manifest is not None and not complete_manifest:
                # A complete-looking manifest with wrong provenance is a
                # tamper, while a truncated/incomplete object is recoverable
                # from the durable per-milestone transaction journal.
                if (
                    manifest_fields
                    == {"schema", "version", "fingerprint_sha256", "sha256"}
                    and not manifest_metadata_valid
                ):
                    raise BenchmarkResumeError(
                        "completed artifact hash manifest mismatch"
                    )
                manifest_needs_publication = True
            try:
                artifact_journal = json.loads(
                    (output / "artifact_journal.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise BenchmarkResumeError(
                    f"cannot read artifact transaction journal: {error}"
                ) from error
            if (
                not isinstance(artifact_journal, Mapping)
                or set(artifact_journal)
                != {"schema", "version", "fingerprint_sha256", "sha256"}
                or artifact_journal["schema"] != f"{SCHEMA}.artifact-journal"
                or artifact_journal["version"] != SCHEMA_VERSION
                or artifact_journal["fingerprint_sha256"] != fingerprint["sha256"]
                or not isinstance(artifact_journal["sha256"], Mapping)
                or not set(artifact_journal["sha256"]).issubset(artifact_paths)
            ):
                raise BenchmarkResumeError("artifact transaction journal mismatch")
            authoritative_bytes = checkpoint_path.read_bytes()
            authoritative_sha256 = hashlib.sha256(authoritative_bytes).hexdigest()
            expected_hashes = dict(artifact_journal["sha256"])
            for step in self.config.milestones[:-1]:
                name = f"milestones/step_{step:06d}.pt"
                if name not in expected_hashes:
                    raise BenchmarkResumeError(
                        f"artifact transaction journal is missing {name}"
                    )
            last_name = f"milestones/step_{self.config.main_updates:06d}.pt"
            expected_hashes[last_name] = authoritative_sha256
            expected_hashes["final.pt"] = authoritative_sha256
            if complete_manifest:
                manifest_hashes = dict(artifact_manifest["sha256"])
                if manifest_hashes != expected_hashes:
                    raise BenchmarkResumeError(
                        "completed artifact hash manifest mismatch"
                    )
            else:
                if manifest_metadata_valid:
                    for name, digest in artifact_manifest["sha256"].items():
                        if expected_hashes.get(name) != digest:
                            raise BenchmarkResumeError(
                                "partial artifact hash manifest mismatch"
                            )
                manifest_needs_publication = True
            required = [
                (
                    f"milestones/step_{step:06d}.pt",
                    output / "milestones" / f"step_{step:06d}.pt",
                    step,
                    step == self.config.main_updates,
                )
                for step in self.config.milestones
            ]
            required.append(("final.pt", final, self.config.main_updates, True))
            for (
                relative_name,
                artifact,
                expected_step,
                must_match_authoritative,
            ) in required:
                if not artifact.is_file():
                    if expected_step != self.config.main_updates:
                        raise BenchmarkResumeError(
                            f"required milestone is missing: {artifact.name}"
                        )
                    started = time.monotonic()
                    io_counters.checkpoint_bytes += _atomic_bytes(
                        artifact, authoritative_bytes
                    )
                    io_counters.checkpoint_writes += 1
                    io_counters.checkpoint_seconds += time.monotonic() - started
                    _persist_io_sidecar(output, str(fingerprint["sha256"]), io_counters)
                self._load_checkpoint(
                    artifact,
                    model=model,
                    optimizer=optimizer,
                    expected_visual_anchor=visual_anchor,
                    expected_fingerprint=fingerprint,
                    expected_compatibility=compatibility,
                    config=self.config,
                    train_config=train_config,
                    mode=resolved_mode,
                    expected_stage="main",
                    expected_main_step=expected_step,
                    mutate_model=False,
                )
                if (
                    hashlib.sha256(artifact.read_bytes()).hexdigest()
                    != expected_hashes[relative_name]
                ):
                    raise BenchmarkResumeError(
                        f"completed artifact hash mismatch: {artifact.name}"
                    )
                if (
                    must_match_authoritative
                    and expected_hashes[relative_name] != authoritative_sha256
                ):
                    raise BenchmarkResumeError(
                        f"completed artifact model identity mismatch: {artifact.name}"
                    )
            if manifest_needs_publication:
                started = time.monotonic()
                _atomic_bytes(
                    artifact_manifest_path,
                    _json_bytes(
                        {
                            "schema": f"{SCHEMA}.artifacts",
                            "version": SCHEMA_VERSION,
                            "fingerprint_sha256": fingerprint["sha256"],
                            "sha256": expected_hashes,
                        }
                    ),
                )
                io_counters.checkpoint_seconds += time.monotonic() - started
                _persist_io_sidecar(output, str(fingerprint["sha256"]), io_counters)
        else:
            missing = [
                name for name, path in artifact_paths.items() if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"required benchmark publications are missing: {', '.join(missing)}"
                )
            started = time.monotonic()
            _atomic_bytes(
                artifact_manifest_path,
                _json_bytes(
                    {
                        "schema": f"{SCHEMA}.artifacts",
                        "version": SCHEMA_VERSION,
                        "fingerprint_sha256": fingerprint["sha256"],
                        "sha256": {
                            name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in artifact_paths.items()
                        },
                    }
                ),
            )
            io_counters.checkpoint_seconds += time.monotonic() - started
            _persist_io_sidecar(output, str(fingerprint["sha256"]), io_counters)
        if not final.is_file():
            raise RuntimeError("authoritative final checkpoint was not published")
        _persist_io_sidecar(output, str(fingerprint["sha256"]), io_counters)
        return BenchmarkTrainingResult(
            mode=resolved_mode,
            completed_warmup_steps=warmup_step,
            completed_main_updates=main_step,
            resumed_from_main_step=resumed_from_main_step,
            redone_main_updates=max(0, observed_at_resume - resumed_from_main_step),
            selection=self.config.selection,
            final_checkpoint=final,
            milestones=tuple(
                output / "milestones" / f"step_{step:06d}.pt"
                for step in self.config.milestones
            ),
            io=io_counters,
        )
