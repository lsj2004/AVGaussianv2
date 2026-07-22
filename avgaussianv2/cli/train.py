from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.backends.visual_ftgspp import FTGSVisualBackend
from avgaussianv2.checkpoint import build_checkpoint_state, save_checkpoint
from avgaussianv2.config import ProjectConfig, load_project_config
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.data.aligned import AlignedAVDataset
from avgaussianv2.experiment.evaluation import move_sample
from avgaussianv2.losses import AudioLoss
from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.rgbd import RGBDConditionEncoder
from avgaussianv2.train import TrainStepStats, run_condition_warmup, run_joint_finetune


@dataclass(frozen=True)
class TrainingBundle:
    model: nn.Module
    samples: Sequence[AlignedAVSample]
    audio_loss_fn: AudioLoss


@dataclass(frozen=True)
class TrainingResult:
    completed_stage: str
    history: list[dict]
    output_dir: str
    artifact_metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "completed_stage": self.completed_stage,
            "steps": len(self.history),
            "output_dir": self.output_dir,
            "artifact_metrics": self.artifact_metrics,
        }


BackendFactory = Callable[[ProjectConfig, torch.device], TrainingBundle]


class _DeviceSampleSequence(Sequence[AlignedAVSample]):
    def __init__(self, samples: Sequence[AlignedAVSample], device: torch.device) -> None:
        self.samples = samples
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [move_sample(sample, self.device) for sample in self.samples[index]]
        return move_sample(self.samples[index], self.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _stats_rows(stats: Sequence[TrainStepStats], stage: str) -> list[dict]:
    return [
        {
            "stage": stage,
            "total": row.total,
            "losses": row.losses,
            "gradient_norms": row.gradient_norms,
            "audio_to_visual_grad_norm": row.audio_to_visual_grad_norm,
        }
        for row in stats
    ]


def _write_ppm(path: Path, rgb: Tensor) -> None:
    array = rgb.detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("RGB preview must have three channels")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + array.tobytes())


def _write_pgm(path: Path, depth: Tensor) -> None:
    image = depth.detach().float().squeeze(-1).cpu()
    finite = torch.isfinite(image)
    if finite.any():
        valid = image[finite]
        low, high = valid.min(), valid.max()
        image = torch.where(finite, (image - low) / (high - low).clamp_min(1e-6), 0)
    else:
        image = torch.zeros_like(image)
    array = image.clamp(0, 1).mul(255).byte().numpy()
    height, width = array.shape
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + array.tobytes())


def _artifact_predictions(model: nn.Module, sample: AlignedAVSample):
    was_training = model.training
    original_condition_enabled = getattr(model, "condition_enabled", None)
    model.eval()
    try:
        with torch.no_grad():
            if original_condition_enabled is not None:
                model.condition_enabled = True
                condition_on_prediction = model(sample)
                model.condition_enabled = False
                condition_off_prediction = model(sample)
                prediction = (
                    condition_on_prediction
                    if original_condition_enabled
                    else condition_off_prediction
                )
                return (
                    prediction,
                    condition_on_prediction,
                    condition_off_prediction.predicted_audio,
                )

            condition_on_prediction = model(sample)
            condition_off = None
            audio_backend = getattr(model, "audio", None)
            if audio_backend is not None and hasattr(audio_backend, "render"):
                condition_off = audio_backend.render(
                    sample.audio_cam_pose,
                    sample.source_audio,
                    condition=None,
                )
            return condition_on_prediction, condition_on_prediction, condition_off
    finally:
        if original_condition_enabled is not None:
            model.condition_enabled = original_condition_enabled
        model.train(was_training)


def _default_backend_factory(config: ProjectConfig, device: torch.device) -> TrainingBundle:
    visual = FTGSVisualBackend.load(
        config.paths.visual_checkpoint,
        config.paths.visual_upstream_root,
    )
    audio = AudioGSBackend.load(
        config.paths.audio_checkpoint,
        embedding_dim=config.model.embedding_dim,
        upstream_root=config.paths.audio_upstream_root,
        model_class=config.model.audio_model_class,
    )
    model = AVGaussianFusionV2(
        visual=visual,
        condition_encoder=RGBDConditionEncoder(
            embedding_dim=config.model.embedding_dim,
            alpha_threshold=config.model.alpha_threshold,
        ),
        audio=audio,
    ).to(device)
    dataset = AlignedAVDataset(config, split="train")

    criterion = audio.build_criterion().to(device)
    return TrainingBundle(model=model, samples=dataset, audio_loss_fn=criterion)


def _write_artifacts(
    output_dir: Path,
    model: nn.Module,
    sample: AlignedAVSample,
    sample_rate: int,
) -> dict[str, float]:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    prediction, condition_on_prediction, condition_off = _artifact_predictions(model, sample)
    audio = prediction.predicted_audio[0].detach().float().cpu().transpose(0, 1).numpy()
    on_audio = (
        condition_on_prediction.predicted_audio[0]
        .detach()
        .float()
        .cpu()
        .transpose(0, 1)
        .numpy()
    )
    sf.write(artifacts / "sample_pred.wav", audio, sample_rate, subtype="FLOAT")
    sf.write(artifacts / "sample_condition_on.wav", on_audio, sample_rate, subtype="FLOAT")
    _write_ppm(artifacts / "sample_rgb.ppm", prediction.rgbd.rgb[0])
    _write_pgm(artifacts / "sample_depth.pgm", prediction.rgbd.depth[0])
    metrics: dict[str, float] = {}
    if condition_off is not None:
        off_audio = condition_off[0].detach().float().cpu().transpose(0, 1).numpy()
        sf.write(
            artifacts / "sample_condition_off.wav",
            off_audio,
            sample_rate,
            subtype="FLOAT",
        )
        metrics["condition_delta_mean_abs"] = float(
            (condition_on_prediction.predicted_audio - condition_off)
            .detach()
            .abs()
            .mean()
            .cpu()
        )
    _write_json(artifacts / "metrics.json", metrics)
    return metrics


def run_training(
    config: ProjectConfig,
    output_dir: str | Path,
    *,
    stage: str = "all",
    warmup_steps: int | None = None,
    joint_steps: int | None = None,
    backend_factory: BackendFactory | None = None,
    device: str | torch.device | None = None,
    condition_off: bool = False,
) -> TrainingResult:
    if stage not in {"warmup", "joint", "all"}:
        raise ValueError("stage must be warmup, joint, or all")
    if condition_off and stage != "joint":
        raise ValueError("condition-off ablation is joint-only")
    config.validate()
    seed_everything(config.train.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bundle = (backend_factory or _default_backend_factory)(config, resolved_device)
    if not bundle.samples:
        raise ValueError("training dataset must not be empty")
    if condition_off:
        if not hasattr(bundle.model, "condition_enabled"):
            raise TypeError("condition-off ablation requires AVGaussianFusionV2")
        bundle.model.condition_enabled = False
    samples = _DeviceSampleSequence(bundle.samples, resolved_device)
    warmup_count = config.train.warmup_steps if warmup_steps is None else warmup_steps
    joint_count = config.train.joint_steps if joint_steps is None else joint_steps
    history: list[dict] = []
    completed_stage = stage
    if stage in {"warmup", "all"}:
        warmup = run_condition_warmup(
            bundle.model,
            samples,
            steps=warmup_count,
            learning_rate=config.train.condition_lr,
            config=config.train,
            audio_loss_fn=bundle.audio_loss_fn,
        )
        history.extend(_stats_rows(warmup, "warmup"))
        completed_stage = "warmup"
    if stage in {"joint", "all"}:
        joint = run_joint_finetune(
            bundle.model,
            samples,
            steps=joint_count,
            config=config.train,
            audio_loss_fn=bundle.audio_loss_fn,
            require_audio_visual_gradient=not condition_off,
        )
        history.extend(_stats_rows(joint, "joint"))
        completed_stage = "joint"

    selected = samples[0]
    artifact_metrics = _write_artifacts(
        output, bundle.model, selected, config.model.sample_rate
    )
    _write_json(output / "resolved_config.json", asdict(config))
    _write_json(output / "loss_history.json", history)
    _write_json(
        output / "gradient_norms.json",
        [{"stage": row["stage"], **row["gradient_norms"]} for row in history],
    )
    _write_json(
        output / "selected_sample.json",
        {
            "scene_id": selected.scene_id,
            "camera": selected.camera,
            "frame_index": selected.frame_index,
            "time_seconds": selected.time_seconds,
        },
    )
    checkpoint_history = [
        {"total": float(row["total"]), "audio_to_visual": float(row["audio_to_visual_grad_norm"])}
        for row in history
    ]
    save_checkpoint(
        output / "checkpoint_latest.pt",
        build_checkpoint_state(
            bundle.model,
            optimizer=None,
            config=config,
            provenance={
                "visual_checkpoint": str(config.paths.visual_checkpoint),
                "audio_checkpoint": str(config.paths.audio_checkpoint),
            },
            stage=completed_stage,
            step=len(history),
            loss_history=checkpoint_history,
        ),
    )
    result = TrainingResult(
        completed_stage,
        history,
        str(output.resolve()),
        artifact_metrics,
    )
    _write_json(output / "run_summary.json", result.to_dict())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RGBD-conditioned AVGaussianFusionV2")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("warmup", "joint", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--joint-steps", type=int, default=None)
    parser.add_argument(
        "--condition-off",
        action="store_true",
        help="disable RGBD conditioning for a joint-only ablation run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_project_config(args.config)
    run_training(
        config,
        output_dir=args.output_dir,
        stage=args.stage,
        warmup_steps=args.warmup_steps,
        joint_steps=args.joint_steps,
        device=args.device,
        condition_off=args.condition_off,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
