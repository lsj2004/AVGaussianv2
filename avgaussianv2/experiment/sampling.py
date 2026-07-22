from __future__ import annotations

import numpy as np

from avgaussianv2.experiment.contracts import SharedIndices


def _draw_epochs(
    size: int, count: int, rng: np.random.Generator
) -> tuple[int, ...]:
    if size <= 0:
        raise ValueError("size must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")

    indices: list[int] = []
    while len(indices) < count:
        permutation = rng.permutation(size)
        remaining = count - len(indices)
        indices.extend(int(index) for index in permutation[:remaining])
    return tuple(indices)


def build_shared_indices(
    dataset_size: int, warmup_steps: int, joint_steps: int, seed: int
) -> SharedIndices:
    rng = np.random.default_rng(seed)
    warmup = _draw_epochs(dataset_size, warmup_steps, rng)
    joint = _draw_epochs(dataset_size, joint_steps, rng)
    return SharedIndices(warmup=warmup, joint=joint)


def evenly_spaced_indices(dataset_size: int, count: int) -> tuple[int, ...]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if count <= 0:
        raise ValueError("count must be positive")

    capped_count = min(dataset_size, count)
    positions = np.linspace(0, dataset_size - 1, num=capped_count)
    return tuple(int(index) for index in np.rint(positions))
