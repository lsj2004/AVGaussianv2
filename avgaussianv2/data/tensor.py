from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import Tensor

from avgaussianv2.contracts import AlignedAVSample


def move_sample(
    sample: AlignedAVSample,
    device: torch.device | str,
) -> AlignedAVSample:
    """Move tensor fields while preserving sample metadata."""
    destination = torch.device(device)
    changes = {
        name: value.to(destination)
        for name, value in vars(sample).items()
        if isinstance(value, Tensor)
    }
    return replace(sample, **changes)


class DeviceSampleSequence(Sequence[AlignedAVSample]):
    """Lazily transfer samples to one device when indexed."""

    def __init__(
        self,
        samples: Sequence[AlignedAVSample],
        device: torch.device,
    ) -> None:
        self.samples = samples
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def records(self):
        return getattr(self.samples, "records", None)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [
                move_sample(sample, self.device)
                for sample in self.samples[index]
            ]
        return move_sample(self.samples[index], self.device)


__all__ = ["DeviceSampleSequence", "move_sample"]
