from __future__ import annotations

import subprocess
import sys

import pytest

from avgaussianv2.cli.pilot import PilotProcessError, _wait_group, parse_gpus
from avgaussianv2.cli.pilot_worker import build_worker_component_identities


def test_three_gpu_parser_is_exact() -> None:
    assert parse_gpus("0,1,2") == (0, 1, 2)
    for invalid in ("0,1", "0,1,1", "0,-1,2", "a,1,2", "0,1,2,3"):
        with pytest.raises(ValueError):
            parse_gpus(invalid)


def test_pilot_help_is_cuda_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "avgaussianv2.cli.pilot", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--verify-only" in result.stdout


def test_parent_and_worker_share_one_component_identity_builder() -> None:
    identities = build_worker_component_identities("example.Model")
    assert identities["model_class"] == "example.Model"
    assert set(identities) == {
        "model_class",
        "warmup_optimizer_factory",
        "joint_optimizer_factory",
        "warmup_optimizer_class",
        "joint_optimizer_class",
        "warmup_step_fn",
        "joint_step_fn",
        "audio_loss_fn",
    }


def test_process_failures_are_aggregated_after_every_wait(tmp_path) -> None:
    class Handle:
        def __init__(self, code):
            self.code = code
            self.waited = False

        def wait(self):
            self.waited = True
            return self.code

        def terminate(self):
            pass

    handles = [Handle(3), Handle(0), Handle(7)]
    jobs = tuple(
        (f"job-{index}", handle, tmp_path / f"{index}.log")
        for index, handle in enumerate(handles)
    )
    with pytest.raises(PilotProcessError) as raised:
        _wait_group(jobs)
    assert all(handle.waited for handle in handles)
    assert [failure[:2] for failure in raised.value.failures] == [
        ("job-0", 3),
        ("job-2", 7),
    ]
