from __future__ import annotations

import subprocess
import sys

import pytest

from avgaussianv2.cli.pilot import (
    LaunchSpec,
    PilotProcessError,
    _evenly_spaced,
    _launch_group,
    _shared_indices,
    _wait_group,
    parse_gpus,
)
from avgaussianv2.cli.pilot_worker import build_worker_component_identities
from avgaussianv2.experiment.contracts import PilotConfig, Variant


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


def test_group_start_exception_terminates_and_reaps_prior_children(tmp_path) -> None:
    class Handle:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return -15

    class Runner:
        assignments = []

        def __init__(self):
            self.handles = [Handle(), Handle()]
            self.position = 0

        def start(self, command, *, env, log_path):
            self.assignments.append((tuple(command), dict(env), log_path))
            if self.position == 2:
                raise OSError("cannot start")
            handle = self.handles[self.position]
            self.position += 1
            return handle

    runner = Runner()
    specs = tuple(
        LaunchSpec(
            f"job-{index}",
            ("python", "-m", "worker", str(index)),
            index,
            tmp_path / f"job-{index}.log",
        )
        for index in range(3)
    )
    with pytest.raises(PilotProcessError) as raised:
        _launch_group(runner, specs)
    assert all(handle.terminated and handle.waited for handle in runner.handles)
    assert [failure[0] for failure in raised.value.failures] == [
        "job-0",
        "job-1",
        "job-2",
    ]
    assert [assignment[1]["CUDA_VISIBLE_DEVICES"] for assignment in runner.assignments] == [
        "0",
        "1",
        "2",
    ]
    assert [assignment[2] for assignment in runner.assignments] == [
        tmp_path / "job-0.log",
        tmp_path / "job-1.log",
        tmp_path / "job-2.log",
    ]


def test_shared_sequences_are_deterministic_and_condition_off_has_no_warmup() -> None:
    pilot = PilotConfig(warmup_steps=7, joint_steps=11)
    first = _shared_indices(123, 5, pilot)
    second = _shared_indices(123, 5, pilot)
    assert first == second
    assert len(first.warmup) == 7
    assert len(first.joint) == 11
    assert first.for_variant(Variant.CONDITION_OFF).warmup == ()
    assert first.for_variant(Variant.CONDITION_OFF).joint == first.joint
    assert _evenly_spaced(100, 32) == _evenly_spaced(100, 32)
    assert _evenly_spaced(3, 32) == (0, 1, 2)
