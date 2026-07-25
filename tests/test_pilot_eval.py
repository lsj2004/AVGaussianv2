from __future__ import annotations

import subprocess
import sys

import pytest

from avgaussianv2.cli.pilot_eval import EvaluationSpec, parse_evaluation_spec


def test_evaluation_spec_matrix_is_exact() -> None:
    assert parse_evaluation_spec("joint_conditioned_on:on") == EvaluationSpec(
        "joint_conditioned_on", True
    )
    assert parse_evaluation_spec("joint_conditioned_off:off") == EvaluationSpec(
        "joint_conditioned_off", False
    )
    for invalid in (
        "baseline_imported:on",
        "frozen_visual_on:off",
        "condition_off:on",
        "unknown:off",
        "joint_conditioned_on",
    ):
        with pytest.raises(ValueError):
            parse_evaluation_spec(invalid)


def test_eval_help_is_cuda_lazy() -> None:
    script = (
        "import sys, runpy; "
        "sys.argv=['pilot_eval','--help']; "
        "runpy.run_module('avgaussianv2.cli.pilot_eval',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--trust-upstream-artifacts" in result.stdout
