from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import avgaussianv2.benchmark.orchestration as orchestration
from avgaussianv2.benchmark.orchestration import OrchestrationError


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess((), 0, stdout=stdout, stderr="")


def _probe_payload(command: list[str] | tuple[str, ...], gpu: str) -> str:
    source = command[-1]
    modules = [
        name
        for name in (
            "numpy",
            "yaml",
            "soundfile",
            "gsplat",
            "tinycudann",
            "scipy",
            "librosa",
            "torchaudio",
        )
        if f"'{name}'" in source
    ]
    return (
        json.dumps(
            {
                "cuda": "12.1",
                "device": f"GPU-{gpu}",
                "device_count": 1,
                "modules": {name: "test" for name in modules},
                "python": command[0],
                "torch": "2.1",
            }
        )
        + "\n"
    )


def test_preflight_probes_every_gpu_in_each_production_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    config = repository / "configs/benchmark_cam38/scene1_opera.yaml"
    calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        argv = tuple(command)
        env = kwargs.get("env")
        calls.append((argv, None if env is None else dict(env)))  # type: ignore[arg-type]
        if argv[0] == "nvidia-smi":
            return _completed("2, 10000\n5, 11000\n7, 12000\n")
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]  # type: ignore[index]
        return _completed(_probe_payload(argv, gpu))

    monkeypatch.setattr(orchestration.subprocess, "run", run)
    monkeypatch.setattr(
        orchestration,
        "_run_probe_command",
        lambda command, *, environment, **_: run(list(command), env=environment),
    )
    monkeypatch.setattr(orchestration.shutil, "which", lambda value: value)
    monkeypatch.setattr(orchestration, "MIN_FREE_BYTES", 0)

    result = orchestration._preflight(
        repository, config, tmp_path, (2, 5, 7), "/opt/av/bin/python"
    )

    probes = [(command, env) for command, env in calls if command[0] != "nvidia-smi"]
    assert len(probes) == 9
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in probes] == [
        "2",
        "2",
        "2",
        "5",
        "5",
        "5",
        "7",
        "7",
        "7",
    ]
    for offset in range(0, 9, 3):
        av, ftgspp, audiogs = probes[offset : offset + 3]
        assert av[0][0] == "/opt/av/bin/python"
        assert ftgspp[0][0].endswith("FreeTimeGSPlusPlus/.venv/bin/python")
        assert audiogs[0][:5] == ("conda", "run", "-n", "avcloud", "python")
        assert "torch.cuda.device_count()==1" in av[0][-1]
        assert "torch.cuda.synchronize()" in av[0][-1]
        assert "'gsplat'" in av[0][-1]
        assert "'tinycudann'" in ftgspp[0][-1]
        assert "'torchaudio'" in audiogs[0][-1]
    assert set(result["gpu_runtime_probes"]) == {"2", "5", "7"}
    assert set(result["gpu_runtime_probes"]["2"]) == {
        "avgaussianv2",
        "ftgspp",
        "audiogs",
    }


def test_preflight_fails_closed_on_environment_cuda_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    config = repository / "configs/benchmark_cam38/scene1_opera.yaml"

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return _completed("0, 10000\n1, 10000\n2, 10000\n")
        if command[0].endswith("FreeTimeGSPlusPlus/.venv/bin/python"):
            raise subprocess.CalledProcessError(
                1, command, stderr="tinycudann import failed"
            )
        return _completed(_probe_payload(command, "0"))

    monkeypatch.setattr(orchestration.subprocess, "run", run)
    monkeypatch.setattr(
        orchestration,
        "_run_probe_command",
        lambda command, *, environment, **_: run(list(command), env=environment),
    )
    monkeypatch.setattr(orchestration.shutil, "which", lambda value: value)
    monkeypatch.setattr(orchestration, "MIN_FREE_BYTES", 0)

    with pytest.raises(
        OrchestrationError, match=r"ftgspp CUDA preflight failed on physical GPU 0"
    ):
        orchestration._preflight(
            repository, config, tmp_path, (0, 1, 2), "/opt/av/bin/python"
        )


def test_suite_launches_no_native_process_when_real_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[tuple[object, ...]] = []

    class Runner:
        def start(self, *args: object, **kwargs: object) -> object:
            starts.append((*args, kwargs))
            raise AssertionError("native process launched after failed preflight")

    monkeypatch.setattr(orchestration, "_native_contracts_valid", lambda *_: False)

    with pytest.raises(OrchestrationError, match="CUDA probe failed"):
        orchestration.run_benchmark_suite(
            repository=tmp_path / "repo",
            output_dir=tmp_path / "output",
            runner=Runner(),  # type: ignore[arg-type]
            _preflight_fn=lambda *_args: (_ for _ in ()).throw(
                OrchestrationError("CUDA probe failed")
            ),
        )

    assert starts == []
    assert not (tmp_path / "output").exists()


def test_probe_timeout_terminates_then_kills_the_whole_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class HungProcess:
        pid = 1234
        returncode = None

        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["start_new_session"] is True

        def communicate(
            self, timeout: float | None = None
        ) -> tuple[str, str]:
            if timeout is not None:
                raise subprocess.TimeoutExpired(("probe",), timeout)
            self.returncode = -9
            return "", ""

    monkeypatch.setattr(orchestration.subprocess, "Popen", HungProcess)
    monkeypatch.setattr(
        orchestration.os,
        "killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        orchestration._run_probe_command(
            ("python", "-c", "pass"),
            environment={},
            timeout_seconds=0.01,
        )

    assert signals == [
        (1234, orchestration.signal.SIGTERM),
        (1234, orchestration.signal.SIGKILL),
    ]
