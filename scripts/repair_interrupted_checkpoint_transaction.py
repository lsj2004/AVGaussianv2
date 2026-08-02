#!/usr/bin/env python3
"""Audit and recover one crash-interrupted benchmark checkpoint transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avgaussianv2.benchmark.artifacts import repository_identity  # noqa: E402
from avgaussianv2.benchmark.output import (  # noqa: E402
    BenchmarkOutputLock,
    validate_output_children,
)
from avgaussianv2.benchmark.training import (  # noqa: E402
    recover_interrupted_checkpoint_transaction,
    verify_resume_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(worker: Path) -> dict[str, object]:
    checkpoints = worker / "checkpoints"
    return {
        "progress_sha256": _sha256(worker / "progress.json"),
        "checkpoint_io_sha256": _sha256(worker / "checkpoint_io.json"),
        "checkpoint_sha256": {
            path.name: _sha256(path)
            for path in sorted(checkpoints.glob("*.pt"))
        },
        "milestones": sorted(
            path.name for path in (worker / "milestones").glob("*.pt")
        ),
        "atomic_temporaries": sorted(
            path.relative_to(worker).as_posix()
            for path in worker.rglob("*.tmp")
        ),
        "final_exists": (worker / "final.pt").is_file(),
        "artifact_hashes_exists": (worker / "artifact_hashes.json").is_file(),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stop-after-main-step", type=int)
    args = parser.parse_args()

    worker = args.worker.resolve()
    manifest_path = args.manifest.resolve()
    repository = args.repository.resolve()
    report_path = args.report.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    training_path = repository / "avgaussianv2/benchmark/training.py"
    identity = repository_identity(repository)

    with BenchmarkOutputLock(worker) as pinned:
        before = _snapshot(pinned)
        recovered = recover_interrupted_checkpoint_transaction(
            pinned,
            worker_manifest=manifest,
            stop_after_main_step=args.stop_after_main_step,
        )
        validate_output_children(pinned)
        verify_resume_artifacts(pinned, worker_manifest=manifest)
        after = _snapshot(pinned)

    report = {
        "schema": "avgaussianv2.interrupted-checkpoint-recovery",
        "version": 1,
        "worker": str(worker),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "tool_repository": identity,
        "tool_sha256": _sha256(script_path),
        "training_module_sha256": _sha256(training_path),
        "stop_after_main_step": args.stop_after_main_step,
        "recovered": recovered,
        "before": before,
        "after": after,
        "verification": "passed",
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
