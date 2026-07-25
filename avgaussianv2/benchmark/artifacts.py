"""Crash-safe immutable generations used by benchmark evaluation and reports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from avgaussianv2.benchmark.output import (
    BenchmarkOutputError,
    BenchmarkOutputLock,
    BenchmarkOutputReadLock,
)

class ArtifactError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_file(path: Path, data: bytes) -> None:
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


def publish_generation(
    output_dir: Path,
    *,
    schema: str,
    files: Mapping[str, bytes],
    identity: Mapping[str, object],
    overwrite: bool = False,
) -> tuple[Path, str]:
    original = Path(os.path.abspath(output_dir))
    try:
        with BenchmarkOutputLock(original) as pinned:
            if (pinned / "current.json").exists() and not overwrite:
                raise ArtifactError(
                    "artifact already exists; use resume or overwrite"
                )
            generation, digest = _publish_pinned(
                pinned, schema=schema, files=files, identity=identity
            )
            relative = generation.relative_to(pinned)
        return original / relative, digest
    except BenchmarkOutputError as error:
        raise ArtifactError(str(error)) from error


def _publish_pinned(
    output_dir: Path,
    *,
    schema: str,
    files: Mapping[str, bytes],
    identity: Mapping[str, object],
) -> tuple[Path, str]:
    generations = output_dir / "generations"
    generations.mkdir(exist_ok=True)
    if generations.is_symlink() or not generations.is_dir():
        raise ArtifactError("artifact generations must be a non-symlink directory")
    generation = f"generation-{uuid.uuid4().hex}"
    staging = generations / f".{generation}.tmp"
    final = generations / generation
    staging.mkdir()
    try:
        hashes: dict[str, str] = {}
        for name, data in sorted(files.items()):
            if (
                not name
                or Path(name).name != name
                or name in {"manifest.json", "current.json"}
            ):
                raise ArtifactError(f"unsafe artifact filename: {name!r}")
            (staging / name).write_bytes(data)
            hashes[name] = sha256(data)
        manifest = {
            "schema": f"{schema}.generation",
            "version": 1,
            "identity": dict(identity),
            "sha256": hashes,
        }
        manifest_data = canonical_json(manifest)
        (staging / "manifest.json").write_bytes(manifest_data)
        for path in staging.iterdir():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, final)
        descriptor = os.open(generations, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digest = sha256(manifest_data)
        _atomic_file(
            output_dir / "current.json",
            canonical_json(
                {
                    "schema": f"{schema}.current",
                    "version": 1,
                    "generation": f"generations/{generation}",
                    "manifest_sha256": digest,
                }
            ),
        )
        return final, digest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_generation(
    output_dir: Path,
    *,
    schema: str,
    expected_identity: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, bytes], dict[str, object], str]:
    original = Path(os.path.abspath(output_dir))
    try:
        with BenchmarkOutputReadLock(original) as pinned:
            generation, files, manifest, digest = _load_pinned(
                pinned,
                schema=schema,
                expected_identity=expected_identity,
            )
            relative = generation.relative_to(pinned)
        return original / relative, files, manifest, digest
    except BenchmarkOutputError as error:
        raise ArtifactError(str(error)) from error


def _load_pinned(
    output_dir: Path,
    *,
    schema: str,
    expected_identity: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, bytes], dict[str, object], str]:
    pointer_path = output_dir / "current.json"
    try:
        pointer_metadata = pointer_path.lstat()
        if (
            not stat.S_ISREG(pointer_metadata.st_mode)
            or pointer_metadata.st_nlink != 1
        ):
            raise ArtifactError("current artifact pointer is unsafe")
        pointer_data = pointer_path.read_bytes()
        pointer = json.loads(pointer_data)
    except (OSError, ValueError) as error:
        raise ArtifactError(f"cannot read current artifact pointer: {error}") from error
    if (
        not isinstance(pointer, Mapping)
        or set(pointer)
        != {"schema", "version", "generation", "manifest_sha256"}
        or pointer["schema"] != f"{schema}.current"
        or pointer["version"] != 1
    ):
        raise ArtifactError("current artifact pointer schema mismatch")
    relative = pointer["generation"]
    if (
        not isinstance(relative, str)
        or not relative.startswith("generations/generation-")
        or Path(relative).parts != ("generations", Path(relative).name)
    ):
        raise ArtifactError("unsafe artifact generation path")
    generation = output_dir / relative
    if generation.is_symlink() or not generation.is_dir():
        raise ArtifactError("artifact generation is missing or unsafe")
    try:
        manifest_path = generation / "manifest.json"
        manifest_metadata = manifest_path.lstat()
        if (
            not stat.S_ISREG(manifest_metadata.st_mode)
            or manifest_metadata.st_nlink != 1
        ):
            raise ArtifactError("artifact manifest is unsafe")
        manifest_data = manifest_path.read_bytes()
        manifest = json.loads(manifest_data)
    except (OSError, ValueError) as error:
        raise ArtifactError(f"cannot read artifact manifest: {error}") from error
    if sha256(manifest_data) != pointer["manifest_sha256"]:
        raise ArtifactError("artifact manifest hash mismatch")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema", "version", "identity", "sha256"}
        or manifest["schema"] != f"{schema}.generation"
        or manifest["version"] != 1
        or not isinstance(manifest["identity"], Mapping)
        or not isinstance(manifest["sha256"], Mapping)
    ):
        raise ArtifactError("artifact manifest schema mismatch")
    if (
        expected_identity is not None
        and dict(manifest["identity"]) != dict(expected_identity)
    ):
        raise ArtifactError("artifact identity mismatch")
    files: dict[str, bytes] = {}
    for name, digest in manifest["sha256"].items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
        ):
            raise ArtifactError("artifact manifest file entry is invalid")
        try:
            artifact = generation / name
            metadata = artifact.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ArtifactError(f"artifact is unsafe: {name}")
            data = artifact.read_bytes()
        except OSError as error:
            raise ArtifactError(f"cannot read artifact {name}: {error}") from error
        if sha256(data) != digest:
            raise ArtifactError(f"artifact hash mismatch: {name}")
        files[name] = data
    actual = {path.name for path in generation.iterdir()}
    expected = {*files, "manifest.json"}
    if actual != expected:
        raise ArtifactError("artifact generation contains unexpected files")
    return generation, files, dict(manifest), str(pointer["manifest_sha256"])
