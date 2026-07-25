from __future__ import annotations

import argparse
import json
import os
import random
import re
import runpy
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _strict_point_prefix(config_path: Path) -> tuple[list[int], int]:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    frames = config["data"]["frames"]
    start = frames["start"]
    stop = frames["stop"]
    stride = config["init"]["keyframe_stride"]
    points = Path(config["init"]["points_path"])
    if (
        type(start) is not int
        or type(stop) is not int
        or type(stride) is not int
        or start != 0
        or stop <= start
        or stride <= 0
        or points.is_symlink()
        or not points.is_dir()
    ):
        raise RuntimeError("unsafe FTGS++ point-resume configuration")
    expected = list(range(start, stop, stride))
    if expected[-1] != stop - 1:
        expected.append(stop - 1)
    names = [f"f{frame:06d}.ply" for frame in expected]
    entries = list(points.iterdir())
    actual = {entry.name for entry in entries}
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or not actual.issubset(names)
        or actual != set(names[: len(actual)])
        or not actual
        or len(actual) == len(names)
    ):
        raise RuntimeError("FTGS++ point resume requires a strict nonempty prefix")
    for name in names[: len(actual)]:
        path = points / name
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise RuntimeError("unsafe FTGS++ point prefix entry")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                if [stream.readline(), stream.readline()] != [
                    b"ply\n",
                    b"format binary_little_endian 1.0\n",
                ]:
                    raise RuntimeError("invalid FTGS++ point prefix header")
                match = re.fullmatch(
                    rb"element vertex ([1-9][0-9]*)\n", stream.readline()
                )
                expected_properties = [
                    b"property float x\n",
                    b"property float y\n",
                    b"property float z\n",
                    b"property uchar red\n",
                    b"property uchar green\n",
                    b"property uchar blue\n",
                    b"end_header\n",
                ]
                if (
                    match is None
                    or [stream.readline() for _ in expected_properties]
                    != expected_properties
                ):
                    raise RuntimeError("invalid FTGS++ point prefix payload")
                vertex_count = int(match.group(1))
                if opened.st_size != stream.tell() + vertex_count * 15:
                    raise RuntimeError("invalid FTGS++ point prefix payload")
                vertices = np.fromfile(
                    stream,
                    dtype=np.dtype(
                        [
                            ("x", "<f4"),
                            ("y", "<f4"),
                            ("z", "<f4"),
                            ("red", "u1"),
                            ("green", "u1"),
                            ("blue", "u1"),
                        ]
                    ),
                    count=vertex_count,
                )
                if len(vertices) != vertex_count or not all(
                    np.isfinite(vertices[axis]).all() for axis in ("x", "y", "z")
                ):
                    raise RuntimeError("invalid FTGS++ point prefix coordinates")
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise RuntimeError("FTGS++ point prefix changed during audit")
        finally:
            os.close(descriptor)
    return expected, len(actual)


def _install_point_resume_guard(arguments: list[str]) -> int:
    if (
        len(arguments) < 3
        or arguments[0] != "dynerf"
        or arguments.count("--scenes") != 1
        or arguments.count("--from") != 1
        or arguments.count("--to") != 1
        or arguments[arguments.index("--from") + 1] != "points"
        or arguments[arguments.index("--to") + 1] != "train"
    ):
        raise RuntimeError("--resume-points requires the exact FTGS++ points-to-train run")
    scene = arguments[arguments.index("--scenes") + 1]
    config_path = Path(arguments[1]) / f"{scene}.toml"
    expected, prefix = _strict_point_prefix(config_path)
    import ftgspp.init.points as point_module

    original = point_module.keyframe_indices

    def remaining_keyframes(num_frames: int, keyframe_stride: int) -> list[int]:
        resolved = original(num_frames, keyframe_stride)
        if resolved != expected:
            raise RuntimeError("FTGS++ runtime keyframes differ from resume audit")
        return resolved[prefix:]

    point_module.keyframe_indices = remaining_keyframes
    print(
        f"FTGS++ point resume: verified/skipping {prefix} of {len(expected)} keyframes"
    )
    return prefix


def seed_everything(seed: int, argv: list[str]) -> dict[str, Any]:
    if seed != 42:
        raise ValueError("strict cam38 FTGS++ seed must be 42")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 must be set before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "schema": "ftgspp_seed_v1",
        "seed": seed,
        "pythonhashseed": "42",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "argv": argv,
    }


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed FTGS++ deterministically before executing upstream code"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--resume-points", action="store_true")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--script", type=Path)
    target.add_argument("--module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    target_name = str(args.script) if args.script else str(args.module)
    argv = [target_name, *arguments]
    record_argv = list(argv)
    if args.resume_points:
        prefix = _install_point_resume_guard(arguments)
        record_argv.append(f"avgaussianv2:resume-points-prefix={prefix}")
    record = seed_everything(args.seed, record_argv)
    _write_record(args.record, record)
    sys.argv = argv
    if args.script:
        runpy.run_path(str(args.script), run_name="__main__")
    else:
        runpy.run_module(str(args.module), run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
