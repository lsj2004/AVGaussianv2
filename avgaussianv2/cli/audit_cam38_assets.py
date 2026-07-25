from __future__ import annotations

import argparse
import json

from avgaussianv2.benchmark.assets import (
    EXPECTED,
    audit_audiogs_conversion,
    audit_ftgspp_flow_cache,
    audit_ftgspp_resume_state,
    audit_ftgspp_seed_record,
    audit_ftgspp_train_source,
    audit_ftgspp_upstream_config,
    audit_initialization_provenance,
    audit_protocol_config,
    prepare_fresh_ftgspp_namespaces,
    quarantine_interrupted_ftgspp_flow_tail,
    render_ftgspp_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict cam38 benchmark assets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--ftgspp-train-source")
    parser.add_argument("--allowed-sampled-root")
    parser.add_argument("--audiogs-conversion")
    parser.add_argument("--expected-clips", type=int)
    parser.add_argument("--expected-audio-root")
    parser.add_argument("--expected-cameras-npz")
    parser.add_argument("--expected-output-root")
    parser.add_argument("--ftgspp-template")
    parser.add_argument("--ftgspp-output")
    parser.add_argument("--repo-root")
    parser.add_argument("--ftgspp-root")
    parser.add_argument("--prepare-ftgspp-namespaces", action="store_true")
    parser.add_argument("--ftgspp-run-root")
    parser.add_argument("--ftgspp-marker-root")
    parser.add_argument("--audit-ftgspp-flow", action="store_true")
    parser.add_argument("--audit-ftgspp-resume", action="store_true")
    parser.add_argument(
        "--recover-interrupted-ftgspp-flow-tail", action="store_true"
    )
    parser.add_argument("--ftgspp-termination-log")
    parser.add_argument("--ftgspp-quarantine-root")
    parser.add_argument("--ftgspp-seed-record")
    args = parser.parse_args()
    raw = audit_protocol_config(args.config)
    scene_id = raw["scene"]["id"]
    if args.provenance:
        audit_initialization_provenance(
            args.provenance, expected_scene=scene_id
        )
    details = {}
    if args.ftgspp_seed_record:
        details["ftgspp_seed"] = audit_ftgspp_seed_record(
            args.ftgspp_seed_record, expected_scene=scene_id
        )
    if args.ftgspp_train_source:
        if not args.allowed_sampled_root:
            parser.error("--ftgspp-train-source requires --allowed-sampled-root")
        details["ftgspp_source"] = audit_ftgspp_train_source(
            args.ftgspp_train_source,
            allowed_sampled_root=args.allowed_sampled_root,
        )
    if args.audiogs_conversion:
        audio_expected = (
            args.expected_clips,
            args.expected_audio_root,
            args.expected_cameras_npz,
            args.expected_output_root,
        )
        if not all(value is not None for value in audio_expected):
            parser.error(
                "--audiogs-conversion requires --expected-clips and all "
                "three expected path arguments"
            )
        if args.expected_clips != EXPECTED[scene_id]["audio_clips"]:
            parser.error(
                f"{scene_id} requires --expected-clips "
                f"{EXPECTED[scene_id]['audio_clips']}"
            )
        details["audiogs"] = audit_audiogs_conversion(
            args.audiogs_conversion,
            expected_scene=EXPECTED[scene_id]["audio_scene"],
            expected_clips=args.expected_clips,
            epochs=61,
            expected_audio_root=args.expected_audio_root,
            expected_cameras_npz=args.expected_cameras_npz,
            expected_output_root=args.expected_output_root,
        )
    render_values = (
        args.ftgspp_template,
        args.ftgspp_output,
        args.repo_root,
        args.ftgspp_root,
    )
    if any(render_values):
        if not all(render_values) or not args.allowed_sampled_root:
            parser.error(
                "FTGS++ render/audit requires --ftgspp-template, --ftgspp-output, "
                "--repo-root, --ftgspp-root, and --allowed-sampled-root"
            )
        if not (
            args.audit_ftgspp_resume
            or args.recover_interrupted_ftgspp_flow_tail
        ):
            render_ftgspp_config(
                args.ftgspp_template,
                args.ftgspp_output,
                repo_root=args.repo_root,
                ftgspp_root=args.ftgspp_root,
                sampled_scene_root=args.allowed_sampled_root,
            )
        details["ftgspp_config"] = audit_ftgspp_upstream_config(
            args.ftgspp_output,
            protocol_config=args.config,
            repo_root=args.repo_root,
            ftgspp_root=args.ftgspp_root,
            sampled_scene_root=args.allowed_sampled_root,
        )
        if args.prepare_ftgspp_namespaces:
            if (
                not args.ftgspp_run_root
                or not args.allowed_sampled_root
                or not args.ftgspp_marker_root
            ):
                parser.error(
                    "--prepare-ftgspp-namespaces requires --ftgspp-run-root "
                    "--ftgspp-marker-root, and --allowed-sampled-root"
                )
            prepare_fresh_ftgspp_namespaces(
                [
                    *details["ftgspp_config"]["namespaces"],
                    args.ftgspp_run_root,
                ],
                scene_id=scene_id,
                source_root=args.allowed_sampled_root,
                marker_root=args.ftgspp_marker_root,
            )
        if args.audit_ftgspp_flow:
            details["ftgspp_flow"] = audit_ftgspp_flow_cache(
                details["ftgspp_config"]["namespaces"][-1],
                frame_count=raw["benchmark"]["expected_test_samples"],
                keyframe_stride=10,
            )
        if args.recover_interrupted_ftgspp_flow_tail:
            if not args.ftgspp_termination_log or not args.ftgspp_quarantine_root:
                parser.error(
                    "--recover-interrupted-ftgspp-flow-tail requires "
                    "--ftgspp-termination-log and --ftgspp-quarantine-root"
                )
            details["ftgspp_recovery"] = quarantine_interrupted_ftgspp_flow_tail(
                details["ftgspp_config"]["namespaces"][-1],
                frame_count=raw["benchmark"]["expected_test_samples"],
                keyframe_stride=10,
                termination_log=args.ftgspp_termination_log,
                quarantine_root=args.ftgspp_quarantine_root,
            )
        if args.audit_ftgspp_resume:
            required = (
                args.ftgspp_train_source,
                args.ftgspp_run_root,
                args.ftgspp_marker_root,
                args.ftgspp_seed_record,
            )
            if not all(required):
                parser.error(
                    "--audit-ftgspp-resume requires --ftgspp-train-source, "
                    "--ftgspp-run-root, --ftgspp-marker-root, and "
                    "--ftgspp-seed-record"
                )
            details["ftgspp_resume"] = audit_ftgspp_resume_state(
                scene_id=scene_id,
                source_root=args.allowed_sampled_root,
                train_source=args.ftgspp_train_source,
                namespaces=details["ftgspp_config"]["namespaces"],
                run_root=args.ftgspp_run_root,
                marker_root=args.ftgspp_marker_root,
                prep_seed_record=args.ftgspp_seed_record,
                frame_count=raw["benchmark"]["expected_test_samples"],
                keyframe_stride=10,
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "scene_id": scene_id,
                "test_camera": "cam38",
                **details,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
