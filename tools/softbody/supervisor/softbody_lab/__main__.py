"""Run with python -m softbody_lab from the sim-infra repository root."""

import argparse
import json
import sys
from pathlib import Path

from .artifacts import read_json, validate_trace, write_json
from .compare import calibrate, compare
from .fixtures import export_fixture
from .runner import ENVIRONMENTS, capture, doctor


def main(argv=None):
    parser = argparse.ArgumentParser(description="ManiSkill soft-body port verification")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local hardware; this does not prove physics")
    record = commands.add_parser("capture", help="Record a real reference or candidate rollout")
    record.add_argument("--source", required=True, type=Path)
    record.add_argument("--output", required=True, type=Path)
    record.add_argument("--role", choices=["reference", "candidate"], required=True)
    record.add_argument("--env-id", choices=ENVIRONMENTS, default="Fill-v0")
    record.add_argument("--seed", type=int, default=101)
    record.add_argument("--steps", type=int, default=20)
    record.add_argument("--control-mode", default="pd_joint_delta_pos")
    record.add_argument("--env-kwargs", type=json.loads, default={})
    record.add_argument("--reset-kwargs", type=json.loads, default={})
    record.add_argument("--replay", type=Path, help="Exported initial-state/action fixture directory")
    record.add_argument("--candidate-sim-backend", choices=['physx_cpu', 'physx_cuda'],
                        help="Explicit candidate execution backend; preserves frozen fixture metadata")
    record.add_argument("--actions-path", type=Path, help="Numeric .npy actions, no pickle")
    record.add_argument("--reference-initial-state-path", type=Path,
                        help="Reference-only numeric .npy demonstration initialization")
    record.add_argument("--record-mpm-wrenches", action="store_true",
                        help="Pinch: actual last rigid-step MPM wrenches; replay follows fixture metadata")
    check = commands.add_parser("validate")
    check.add_argument("trace", type=Path)
    export = commands.add_parser("export-fixture")
    export.add_argument("trace", type=Path)
    export.add_argument("output", type=Path)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("reference", type=Path)
    calibration.add_argument("repeats", nargs="+", type=Path)
    calibration.add_argument("--ranges", required=True, type=Path,
                             help="JSON with per-metric floors and maximums")
    calibration.add_argument("--output", required=True, type=Path)
    diff = commands.add_parser("compare")
    diff.add_argument("reference", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--protocol", required=True, type=Path)
    diff.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "capture":
            kwargs = vars(args).copy()
            kwargs.pop("command")
            result = {"recording": str(capture(**kwargs))}
        elif args.command == "validate":
            m = validate_trace(args.trace)
            result = {"valid": True, "samples": len(m["samples"]),
                      "meaning": "Artifact integrity only; physics acceptance needs compare."}
        elif args.command == "export-fixture":
            result = {"fixture": str(export_fixture(args.trace, args.output))}
        elif args.command == "calibrate":
            ranges = read_json(args.ranges)
            result = calibrate(args.reference, args.repeats, **ranges)
            write_json(args.output, result)
        else:
            result = compare(args.reference, args.candidate, read_json(args.protocol))
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({"error": str(exc), "kind": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
