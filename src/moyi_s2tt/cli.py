"""Command-line entry point for reproducible lab operations."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import validate_config_tree
from .data.manifest import iter_manifest
from .data.source import iter_source_manifest
from .data.splits import find_split_leakage
from .directions import ALL_DIRECTIONS
from .evaluation.contracts import load_frozen_evaluation
from .teachers.catalog import load_teacher_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moyi-s2tt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-directions", help="list canonical translation directions")
    validate = subparsers.add_parser(
        "validate-configs", help="validate repository config contracts"
    )
    validate.add_argument("--root", type=Path, default=Path.cwd())
    manifest = subparsers.add_parser("validate-manifest", help="validate JSONL and split safety")
    manifest.add_argument("path", type=Path)
    source_manifest = subparsers.add_parser(
        "validate-source-manifest", help="validate source JSONL and split safety"
    )
    source_manifest.add_argument("path", type=Path)
    evaluation = subparsers.add_parser(
        "validate-evaluation", help="validate a frozen evaluation contract"
    )
    evaluation.add_argument("path", type=Path)
    teachers = subparsers.add_parser(
        "validate-teachers", help="validate declarative teacher candidates"
    )
    teachers.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-directions":
        print(json.dumps([direction.key for direction in ALL_DIRECTIONS]))
        return 0
    if args.command == "validate-configs":
        print(json.dumps(validate_config_tree(args.root), sort_keys=True))
        return 0
    if args.command == "validate-manifest":
        records = list(iter_manifest(args.path))
        issues = find_split_leakage(records)
        if issues:
            raise ValueError(f"manifest has {len(issues)} leakage or duplicate issue(s)")
        print(json.dumps({"rows": len(records), "split_issues": 0}, sort_keys=True))
        return 0
    if args.command == "validate-source-manifest":
        source_records = list(iter_source_manifest(args.path))
        issues = find_split_leakage(source_records)
        if issues:
            raise ValueError(f"source manifest has {len(issues)} leakage or duplicate issue(s)")
        print(json.dumps({"rows": len(source_records), "split_issues": 0}, sort_keys=True))
        return 0
    if args.command == "validate-evaluation":
        frozen = load_frozen_evaluation(args.path)
        print(
            json.dumps(
                {
                    "sha256": frozen.sha256,
                    "test": len(frozen.test_semantic_ids),
                    "training_exclusions": len(frozen.training_exclusion_semantic_ids),
                    "validation": len(frozen.validation_semantic_ids),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-teachers":
        specs = load_teacher_catalog(args.root / "configs" / "teachers")
        print(
            json.dumps(
                {
                    "approved": sum(spec.status == "approved" for spec in specs),
                    "teachers": len(specs),
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
