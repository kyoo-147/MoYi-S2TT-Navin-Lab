"""Command-line entry point for reproducible lab operations."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import validate_config_tree
from .data.manifest import iter_manifest
from .data.splits import find_split_leakage
from .directions import ALL_DIRECTIONS
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
