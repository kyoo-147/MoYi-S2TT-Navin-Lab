"""Single shared entry point used by local commands and thin Colab notebooks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .colab import environment_report

STAGES = ("data", "teacher", "tiny", "sequence-kd", "evaluation", "export")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m moyi_s2tt.runtime.runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment")
    stage = subparsers.add_parser("stage")
    stage.add_argument("name", choices=STAGES)
    args = parser.parse_args(argv)
    if args.command == "environment":
        print(environment_report().model_dump_json(indent=2))
        return 0
    if args.command == "stage":
        print(json.dumps({"stage": args.name, "status": "execution_not_implemented"}))
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
