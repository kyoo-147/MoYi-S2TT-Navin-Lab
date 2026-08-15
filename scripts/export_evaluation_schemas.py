"""Regenerate checked-in frozen-evaluation and reference JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path

from moyi_s2tt.evaluation.contracts import EvaluationReference, FrozenEvaluationSet


def write_schema(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    write_schema(
        Path("data/schemas/frozen-evaluation.schema.json"),
        FrozenEvaluationSet.model_json_schema(),
    )
    write_schema(
        Path("data/schemas/evaluation-reference.schema.json"),
        EvaluationReference.model_json_schema(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
