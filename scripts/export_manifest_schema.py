"""Regenerate the checked-in canonical manifest JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from moyi_s2tt.data.manifest import ManifestRecord


def main() -> int:
    destination = Path("data/schemas/manifest.schema.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(ManifestRecord.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
