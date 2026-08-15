"""Export the sanitized training evidence schema."""

import json
from pathlib import Path

from moyi_s2tt.training.reports import TrainingEvidence


def main() -> int:
    path = Path("data/schemas/training-evidence.schema.json")
    payload = json.dumps(TrainingEvidence.model_json_schema(), indent=2, sort_keys=True)
    path.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
