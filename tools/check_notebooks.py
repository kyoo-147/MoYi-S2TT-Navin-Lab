"""Reject notebook outputs, execution state, and credential-like content."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SENSITIVE_PATTERNS = (
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"drive\.google\.com/(?:drive/)?folders/[A-Za-z0-9_-]+"),
)


def check_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                errors.append(f"{path}: cell {index} contains outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"{path}: cell {index} contains execution state")
    serialized = json.dumps(notebook)
    if any(pattern.search(serialized) for pattern in SENSITIVE_PATTERNS):
        errors.append(f"{path}: credential or private Drive reference detected")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = [
        error
        for path in sorted((root / "notebooks").glob("*.ipynb"))
        for error in check_notebook(path)
    ]
    if errors:
        print("Notebook check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Notebook check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
