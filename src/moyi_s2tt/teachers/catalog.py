"""Validated loading for declarative teacher candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import TeacherSpec


def load_teacher_spec(path: Path) -> TeacherSpec:
    with path.open(encoding="utf-8") as handle:
        value: Any = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected teacher YAML object in {path}")
    return TeacherSpec.model_validate(value)


def load_teacher_catalog(root: Path) -> tuple[TeacherSpec, ...]:
    specs = tuple(load_teacher_spec(path) for path in sorted(root.glob("*.yaml")))
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("teacher catalog contains duplicate IDs")
    return specs
