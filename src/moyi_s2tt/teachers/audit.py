"""Validated configuration and sanitized evidence for real teacher audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from ..config import StrictModel
from ..directions import DIRECTION_KEYS
from .base import GenerationConfig


class AuditTeacher(StrictModel):
    spec: str
    batch_size: int = Field(gt=0)
    parameters: dict[str, str | int | float | bool | None]

    def generation(self, seed: int) -> GenerationConfig:
        return GenerationConfig(seed=seed, batch_size=self.batch_size, parameters=self.parameters)


class HoldTeacher(StrictModel):
    status: str
    reason: str


class TeacherAuditConfig(StrictModel):
    schema_version: int = 1
    id: str
    direction: str
    seed: int
    max_rows: int = Field(gt=0, le=500)
    max_audio_seconds: float = Field(gt=0, le=30)
    source_split: str
    frozen_evaluation: str
    asr: AuditTeacher
    mt: AuditTeacher
    direct_s2tt: AuditTeacher
    seamless_m4t: HoldTeacher

    def model_post_init(self, _context: Any) -> None:
        if self.direction not in DIRECTION_KEYS:
            raise ValueError(f"unsupported audit direction: {self.direction}")


class TeacherAuditEvidence(StrictModel):
    schema_version: int = 1
    audit_id: str
    status: str
    rows_requested: int
    rows_completed: int
    cache_hits: dict[str, int]
    elapsed_seconds: dict[str, float]
    teacher_revisions: dict[str, str]
    generation_sha256: dict[str, str]
    gpu: str
    package_versions: dict[str, str | None]
    aggregate: dict[str, float | str] = Field(default_factory=dict)
    output_location: str = "private_storage_only"
    output_text_committed: bool = False
    limitations: tuple[str, ...] = ()


def load_audit_config(path: Path) -> TeacherAuditConfig:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TeacherAuditConfig.model_validate(value)


def write_sanitized_evidence(path: Path, evidence: TeacherAuditEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
