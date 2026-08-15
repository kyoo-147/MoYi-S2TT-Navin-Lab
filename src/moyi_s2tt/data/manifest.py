"""Canonical, provenance-preserving S2TT manifest records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..config import StrictModel

LanguageCode = Literal["vi", "en", "zh", "ko"]
Split = Literal["train", "validation", "test"]
TargetKind = Literal["gold", "teacher", "synthetic"]
Domain = Literal["conversation", "industrial", "general_read"]
MaterializationStatus = Literal["metadata_only", "ready"]
FilterStatus = Literal["unreviewed", "keep", "reject"]


class ManifestRecord(StrictModel):
    """One source-audio to target-text example with complete lineage."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    source_item_id: str
    semantic_group_id: str
    source_locale: str
    source_audio_ref: str
    audio_path: str | None = None
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duration_s: float = Field(gt=0, le=30)
    sample_rate: int = Field(default=16_000, gt=0)
    speaker_id: str | None = None
    src_lang: LanguageCode
    tgt_lang: LanguageCode
    src_text: str = Field(min_length=1)
    tgt_text: str = Field(min_length=1)
    target_kind: TargetKind
    domain: Domain
    split: Split
    source_dataset: str
    source_revision: str
    source_license: str
    teacher_id: str | None = None
    teacher_revision: str | None = None
    teacher_license: str | None = None
    generation_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    teacher_generated_at: str | None = None
    teacher_chain_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    materialization_status: MaterializationStatus = "metadata_only"
    filter_decision: FilterStatus = "unreviewed"
    filter_reasons: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_lineage(self) -> ManifestRecord:
        if self.src_lang == self.tgt_lang:
            raise ValueError("source and target languages must differ")
        if self.materialization_status == "ready" and not (self.audio_path and self.audio_sha256):
            raise ValueError("ready records require audio_path and audio_sha256")
        teacher_fields = (
            self.teacher_id,
            self.teacher_revision,
            self.teacher_license,
            self.generation_config_sha256,
            self.teacher_generated_at,
        )
        if self.target_kind == "teacher" and not all(teacher_fields):
            raise ValueError("teacher targets require teacher id, revision, and generation hash")
        if self.target_kind != "teacher" and (any(teacher_fields) or self.teacher_chain_sha256):
            raise ValueError("teacher provenance is only valid for teacher targets")
        if self.target_kind != "teacher" and self.filter_decision != "unreviewed":
            raise ValueError("filter decisions are only valid for teacher targets")
        if self.filter_decision == "reject" and not self.filter_reasons:
            raise ValueError("rejected teacher targets require filter reasons")
        if self.filter_decision != "reject" and self.filter_reasons:
            raise ValueError("filter reasons are only valid for rejected teacher targets")
        return self


def iter_manifest(path: Path) -> Iterator[ManifestRecord]:
    """Stream and validate JSONL records."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield ManifestRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"invalid manifest row {path}:{line_number}: {error}") from error


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> int:
    """Write deterministic UTF-8 JSONL without silently appending stale rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = json.dumps(
                record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            handle.write(payload)
            handle.write("\n")
            count += 1
    return count
